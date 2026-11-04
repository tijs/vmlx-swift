// Copyright © 2026 Apple Inc.
import Foundation
import CryptoKit
import MLX
import MLXNN
import MLXLMCommon
import Testing
@testable import MLXVLM

struct Qwen35AffineMixedDescriptorTests {
    private func model() throws -> Qwen35 {
        let json = """
        {"model_type":"qwen3_5_moe","text_config":{"hidden_size":64,"num_hidden_layers":2,
         "num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"vocab_size":128,
         "linear_num_value_heads":2,"linear_num_key_heads":2,"linear_key_head_dim":16,
         "linear_value_head_dim":16,"linear_conv_kernel_dim":4,"full_attention_interval":2,
         "num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":64,
         "shared_expert_intermediate_size":64,"tie_word_embeddings":false}}
        """
        return try Qwen35(JSONDecoder().decode(Qwen35Configuration.self, from: Data(json.utf8)), requesting: [.text])
    }
    private func align(_ url: URL) throws {
        let bytes = try Data(contentsOf: url)
        let length = bytes.prefix(8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
        let end = 8 + Int(length)
        var entries = try #require(JSONSerialization.jsonObject(with: bytes.subdata(in: 8..<end)) as? [String: Any])
        var records: [(String, [String: Any], Int, Data)] = []
        for (key, value) in entries where key != "__metadata__" {
            let descriptor = try #require(value as? [String: Any])
            let dtype = try #require(descriptor["dtype"] as? String)
            let offsets = try #require(descriptor["data_offsets"] as? [Int])
            guard offsets.count == 2, offsets[0] >= 0, offsets[0] <= offsets[1],
                offsets[1] <= bytes.count - end, ["U32", "F32", "BF16", "F16"].contains(dtype) else {
                throw NSError(domain: "Qwen35MixedFixture", code: 1)
            }
            let alignment = ["U32", "F32"].contains(dtype) ? 4 : 2
            records.append((key, descriptor, alignment, bytes.subdata(in: (end + offsets[0])..<(end + offsets[1]))))
        }
        records.sort { $0.2 == $1.2 ? $0.0 < $1.0 : $0.2 > $1.2 }
        var payload = Data()
        for (key, original, alignment, value) in records {
            guard payload.count % alignment == 0, value.count % alignment == 0 else {
                throw NSError(domain: "Qwen35MixedFixture", code: 2)
            }
            var descriptor = original
            descriptor["data_offsets"] = [payload.count, payload.count + value.count]
            entries[key] = descriptor
            payload.append(value)
        }
        var header = try JSONSerialization.data(withJSONObject: entries, options: [.sortedKeys])
        header.append(Data(repeating: 32, count: (4096 - (8 + header.count) % 4096) % 4096))
        var size = UInt64(header.count).littleEndian
        var output = Data(bytes: &size, count: 8); output.append(header); output.append(payload)
        for (key, _, _, value) in records {
            let descriptor = try #require(entries[key] as? [String: Any])
            let offsets = try #require(descriptor["data_offsets"] as? [Int])
            #expect(payload.subdata(in: offsets[0]..<offsets[1]) == value, "tensor bytes: \(key)")
        }
        try output.write(to: url)
    }
    private func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    @Test("two-layer mixed 4-bit expert and 8-bit gate headers match pinned policy")
    func mixed() throws {
        try MLXMetalTestLock.withLock {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let reference = try model()
            var dense = Dictionary(uniqueKeysWithValues: reference.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) })
            let normalKey = "language_model.model.layers.0.input_layernorm.weight"
            dense[normalKey] = try #require(dense[normalKey]).asType(.float32)
            try reference.update(parameters: ModuleParameters.unflattened(dense), verify: .all)
            var replacements: [(String, Module)] = []
            for (path, leaf) in reference.leafModules().flattened() {
                if path.contains(".mlp.switch_mlp."), let expert = leaf as? SwitchLinear {
                    replacements.append((path, QuantizedSwitchLinear(expert, groupSize: 64, bits: 4)))
                } else if (path.hasSuffix(".mlp.gate") || path.hasSuffix(".mlp.shared_expert_gate")), let gate = leaf as? Linear {
                    replacements.append((path, QuantizedLinear(gate, groupSize: 64, bits: 8)))
                }
            }
            #expect(replacements.count == 10)
            try reference.update(modules: ModuleChildren.unflattened(replacements), verify: .none)
            let arrays = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())
            let files = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
            var shards: [[String: MLXArray]] = [[:], [:]]
            var index: [String: String] = [:]
            for (i, key) in arrays.keys.sorted().enumerated() { shards[i % 2][key] = arrays[key]; index[key] = files[i % 2] }
            for i in 0..<2 {
                let url = root.appendingPathComponent(files[i])
                try MLX.save(arrays: shards[i], metadata: ["format": "mlx"], url: url)
                try align(url)
            }
            let indexData = try JSONSerialization.data(withJSONObject: ["weight_map": index], options: [.sortedKeys])
            try indexData.write(to: root.appendingPathComponent("model.safetensors.index.json"))
            var quant: [String: Any] = ["bits": 4, "group_size": 64, "mode": "affine"]
            for layer in 0..<2 { for leaf in ["gate", "shared_expert_gate"] {
                quant["language_model.model.layers.\(layer).mlp.\(leaf)"] = ["bits": 8, "group_size": 64]
            }}
            let encoded = try JSONEncoder().encode(reference.config)
            var config = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
            config["quantization"] = quant; config["quantization_config"] = quant
            var configData = try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            let configURL = root.appendingPathComponent("config.json")
            try configData.write(to: configURL)
            func pins(_ data: Data) -> [String: String] {
                ["config.json": digest(data), "model.safetensors.index.json": digest(indexData)]
            }
            let validator = try Qwen35AffineCheckpointValidation(model: model(), bits: 4, groupSize: 64,
                directory: root, shardNames: Set(files), metadataHashes: pins(configData))
            switch try validator.validateSelectedDescriptors() { case .requiresNormValueCheck: break }
            let loaded = try model()
            try loadWeights(modelDirectory: root, model: loaded,
                quantization: .init(groupSize: 64, bits: 4))
            let loadedParameters = Dictionary(uniqueKeysWithValues: loaded.parameters().flattened())
            #expect(try #require(loadedParameters[normalKey]).dtype == .bfloat16)
            var f32GateWeights: [String: MLXArray] = [:]
            for layer in 0..<2 { for leaf in ["gate", "shared_expert_gate"] {
                let path = "language_model.model.layers.\(layer).mlp.\(leaf)"
                let packed = try #require(arrays[path + ".weight"])
                let scales = try #require(arrays[path + ".scales"])
                let biases = try #require(arrays[path + ".biases"])
                let expected = MLX.dequantized(packed, scales: scales, biases: biases,
                    groupSize: 64, bits: 8).asType(.float32)
                let actual = try #require(loadedParameters[path + ".weight"])
                f32GateWeights[path + ".weight"] = expected
                #expect(actual.dtype == .bfloat16, "current loaded gate storage precision: \(path)")
                #expect(MLX.arrayEqual(actual, expected.asType(.bfloat16)).item(Bool.self),
                    "loaded gate dequantization: \(path)")
            }}
            let modules = Dictionary(uniqueKeysWithValues: loaded.namedModules())
            let moe = try #require(modules["language_model.model.layers.0.mlp"] as? Qwen35Language.SparseMoeBlock)
            let input = MLXArray.ones([1, 2, 64]).asType(.bfloat16)
            let tokens = MLXArray([1, 2]).reshaped([1, 2])
            let bfGate = moe.gate(input)
            let bfSharedGate = moe.sharedExpertGate(input)
            let bfMoe = moe(input)
            let bfLogits = loaded(tokens, cache: loaded.newCache(parameters: nil))
            MLX.eval(bfGate, bfSharedGate, bfMoe, bfLogits)
            try loaded.update(parameters: ModuleParameters.unflattened(f32GateWeights), verify: .noUnusedKeys)
            let fpGate = moe.gate(input)
            let fpSharedGate = moe.sharedExpertGate(input)
            let fpMoe = moe(input)
            let fpLogits = loaded(tokens, cache: loaded.newCache(parameters: nil))
            MLX.eval(fpGate, fpSharedGate, fpMoe, fpLogits)
            let maxDifference = MLX.max(MLX.abs(fpLogits.asType(.float32) - bfLogits.asType(.float32))).item(Float.self)
            FileHandle.standardError.write(Data(("[Qwen35SyntheticGatePrecision] bf_gate=\(bfGate.dtype) fp_gate=\(fpGate.dtype) bf_shared_gate=\(bfSharedGate.dtype) fp_shared_gate=\(fpSharedGate.dtype) bf_moe=\(bfMoe.dtype) fp_moe=\(fpMoe.dtype) bf_logits=\(bfLogits.dtype) fp_logits=\(fpLogits.dtype) max_logits_difference=\(maxDifference)\n").utf8))
            var missing = arrays
            missing.removeValue(forKey: "language_model.model.layers.0.mlp.gate.scales")
            #expect(throws: (any Error).self) { try validator.validate(weights: missing, metadata: ["format": "mlx"]) }
            var malformed = arrays
            malformed["language_model.model.layers.0.mlp.gate.weight"] = MLXArray.zeros([2, 1])
            #expect(throws: (any Error).self) { try validator.validate(weights: malformed, metadata: ["format": "mlx"]) }
            quant["language_model.model.layers.0.mlp.gate"] = ["bits": 6, "group_size": 64]
            config["quantization"] = quant; config["quantization_config"] = quant
            configData = try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            try configData.write(to: configURL)
            #expect(throws: (any Error).self) {
                try Qwen35AffineCheckpointValidation(model: model(), bits: 4, groupSize: 64,
                    directory: root, shardNames: Set(files), metadataHashes: pins(configData))
            }
        }
    }
}
