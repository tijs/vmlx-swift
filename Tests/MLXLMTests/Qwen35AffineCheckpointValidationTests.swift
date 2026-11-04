// Copyright © 2026 Apple Inc.
import Foundation
import CryptoKit
import MLX
import MLXNN
import MLXLMCommon
import Testing
@testable import MLXVLM

struct Qwen35AffineCheckpointValidationTests {
    /// Pad the JSON header only; tensor-relative offsets and payload stay intact.
    private func alignFixtureHeader(_ url: URL) throws {
        let original = try Data(contentsOf: url)
        let length = original.prefix(8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
        let oldEnd = 8 + Int(length)
        var header = original.subdata(in: 8..<oldEnd)
        let entries = try #require(JSONSerialization.jsonObject(with: header) as? [String: Any])
        for (key, raw) in entries where key != "__metadata__" {
            let tensor = try #require(raw as? [String: Any])
            let dtype = try #require(tensor["dtype"] as? String)
            let offsets = try #require(tensor["data_offsets"] as? [Int])
            let alignment = ["U32", "F32"].contains(dtype) ? 4 : 2
            #expect(offsets.count == 2 && offsets[0] % alignment == 0, "relative alignment: \(key)")
            guard offsets.count == 2 && offsets[0] % alignment == 0 else {
                throw NSError(domain: "Qwen35Fixture", code: 1)
            }
        }
        header.append(Data(repeating: 32, count: (4096 - (8 + header.count) % 4096) % 4096))
        var newLength = UInt64(header.count).littleEndian
        var output = Data(bytes: &newLength, count: 8)
        output.append(header)
        output.append(original.suffix(from: oldEnd))
        try output.write(to: url)
        #expect((8 + header.count) % 4096 == 0)
        #expect(output.suffix(from: 8 + header.count) == original.suffix(from: oldEnd))
    }

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
    @Test("complete native raw checkpoint validates; mutations and sidecars fail")
    func complete() throws {
        try MLXMetalTestLock.withLock {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let reference = try model()
            let dense = Dictionary(uniqueKeysWithValues: reference.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) })
            try reference.update(parameters: ModuleParameters.unflattened(dense), verify: .all)
            var updates: [(String, Module)] = []
            for (path, leaf) in reference.leafModules().flattened() {
                if path.contains(".mlp.switch_mlp."), let linear = leaf as? SwitchLinear {
                    updates.append((path, QuantizedSwitchLinear(linear, groupSize: 64, bits: 4)))
                }
            }
            try reference.update(modules: ModuleChildren.unflattened(updates), verify: .none)
            let arrays = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())
            let shard = root.appendingPathComponent("model.safetensors")
            try MLX.save(arrays: arrays, metadata: ["format": "mlx"], url: shard)
            let checkpoint = try MLX.loadArrays(url: shard)
            let validator = try Qwen35AffineCheckpointValidation(model: model(), bits: 4, groupSize: 64,
                directory: root, shardNames: ["model.safetensors"])
            let metadata = ["format": "mlx"]
            try validator.validateSelectedFiles()
            let norm = "language_model.model.layers.0.input_layernorm.weight"
            let expert = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
            for key in [norm, expert] {
                var bad = checkpoint; bad.removeValue(forKey: key)
                #expect(throws: (any Error).self) { try validator.validate(weights: bad, metadata: metadata) }
            }
            var bad = checkpoint; bad[norm] = MLXArray.zeros([1])
            #expect(throws: (any Error).self) { try validator.validate(weights: bad, metadata: metadata) }
            bad = checkpoint; bad[expert] = MLXArray.zeros([1])
            #expect(throws: (any Error).self) { try validator.validate(weights: bad, metadata: metadata) }
            bad = checkpoint; bad["model.language_model.layers.0.input_layernorm.weight"] = checkpoint[norm]
            #expect(throws: (any Error).self) { try validator.validate(weights: bad, metadata: metadata) }
            bad = checkpoint; bad["language_model.model.layers.0.mlp.gate.scales"] = MLXArray.zeros([2, 1])
            #expect(throws: (any Error).self) { try validator.validate(weights: bad, metadata: metadata) }
            bad = checkpoint
            for (key, value) in checkpoint where key.hasSuffix(".input_layernorm.weight") || key.hasSuffix(".post_attention_layernorm.weight") {
                bad[key] = MLXArray.zeros(value.shape).asType(value.dtype)
            }
            #expect(throws: (any Error).self) { try validator.validate(weights: bad, metadata: metadata) }
            #expect(throws: (any Error).self) { try validator.validate(weights: checkpoint, metadata: [:]) }
            try Data().write(to: root.appendingPathComponent("hadamard.json"))
            #expect(throws: (any Error).self) {
                try Qwen35AffineCheckpointValidation(model: model(), bits: 4, groupSize: 64,
                    directory: root, shardNames: ["model.safetensors"])
            }
        }
    }

    @Test("pinned indexed two-shard checkpoint rejects missing and extra files")
    func twoShards() throws {
        try MLXMetalTestLock.withLock {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let reference = try model()
            let dense = Dictionary(uniqueKeysWithValues: reference.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) })
            try reference.update(parameters: ModuleParameters.unflattened(dense), verify: .all)
            var updates: [(String, Module)] = []
            for (path, leaf) in reference.leafModules().flattened() {
                if path.contains(".mlp.switch_mlp."), let linear = leaf as? SwitchLinear {
                    updates.append((path, QuantizedSwitchLinear(linear, groupSize: 64, bits: 4)))
                }
            }
            try reference.update(modules: ModuleChildren.unflattened(updates), verify: .none)
            let arrays = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())
            let names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
            var parts: [[String: MLXArray]] = [[:], [:]]
            var weightMap: [String: String] = [:]
            for (index, key) in arrays.keys.sorted().enumerated() {
                let part = index % 2
                parts[part][key] = arrays[key]
                weightMap[key] = names[part]
            }
            for index in 0..<2 {
                try MLX.save(arrays: parts[index], metadata: ["format": "mlx"],
                    url: root.appendingPathComponent(names[index]))
                try alignFixtureHeader(root.appendingPathComponent(names[index]))
            }
            let indexData = try JSONSerialization.data(withJSONObject: ["weight_map": weightMap], options: [.sortedKeys])
            try indexData.write(to: root.appendingPathComponent("model.safetensors.index.json"))
            let hash = SHA256.hash(data: indexData).map { String(format: "%02x", $0) }.joined()
            let pins = ["model.safetensors.index.json": hash]
            let validator = try Qwen35AffineCheckpointValidation(model: model(), bits: 4, groupSize: 64,
                directory: root, shardNames: Set(names), metadataHashes: pins)
            try validator.validateSelectedFiles()
            switch try validator.validateSelectedDescriptors() {
            case .requiresNormValueCheck: break
            }
            let configuration = try model().config
            let encoded = try JSONEncoder().encode(configuration)
            var object = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
            object["quantization"] = ["bits": 4, "group_size": 64, "mode": "affine"]
            var configData = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
            let configPath = root.appendingPathComponent("config.json")
            try configData.write(to: configPath)
            func configPins(_ data: Data) -> [String: String] {
                ["model.safetensors.index.json": hash,
                 "config.json": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()]
            }
            let withConfig = try Qwen35AffineCheckpointValidation(model: model(), bits: 4, groupSize: 64,
                directory: root, shardNames: Set(names), metadataHashes: configPins(configData))
            try withConfig.validateSelectedFiles()
            object["quantization"] = ["bits": 8, "group_size": 64, "mode": "affine"]
            configData = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
            try configData.write(to: configPath)
            #expect(throws: (any Error).self) {
                try Qwen35AffineCheckpointValidation(model: model(), bits: 4, groupSize: 64,
                    directory: root, shardNames: Set(names), metadataHashes: configPins(configData))
            }
            try FileManager.default.removeItem(at: configPath)
            let badRoot = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: badRoot, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: badRoot) }
            for index in 0..<2 {
                try MLX.save(arrays: parts[index], metadata: ["format": index == 0 ? "mlx" : "other"],
                    url: badRoot.appendingPathComponent(names[index]))
                try alignFixtureHeader(badRoot.appendingPathComponent(names[index]))
            }
            try indexData.write(to: badRoot.appendingPathComponent("model.safetensors.index.json"))
            let badMetadata = try Qwen35AffineCheckpointValidation(model: model(), bits: 4, groupSize: 64,
                directory: badRoot, shardNames: Set(names), metadataHashes: pins)
            #expect(throws: (any Error).self) { try badMetadata.validateSelectedFiles() }
            #expect(throws: (any Error).self) { try badMetadata.validateSelectedDescriptors() }
            let dtypeRoot = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: dtypeRoot, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: dtypeRoot) }
            for name in names { try FileManager.default.copyItem(at: root.appendingPathComponent(name), to: dtypeRoot.appendingPathComponent(name)) }
            try indexData.write(to: dtypeRoot.appendingPathComponent("model.safetensors.index.json"))
            for name in names {
                let url = dtypeRoot.appendingPathComponent(name)
                let data = try Data(contentsOf: url)
                let length = data.prefix(8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
                let end = 8 + Int(length)
                let header = try #require(String(data: data.subdata(in: 8..<end), encoding: .utf8))
                let replaced = header.replacingOccurrences(of: "\"U32\"", with: "\"F32\"")
                var output = Data(data.prefix(8)); output.append(Data(replaced.utf8)); output.append(data.suffix(from: end))
                try output.write(to: url)
            }
            let badDtype = try Qwen35AffineCheckpointValidation(model: model(), bits: 4, groupSize: 64,
                directory: dtypeRoot, shardNames: Set(names), metadataHashes: pins)
            #expect(throws: (any Error).self) { try badDtype.validateSelectedDescriptors() }
            #expect(throws: (any Error).self) {
                try Qwen35AffineCheckpointValidation(model: model(), bits: 4, groupSize: 64,
                    directory: root, shardNames: Set(names), metadataHashes: ["model.safetensors.index.json": String(repeating: "0", count: 64)])
            }
            try Data().write(to: root.appendingPathComponent(".unexpected"))
            #expect(throws: (any Error).self) {
                try Qwen35AffineCheckpointValidation(model: model(), bits: 4, groupSize: 64,
                    directory: root, shardNames: Set(names), metadataHashes: pins)
            }
            try FileManager.default.removeItem(at: root.appendingPathComponent(".unexpected"))
            try FileManager.default.removeItem(at: root.appendingPathComponent(names[1]))
            #expect(throws: (any Error).self) {
                try Qwen35AffineCheckpointValidation(model: model(), bits: 4, groupSize: 64,
                    directory: root, shardNames: Set(names), metadataHashes: pins)
            }
        }
    }
}
