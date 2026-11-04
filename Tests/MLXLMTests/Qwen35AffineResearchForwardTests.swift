// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

@Suite("experimental Qwen35 VLM affine MoE forward", .serialized)
struct Qwen35AffineResearchForwardTests {
    private let prefix = "language_model.model.layers.0.mlp.switch_mlp."
    private let limits = AffineStreamingSwitchGLU.ResearchLimits(
        maxTokens: 65_536, maxRoutesPerToken: 4, maxWindowTokens: 128,
        maxGroupSize: 32, maxMappedExpertBytes: 2 * 1024 * 1024,
        maxRetainedOutputBytes: 128 * 1024 * 1024)

    private func fixture() throws -> (URL, [String: MLXArray]) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen35-vlm-affine-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        do {
            var tensors: [String: MLXArray] = [:]
            for (projection, name) in ["gate_proj", "up_proj", "down_proj"].enumerated() {
                var weights: [MLXArray] = []
                var scales: [MLXArray] = []
                var biases: [MLXArray] = []
                for expert in 0 ..< 4 {
                    let values = (0 ..< 64 * 64).map { i in
                        Float((i * 13 + expert * 17 + projection * 23) % 101 - 50) / 200
                    }
                    let q = MLX.quantized(MLXArray(values, [64, 64]),
                        groupSize: 64, bits: 4, mode: .affine)
                    weights.append(q.wq)
                    scales.append(q.scales.asType(.bfloat16))
                    biases.append(q.biases!.asType(.bfloat16))
                }
                tensors[prefix + name + ".weight"] = stacked(weights, axis: 0)
                tensors[prefix + name + ".scales"] = stacked(scales, axis: 0)
                tensors[prefix + name + ".biases"] = stacked(biases, axis: 0)
            }
            MLX.eval(Array(tensors.values))
            let shard = "model.safetensors"
            let shardURL = directory.appendingPathComponent(shard)
            try MLX.save(arrays: tensors, url: shardURL)
            let saved = try Data(contentsOf: shardURL)
            let oldLength = saved.prefix(8).withUnsafeBytes {
                $0.loadUnaligned(as: UInt64.self).littleEndian
            }
            var header = saved.subdata(in: 8 ..< (8 + Int(oldLength)))
            let padding = (4096 - (8 + header.count) % 4096) % 4096
            header.append(Data(repeating: 0x20, count: padding))
            var rewritten = Data()
            var headerLength = UInt64(header.count).littleEndian
            rewritten.append(contentsOf: withUnsafeBytes(of: &headerLength) { Array($0) })
            rewritten.append(header)
            rewritten.append(saved.suffix(from: 8 + Int(oldLength)))
            try rewritten.write(to: shardURL)
            let map = Dictionary(uniqueKeysWithValues: tensors.keys.map { ($0, shard) })
            try JSONSerialization.data(withJSONObject: ["weight_map": map], options: [.sortedKeys])
                .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
            return (directory, tensors)
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    private func makeBlock(_ tensors: [String: MLXArray], normalize: Bool,
                           compiled: Bool = false) throws -> Qwen35Language.SparseMoeBlock {
        let json = """
            {"hidden_size":64,"num_experts":4,"num_experts_per_tok":2,
             "moe_intermediate_size":64,"shared_expert_intermediate_size":64,
             "norm_topk_prob":\(normalize)}
            """
        let config = try JSONDecoder().decode(Qwen35Configuration.TextConfiguration.self,
            from: Data(json.utf8))
        let block = Qwen35Language.SparseMoeBlock(config, layerIdx: 0,
            allowFusedGateUpCache: false, compileDecodeRegions: compiled)
        guard let resident = block.switchMLP as? SwitchGLU else {
            throw AffineStreamingExpertError.invalidInput("fixture requires resident affine switch")
        }
        func projection(_ name: String) throws -> QuantizedSwitchLinear {
            QuantizedSwitchLinear(inputDims: 64, outputDims: 64, numExperts: 4,
                weight: try #require(tensors[prefix + name + ".weight"]),
                scales: try #require(tensors[prefix + name + ".scales"]),
                biases: try #require(tensors[prefix + name + ".biases"]),
                groupSize: 64, bits: 4, mode: .affine)
        }
        try resident.update(modules: ModuleChildren.unflattened([
            ("gate_proj", try projection("gate_proj")),
            ("up_proj", try projection("up_proj")),
            ("down_proj", try projection("down_proj")),
        ]), verify: .all)
        return block
    }

    @Test("eager router and shared expert match resident block; mixed dtype and cancellation reject")
    func parityAndFailures() throws {
        try MLXMetalTestLock.withLock {
            let (directory, tensors) = try fixture()
            defer { try? FileManager.default.removeItem(at: directory) }
            let catalog = AffineStreamingExpertCatalog(numExperts: 4, inputDims: 64,
                hiddenDims: 64, layout: .qwen35)
            try catalog.configure(modelDirectory: directory, layerCount: 1)
            let reducer = AffineStreamingSwitchGLU(inputDims: 64, hiddenDims: 64,
                numExperts: 4, layerIndex: 0, catalog: catalog)
            for normalize in [false, true] {
                let block = try makeBlock(tensors, normalize: normalize)
                let probe = MLXArray([Float](repeating: 0.25, count: 64), [1, 1, 64]).asType(.bfloat16)
                #expect(block.gate(probe).dtype == .float32)
                do {
                    _ = try block.researchAffineForward(probe, reducer: reducer, limits: limits,
                        groupLimit: 16, tokenWindow: 8)
                    Issue.record("mixed router dtype was accepted")
                } catch AffineStreamingExpertError.invalidInput(let reason) {
                    #expect(reason.contains("mixed router/activation dtypes"))
                } catch { Issue.record("unexpected error: \(error)") }
                try block.gate.update(parameters: ModuleParameters.unflattened([
                    "weight": block.gate.weight.asType(.bfloat16)
                ]), verify: .all)
                for tokens in [1, 8, 33] {
                    let values = (0 ..< tokens * 64).map { Float(($0 * 17) % 101 - 50) / 37 }
                    let x = MLXArray(values, [1, tokens, 64]).asType(.bfloat16)
                    #expect(block.gate(x).dtype == .bfloat16)
                    let expected = block(x)
                    let actual = try block.researchAffineForward(x, reducer: reducer, limits: limits,
                        groupLimit: 16, tokenWindow: 8)
                    MLX.eval(expected, actual)
                    let maxError = MLX.max(MLX.abs(expected.asType(.float32)
                        - actual.asType(.float32))).item(Float.self)
                    print("QWEN35_VLM_AFFINE_PARITY norm=\(normalize) tokens=\(tokens) max_abs=\(maxError)")
                    #expect(actual.dtype == expected.dtype && actual.shape == expected.shape)
                    #expect(MLX.arrayEqual(actual, expected).item(Bool.self),
                        "norm=\(normalize) tokens=\(tokens) max_abs=\(maxError)")
                }
                #expect(throws: CancellationError.self) {
                    try block.researchAffineForward(probe, reducer: reducer, limits: limits,
                        groupLimit: 16, tokenWindow: 8, shouldCancel: { true })
                }
            }
            let unsupported = try makeBlock(tensors, normalize: true, compiled: true)
            let x = MLXArray([Float](repeating: 0, count: 64), [1, 1, 64]).asType(.bfloat16)
            #expect(throws: AffineStreamingExpertError.self) {
                try unsupported.researchAffineForward(x, reducer: reducer, limits: limits,
                    groupLimit: 16, tokenWindow: 8)
            }
            let block = try makeBlock(tensors, normalize: true)
            let wrongLayer = AffineStreamingSwitchGLU(inputDims: 64, hiddenDims: 64,
                numExperts: 4, layerIndex: 1, catalog: catalog)
            let wrongHidden = AffineStreamingSwitchGLU(inputDims: 64, hiddenDims: 32,
                numExperts: 4, layerIndex: 0, catalog: catalog)
            for wrong in [wrongLayer, wrongHidden] {
                #expect(throws: AffineStreamingExpertError.self) {
                    try block.researchAffineForward(x, reducer: wrong, limits: limits,
                        groupLimit: 16, tokenWindow: 8)
                }
            }
        }
    }
}
