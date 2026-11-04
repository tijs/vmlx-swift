// Copyright © 2026 Apple Inc.
import Foundation
import MLX
import MLXNN
import MLXLMCommon
import Testing
@testable import MLXVLM

/// Tiny synthetic full-resident checkpoint: no research exclusion or serving.
struct Qwen35AffineFullLoaderFixtureTests {
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
    private func save(_ arrays: [String: MLXArray], in directory: URL) throws {
        try MLX.save(arrays: arrays, metadata: ["format": "mlx"],
            url: directory.appendingPathComponent("model.safetensors"))
    }
    @Test("full generic load preserves tiny hybrid checkpoint and exposes missing resident weights")
    func fullCheckpoint() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let reference = try model()
            let normKey = "language_model.model.layers.0.input_layernorm.weight"
            var dense: [String: MLXArray] = [:]
            for (key, value) in reference.parameters().flattened() {
                dense[key] = (key == normKey ? value * 2 : value).asType(.bfloat16)
            }
            try reference.update(parameters: ModuleParameters.unflattened(dense), verify: .all)
            var replacements: [(String, Module)] = []
            for (path, leaf) in reference.leafModules().flattened() {
                if path.contains(".mlp.switch_mlp."), let linear = leaf as? SwitchLinear {
                    replacements.append((path, QuantizedSwitchLinear(linear, groupSize: 64, bits: 4)))
                }
            }
            #expect(replacements.count == 6)
            try reference.update(modules: ModuleChildren.unflattened(replacements), verify: .none)
            var checkpoint: [String: MLXArray] = [:]
            for (key, value) in reference.parameters().flattened() { checkpoint[key] = value }
            MLX.eval(Array(checkpoint.values))
            try save(checkpoint, in: directory)
            let loaded = try model()
            try loadWeights(modelDirectory: directory, model: loaded,
                quantization: .init(groupSize: 64, bits: 4))
            let actual = Dictionary(uniqueKeysWithValues: loaded.parameters().flattened())
            #expect(Set(actual.keys) == Set(checkpoint.keys))
            for (key, expected) in checkpoint {
                guard let value = actual[key] else { Issue.record("missing parameter \(key)"); continue }
                #expect(value.shape == expected.shape, "shape \(key)")
                #expect(value.dtype == expected.dtype, "dtype \(key)")
                #expect(MLX.arrayEqual(value, expected).item(Bool.self), "value \(key)")
            }
            let tokens = MLXArray([1, 2]).reshaped([1, 2])
            let a = reference(tokens, cache: reference.newCache(parameters: nil))
            let b = loaded(tokens, cache: loaded.newCache(parameters: nil))
            MLX.eval(a, b)
            #expect(MLX.arrayEqual(a, b).item(Bool.self))

            // Current default loader only checks unused supplied keys. It
            // accepts this missing resident norm and leaves initialization in place.
            var incomplete = checkpoint
            incomplete.removeValue(forKey: normKey)
            let incompleteDirectory = directory.appendingPathComponent("incomplete")
            try FileManager.default.createDirectory(at: incompleteDirectory, withIntermediateDirectories: true)
            try save(incomplete, in: incompleteDirectory)
            let missing = try model()
            try loadWeights(modelDirectory: incompleteDirectory, model: missing,
                quantization: .init(groupSize: 64, bits: 4))
            let missingParameters = Dictionary(uniqueKeysWithValues: missing.parameters().flattened())
            let retained = try #require(missingParameters[normKey])
            let expectedNorm = try #require(checkpoint[normKey])
            #expect(!MLX.arrayEqual(retained, expectedNorm).item(Bool.self))
        }
    }
}
