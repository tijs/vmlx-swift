// Copyright © 2026 Apple Inc.
import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXVLM

struct Qwen35FusedExpertSafetyTests {
    private let key = "language_model.model.layers.0.mlp.experts.gate_up_proj"
    private func model() throws -> Qwen35MoE {
        let json = """
        {"model_type":"qwen3_5_moe","text_config":{"hidden_size":32,"num_hidden_layers":1,
         "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"vocab_size":100,
         "linear_num_value_heads":2,"linear_num_key_heads":2,"linear_key_head_dim":8,
         "linear_value_head_dim":8,"full_attention_interval":1,"num_experts":2,
         "num_experts_per_tok":1,"moe_intermediate_size":32,"shared_expert_intermediate_size":32}}
        """
        return Qwen35MoE(try JSONDecoder().decode(Qwen35Configuration.self, from: Data(json.utf8)))
    }
    @Test("loader rejects scalar and odd fused expert split before sanitizing")
    func rejectsMalformed() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            for (index, tensor) in [MLXArray(Float(1)), MLXArray.zeros([2, 3, 32])].enumerated() {
                let item = directory.appendingPathComponent("item-\(index)")
                try FileManager.default.createDirectory(at: item, withIntermediateDirectories: true)
                try MLX.save(arrays: [key: tensor], url: item.appendingPathComponent("model.safetensors"))
                #expect(throws: Qwen35MoE.InvalidFusedExpertWeightError.self) {
                    try loadWeights(modelDirectory: item, model: model())
                }
            }
        }
    }
    @Test("even fused expert split remains supported")
    func validSplit() throws {
        let m = try model()
        let weights = [key: MLXArray.zeros([2, 64, 32])]
        try m.validateBeforeSanitize(weights: weights, metadata: ["format": "mlx"])
        let sanitized = m.sanitize(weights: weights, metadata: ["format": "mlx"])
        #expect(sanitized["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape == [2, 32, 32])
        #expect(sanitized["language_model.model.layers.0.mlp.switch_mlp.up_proj.weight"]?.shape == [2, 32, 32])
    }
}
