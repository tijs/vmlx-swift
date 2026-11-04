// Copyright © 2026 Apple Inc.
import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXVLM

struct Qwen35ConvSanitizeSafetyTests {
    private func model() throws -> Qwen35 {
        let json = """
        {"model_type":"qwen3_5_moe","text_config":{"hidden_size":32,"num_hidden_layers":1,
         "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"vocab_size":100,
         "linear_num_value_heads":2,"linear_num_key_heads":2,"linear_key_head_dim":8,
         "linear_value_head_dim":8,"full_attention_interval":1,"num_experts":2,
         "num_experts_per_tok":1,"moe_intermediate_size":32,"shared_expert_intermediate_size":32}}
        """
        return Qwen35(try JSONDecoder().decode(Qwen35Configuration.self, from: Data(json.utf8)))
    }
    @Test("actual loader rejects scalar convolution weight before sanitization")
    func scalarLoad() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let key = "language_model.model.layers.0.linear_attn.conv1d.weight"
            try MLX.save(arrays: [key: MLXArray(Float(1))],
                url: directory.appendingPathComponent("model.safetensors"))
            #expect(throws: Qwen35.InvalidConvolutionWeightError.self) {
                try loadWeights(modelDirectory: directory, model: model())
            }
        }
    }
    @Test("valid convolution shapes pass; underspecified unsanitized shapes fail")
    func shapes() throws {
        let m = try model()
        let key = "language_model.model.layers.0.linear_attn.conv1d.weight"
        for shape in [[4, 2, 1], [4, 1, 2]] {
            let weights = [key: MLXArray.zeros(shape)]
            try m.validateBeforeSanitize(weights: weights, metadata: ["format": "mlx"])
            _ = m.sanitize(weights: weights, metadata: ["format": "mlx"])
        }
        for shape in [[2], [2, 2]] {
            #expect(throws: Qwen35.InvalidConvolutionWeightError.self) {
                try m.validateBeforeSanitize(weights: [key: MLXArray.zeros(shape)], metadata: [:])
            }
        }
    }
}
