// Copyright © 2026 Apple Inc.
import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXVLM

struct Qwen35VisionPatchSafetyTests {
    private func model(vision: Bool) throws -> Qwen35 {
        let tower = vision ? """
        ,"vision_config":{"model_type":"qwen3_vl","depth":1,"hidden_size":16,
        "intermediate_size":32,"out_hidden_size":32,"num_heads":4,"patch_size":2,
        "spatial_merge_size":2,"temporal_patch_size":1,"num_position_embeddings":32}
        """ : ""
        let json = """
        {"model_type":"qwen3_5_moe","text_config":{"hidden_size":32,"num_hidden_layers":1,
         "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"vocab_size":100,
         "linear_num_value_heads":2,"linear_num_key_heads":2,"linear_key_head_dim":8,
         "linear_value_head_dim":8,"full_attention_interval":1,"num_experts":2,
         "num_experts_per_tok":1,"moe_intermediate_size":32,"shared_expert_intermediate_size":32}\(tower)}
        """
        return Qwen35(try JSONDecoder().decode(Qwen35Configuration.self, from: Data(json.utf8)))
    }
    @Test("slow retained vision sanitizer rejects bad rank and accepts five axes")
    func slowPath() throws {
        let key = "model.visual.patch_embed.proj.weight"
        let m = try model(vision: true)
        #expect(throws: Qwen35.InvalidVisionPatchWeightError.self) {
            try m.validateBeforeSanitize(weights: [key: MLXArray(Float(1))], metadata: [:])
        }
        let input = [key: MLXArray.zeros([16, 3, 1, 2, 2])]
        try m.validateBeforeSanitize(weights: input, metadata: [:])
        let output = m.sanitize(weights: input, metadata: [:])
        #expect(output["vision_tower.patch_embed.proj.weight"]?.shape == [16, 1, 2, 2, 3])
        try model(vision: false).validateBeforeSanitize(weights: [key: MLXArray(Float(1))], metadata: [:])
    }
    @Test("MLX fast path does not invoke vision transpose")
    func fastPath() throws {
        let key = "vision_tower.patch_embed.proj.weight"
        let m = try model(vision: true)
        let input = [key: MLXArray(Float(1))]
        try m.validateBeforeSanitize(weights: input, metadata: ["format": "mlx"])
        #expect(m.sanitize(weights: input, metadata: ["format": "mlx"])[key]?.ndim == 0)
    }
    @Test("actual generic loader rejects malformed slow-path patch weight")
    func loader() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            try MLX.save(arrays: ["model.visual.patch_embed.proj.weight": MLXArray(Float(1))],
                url: directory.appendingPathComponent("model.safetensors"))
            #expect(throws: Qwen35.InvalidVisionPatchWeightError.self) {
                try loadWeights(modelDirectory: directory, model: model(vision: true))
            }
        }
    }
}
