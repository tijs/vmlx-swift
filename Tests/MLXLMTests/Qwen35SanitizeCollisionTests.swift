// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXVLM

@Suite("Qwen35 sanitizer collision validation", .serialized)
struct Qwen35SanitizeCollisionTests {
    private func model(tied: Bool = false, mtp: Int = 0, vision: Bool = false) throws -> Qwen35 {
        let visionJSON = vision ? """
        ,"vision_config":{"model_type":"qwen3_vl","depth":1,"hidden_size":16,
        "intermediate_size":32,"out_hidden_size":32,"num_heads":4,"patch_size":2,
        "spatial_merge_size":2,"temporal_patch_size":1,"num_position_embeddings":32}
        """ : ""
        let json = """
        {"model_type":"qwen3_5_moe","text_config":{"hidden_size":32,"num_hidden_layers":1,
        "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"vocab_size":100,
        "linear_num_value_heads":4,"linear_num_key_heads":2,"linear_key_head_dim":8,
        "linear_value_head_dim":8,"linear_conv_kernel_dim":2,"full_attention_interval":1,
        "num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":32,
        "shared_expert_intermediate_size":32,"tie_word_embeddings":\(tied),
        "mtp_num_hidden_layers":\(mtp)}\(visionJSON)}
        """
        return Qwen35(try JSONDecoder().decode(Qwen35Configuration.self, from: Data(json.utf8)))
    }

    private func weights(_ keys: [String]) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: keys.map { ($0, MLXArray([Float(1)])) })
    }

    @Test("language and head aliases fail instead of overwriting")
    func languageAndHeadAliases() throws {
        let m = try model()
        let aliases = weights(["model.language_model.layers.0.weight", "language_model.model.layers.0.weight"])
        do {
            try m.validateBeforeSanitize(weights: aliases, metadata: [:])
            Issue.record("expected alias collision")
        } catch let error as Qwen35.SanitizationCollisionError {
            #expect(error.destination == "language_model.model.layers.0.weight")
            #expect(error.sources.count == 2)
        }
        #expect(throws: Qwen35.SanitizationCollisionError.self) {
            try m.validateBeforeSanitize(weights: weights(["lm_head.weight", "language_model.lm_head.weight"]), metadata: [:])
        }
        // The tied head removes the unprefixed source before remapping.
        try model(tied: true).validateBeforeSanitize(
            weights: weights(["lm_head.weight", "language_model.lm_head.weight"]), metadata: [:])
    }

    @Test("vision aliases count only when the tower is retained")
    func visionAliases() throws {
        let aliases = weights(["model.visual.block.weight", "vision_tower.block.weight"])
        try model().validateBeforeSanitize(weights: aliases, metadata: [:])
        #expect(throws: Qwen35.SanitizationCollisionError.self) {
            try model(vision: true).validateBeforeSanitize(weights: aliases, metadata: [:])
        }
    }

    @Test("MTP aliases count only when native MTP is enabled")
    func mtpAliases() throws {
        let aliases = weights(["mtp.layers.0.weight", "language_model.mtp.layers.0.weight"])
        try model().validateBeforeSanitize(weights: aliases, metadata: [:])
        #expect(throws: Qwen35.SanitizationCollisionError.self) {
            try model(mtp: 1).validateBeforeSanitize(weights: aliases, metadata: [:])
        }
    }

    @Test("MLX fast path leaves head spelling unchanged without other remapping")
    func mlxFastPath() throws {
        let aliases = weights(["lm_head.weight", "language_model.lm_head.weight"])
        try model().validateBeforeSanitize(weights: aliases,
            metadata: ["format": "mlx", "norm_convention": "shifted"])
    }

    @Test("loader invokes collision validation before model update")
    func loaderRejectsCollision() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let aliases = weights(["model.language_model.layers.0.weight", "language_model.model.layers.0.weight"])
            try MLX.save(arrays: aliases, url: directory.appendingPathComponent("model.safetensors"))
            #expect(throws: Qwen35.SanitizationCollisionError.self) {
                try loadWeights(modelDirectory: directory, model: model())
            }
        }
    }
}
