// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Covers `LagunaModel.sanitize`'s absorption of a leading `language_model.`
/// wrapper.
///
/// Some conversions wrap the whole text-only body under a VLM-style
/// `language_model.` prefix — `language_model.model.*` and
/// `language_model.lm_head.*` — while `LagunaModel` binds its modules at the
/// top level. Without the unwrap the loader fails with
/// `Unhandled keys [language_model]`. These tests pin the sanitizer itself
/// (the call site users actually hit), not just the shared
/// `Weights.stripLanguageModelPrefix` helper.
@Suite("LagunaModel.sanitize unwraps a language_model prefix")
struct LagunaSanitizeLanguageModelPrefixTests {

    /// Smallest config that constructs; `sanitize` only routes keys, so the
    /// dims are irrelevant beyond being valid. `tie_word_embeddings: false`
    /// keeps the separate `lm_head` that the wrapped checkpoints ship.
    private static let tinyConfig = #"""
        {
          "model_type": "laguna",
          "hidden_size": 8,
          "intermediate_size": 16,
          "num_hidden_layers": 2,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "num_attention_heads_per_layer": [2, 2],
          "head_dim": 4,
          "max_position_embeddings": 64,
          "vocab_size": 32,
          "rms_norm_eps": 1.0e-5,
          "tie_word_embeddings": false,
          "layer_types": ["full_attention", "sliding_attention"],
          "mlp_layer_types": ["dense", "sparse"],
          "sliding_window": 4,
          "moe_intermediate_size": 4,
          "shared_expert_intermediate_size": 4,
          "num_experts": 4,
          "num_experts_per_tok": 2,
          "gating": "per-head"
        }
        """#

    private func makeModel() throws -> LagunaModel {
        let cfg = try JSONDecoder().decode(
            LagunaConfiguration.self, from: Data(Self.tinyConfig.utf8))
        return LagunaModel(cfg, jangtq: nil)
    }

    /// Distinct extent per tensor so a mis-bind shows up as the wrong shape.
    private func tagged(_ tag: Int) -> MLXArray { MLXArray.zeros([tag]) }

    @Test("a wrapped body unwraps to the module paths after the model. strip")
    func unwrapsWrappedBody() throws {
        try MLXMetalTestLock.withLock {
            let sanitized = try makeModel().sanitize(weights: [
                "language_model.model.embed_tokens.weight": tagged(3),
                "language_model.model.layers.0.input_layernorm.weight": tagged(5),
                "language_model.model.layers.0.self_attn.q_proj.weight": tagged(7),
            ])

            #expect(sanitized["embed_tokens.weight"]?.shape == [3])
            #expect(sanitized["layers.0.input_layernorm.weight"]?.shape == [5])
            #expect(sanitized["layers.0.self_attn.q_proj.weight"]?.shape == [7])
            #expect(sanitized["language_model.model.layers.0.input_layernorm.weight"] == nil)
        }
    }

    @Test("a wrapped routed-expert tensor still hits the switch_mlp normalization")
    func unwrappedExpertsStillNormalize() throws {
        try MLXMetalTestLock.withLock {
            let sanitized = try makeModel().sanitize(weights: [
                "language_model.model.layers.1.mlp.experts.gate_up_proj.weight":
                    MLXArray.ones([4, 8, 2]),
                "language_model.model.layers.1.mlp.experts.e_score_correction_bias":
                    tagged(4),
            ])

            // `language_model.` unwrap → existing `model.` strip → experts →
            // switch_mlp remap → fused gate_up split, all in one pass.
            #expect(sanitized["layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape == [4, 4, 2])
            #expect(sanitized["layers.1.mlp.switch_mlp.up_proj.weight"]?.shape == [4, 4, 2])
            #expect(sanitized["layers.1.mlp.e_score_correction_bias"] != nil)
        }
    }

    @Test("a wrapped lm_head lands on the top-level lm_head")
    func unwrapsWrappedHead() throws {
        try MLXMetalTestLock.withLock {
            let sanitized = try makeModel().sanitize(weights: [
                "language_model.lm_head.weight": tagged(11),
                "language_model.lm_head.scales": tagged(12),
                "language_model.lm_head.biases": tagged(13),
            ])

            #expect(sanitized["lm_head.weight"]?.shape == [11])
            #expect(sanitized["lm_head.scales"]?.shape == [12])
            #expect(sanitized["lm_head.biases"]?.shape == [13])
            #expect(sanitized["language_model.lm_head.weight"] == nil)
        }
    }

    /// Ordinary unwrapped checkpoints — the common case — must be untouched
    /// by the new unwrap.
    @Test("ordinary unwrapped weights pass through unchanged")
    func passesThroughUnwrapped() throws {
        try MLXMetalTestLock.withLock {
            let weights: [String: MLXArray] = [
                "model.embed_tokens.weight": tagged(3),
                "model.layers.0.input_layernorm.weight": tagged(5),
                "lm_head.weight": tagged(11),
            ]
            let sanitized = try makeModel().sanitize(weights: weights)

            #expect(sanitized["embed_tokens.weight"]?.shape == [3])
            #expect(sanitized["layers.0.input_layernorm.weight"]?.shape == [5])
            #expect(sanitized["lm_head.weight"]?.shape == [11])
        }
    }
}
