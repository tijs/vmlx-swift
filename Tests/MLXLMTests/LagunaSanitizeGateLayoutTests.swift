// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Covers `LagunaModel.sanitize`'s normalization of the actual router gate
/// tensor layout on current affine bundles.
///
/// The real checkpoint nests the quantized router under
/// `layers.N.mlp.gate.proj.{weight,scales,biases}` and stores the sigmoid
/// correction bias at `layers.N.mlp.gate.e_score_correction_bias`, while
/// `LagunaMoE` declares `gate` as a plain `Linear` (bound at
/// `mlp.gate.{weight}`) with `e_score_correction_bias` as a sibling
/// parameter. Without the remap the loader fails with
/// `Unhandled keys ["e_score_correction_bias", "proj"] in layers.1.mlp.gate`.
///
/// These tests pin sanitize's two rewrites:
///   1. `mlp.gate.proj.` → `mlp.gate.` (flatten the quantized wrapper so
///      Load.swift's `.gate.{weight,scales,biases}` dequantization applies)
///   2. `mlp.gate.e_score_correction_bias` → `mlp.e_score_correction_bias`
///      (hoist to the sibling slot `LagunaMoE` binds)
/// and assert ordinary unwrapped / already-flattened paths stay untouched.
@Suite("LagunaModel.sanitize normalizes the router gate layout")
struct LagunaSanitizeGateLayoutTests {

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

    @Test("a wrapped gate.proj quant router flattens to gate.*")
    func flattensWrappedGateProj() throws {
        try MLXMetalTestLock.withLock {
            let sanitized = try makeModel().sanitize(weights: [
                "language_model.model.layers.1.mlp.gate.proj.weight": tagged(11),
                "language_model.model.layers.1.mlp.gate.proj.scales": tagged(12),
                "language_model.model.layers.1.mlp.gate.proj.biases": tagged(13),
            ])

            #expect(sanitized["layers.1.mlp.gate.weight"]?.shape == [11])
            #expect(sanitized["layers.1.mlp.gate.scales"]?.shape == [12])
            #expect(sanitized["layers.1.mlp.gate.biases"]?.shape == [13])
            #expect(sanitized["layers.1.mlp.gate.proj.weight"] == nil)
        }
    }

    @Test("a wrapped gate correction bias hoists to the mlp sibling slot")
    func hoistsWrappedGateBias() throws {
        try MLXMetalTestLock.withLock {
            let sanitized = try makeModel().sanitize(weights: [
                "language_model.model.layers.1.mlp.gate.e_score_correction_bias": tagged(4)
            ])

            #expect(sanitized["layers.1.mlp.e_score_correction_bias"]?.shape == [4])
            #expect(sanitized["layers.1.mlp.gate.e_score_correction_bias"] == nil)
        }
    }

    @Test("an unwrapped gate.proj quant router flattens the same way")
    func flattensUnwrappedGateProj() throws {
        try MLXMetalTestLock.withLock {
            let sanitized = try makeModel().sanitize(weights: [
                "model.layers.1.mlp.gate.proj.weight": tagged(11),
                "model.layers.1.mlp.gate.proj.scales": tagged(12),
                "model.layers.1.mlp.gate.proj.biases": tagged(13),
            ])

            #expect(sanitized["layers.1.mlp.gate.weight"]?.shape == [11])
            #expect(sanitized["layers.1.mlp.gate.scales"]?.shape == [12])
            #expect(sanitized["layers.1.mlp.gate.biases"]?.shape == [13])
        }
    }

    @Test("an already-flattened gate layout passes through unchanged")
    func preservesFlattenedGate() throws {
        try MLXMetalTestLock.withLock {
            let sanitized = try makeModel().sanitize(weights: [
                "model.layers.1.mlp.gate.weight": tagged(11),
                "model.layers.1.mlp.gate.scales": tagged(12),
                "model.layers.1.mlp.gate.biases": tagged(13),
                "model.layers.1.mlp.e_score_correction_bias": tagged(4),
            ])

            #expect(sanitized["layers.1.mlp.gate.weight"]?.shape == [11])
            #expect(sanitized["layers.1.mlp.gate.scales"]?.shape == [12])
            #expect(sanitized["layers.1.mlp.gate.biases"]?.shape == [13])
            #expect(sanitized["layers.1.mlp.e_score_correction_bias"]?.shape == [4])
        }
    }

    @Test("expert and shared-expert projections are untouched by the gate remaps")
    func preservesExpertAndSharedPaths() throws {
        try MLXMetalTestLock.withLock {
            let sanitized = try makeModel().sanitize(weights: [
                "model.layers.1.mlp.switch_mlp.gate_proj.weight": tagged(21),
                "model.layers.1.mlp.switch_mlp.up_proj.scales": tagged(22),
                "model.layers.1.mlp.switch_mlp.down_proj.biases": tagged(23),
                "model.layers.1.mlp.shared_expert.gate_proj.weight": tagged(31),
                "model.layers.1.mlp.shared_expert.up_proj.weight": tagged(32),
                "model.layers.1.mlp.shared_expert.down_proj.weight": tagged(33),
            ])

            // `gate_proj` (expert MLP) and `shared_expert` must survive the
            // `.mlp.gate.proj.` / `.mlp.gate.e_score_correction_bias` rewrites
            // byte-for-byte.
            #expect(sanitized["layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape == [21])
            #expect(sanitized["layers.1.mlp.switch_mlp.up_proj.scales"]?.shape == [22])
            #expect(sanitized["layers.1.mlp.switch_mlp.down_proj.biases"]?.shape == [23])
            #expect(sanitized["layers.1.mlp.shared_expert.gate_proj.weight"]?.shape == [31])
            #expect(sanitized["layers.1.mlp.shared_expert.up_proj.weight"]?.shape == [32])
            #expect(sanitized["layers.1.mlp.shared_expert.down_proj.weight"]?.shape == [33])
        }
    }

    @Test("a hypothetical gate.proj.e_score_correction_bias still hoists")
    func hoistsNestedBiasVariant() throws {
        try MLXMetalTestLock.withLock {
            let sanitized = try makeModel().sanitize(weights: [
                "model.layers.1.mlp.gate.proj.e_score_correction_bias": tagged(4)
            ])

            // Flatten runs first (`gate.proj.` → `gate.`), then the bias remap
            // hoists the result to the sibling slot.
            #expect(sanitized["layers.1.mlp.e_score_correction_bias"]?.shape == [4])
            #expect(sanitized["layers.1.mlp.gate.e_score_correction_bias"] == nil)
            #expect(sanitized["layers.1.mlp.gate.proj.e_score_correction_bias"] == nil)
        }
    }
}
