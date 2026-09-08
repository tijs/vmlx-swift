// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Pins the `LagunaMoE` gate that enables `SwitchGLU(compileSeparatedDecode:)`
/// for affine (non-codebook) Laguna bundles.
///
/// `compileSeparatedDecode: true` engages the validated
/// `Qwen4ExpCompiledRoutedSwitchGLU` trusted region — a single fused
/// gate/up/silu/down decode trace — the same default enablement Qwen3.5 18B
/// already ships for its identical routed-MoE geometry. The region is
/// guarded at call time and falls back to the generic
/// three-`gatherQuantizedMM` path on any mismatch, but the *default* is
/// deliberately reserved for the verified affine S-2.1 XS archetype so other
/// Laguna variants keep their historical eager path. The
/// `VMLX_LAGUNA_COMPILE_DECODE_REGIONS` env flag overrides the default.
///
/// These tests run without the 23 GB checkpoint: the gate matrix is pure
/// config logic, and the parity case exercises the compiled-region flag on a
/// tiny bf16 affine SwitchGLU (no model weights, no GPU-heavy geometry).
@Suite("LagunaMoE compiled decode region gate")
struct LagunaCompileDecodeRegionsTests {

    /// Decodes a full `LagunaConfiguration` with the S-2.1 XS affine
    /// signature unless a field is overridden. The hybrid attention layout
    /// mirrors the bundle: 10 full-attention + 30 sliding-attention layers.
    private func decode(
        hiddenSize: Int = 2048,
        moeIntermediate: Int = 512,
        layers: Int = 40,
        experts: Int = 256,
        topK: Int = 8
    ) throws -> LagunaConfiguration {
        let fullCount = min(10, layers)
        let slidingCount = layers - fullCount
        let heads =
            Array(repeating: 48, count: fullCount)
            + Array(repeating: 72, count: slidingCount)
        let json: [String: Any] = [
            "model_type": "laguna",
            "hidden_size": hiddenSize,
            "intermediate_size": hiddenSize * 4,
            "num_hidden_layers": layers,
            "num_attention_heads": 48,
            "num_key_value_heads": 8,
            "num_attention_heads_per_layer": heads,
            "head_dim": 128,
            "max_position_embeddings": 131072,
            "vocab_size": 100352,
            "rms_norm_eps": 1.0e-6,
            "tie_word_embeddings": false,
            "layer_types": Array(repeating: "full_attention", count: fullCount)
                + Array(repeating: "sliding_attention", count: slidingCount),
            "mlp_layer_types": ["dense"]
                + Array(repeating: "sparse", count: max(0, layers - 1)),
            "sliding_window": 512,
            "moe_intermediate_size": moeIntermediate,
            "shared_expert_intermediate_size": moeIntermediate,
            "num_experts": experts,
            "num_experts_per_tok": topK,
            "gating": "per-head",
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        return try JSONDecoder().decode(LagunaConfiguration.self, from: data)
    }

    @Test("affine S-2.1 XS signature enables the region by default")
    func defaultsOnForAffineS21XS() throws {
        let cfg = try decode()
        #expect(
            LagunaMoE.shouldCompileSeparatedDecode(cfg, jangtq: nil)
                == true)
    }

    @Test("the codebook (mxtq) path never enables the region")
    func staysOffForCodebookBundles() throws {
        let cfg = try decode()
        #expect(
            LagunaMoE.shouldCompileSeparatedDecode(
                cfg, jangtq: LagunaMoEContext(bits: 2, mxtqSeed: 42))
                == false)
    }

    @Test("non-archetype geometries keep the generic path by default")
    func staysOffForOtherGeometries() throws {
        let variants: [(label: String, cfg: LagunaConfiguration)] = [
            ("larger hidden", try decode(hiddenSize: 2560)),
            ("larger moe", try decode(moeIntermediate: 640)),
            ("fewer experts", try decode(experts: 128)),
            ("fewer layers", try decode(layers: 32)),
            ("higher top-k", try decode(topK: 16)),
            ("smaller hidden and moe", try decode(hiddenSize: 1024, moeIntermediate: 256)),
        ]
        for variant in variants {
            #expect(
                LagunaMoE.shouldCompileSeparatedDecode(variant.cfg, jangtq: nil)
                    == false,
                "expected generic path for \\(variant.label)")
        }
    }

    @Test("env flag off beats the archetype default")
    func envOffBeatsDefault() throws {
        let cfg = try decode()
        for raw in ["0", "false", "off", "no"] {
            #expect(
                LagunaMoE.shouldCompileSeparatedDecode(
                    cfg, jangtq: nil,
                    environment: ["VMLX_LAGUNA_COMPILE_DECODE_REGIONS": raw])
                    == false,
                "expected env \\(raw) to disable the region")
        }
    }

    @Test("env flag on forces the region for a non-archetype geometry")
    func envOnForcesNonArchetype() throws {
        let cfg = try decode(hiddenSize: 2560)
        #expect(
            LagunaMoE.shouldCompileSeparatedDecode(
                cfg, jangtq: nil,
                environment: ["VMLX_LAGUNA_COMPILE_DECODE_REGIONS": "1"])
                == true)
    }

    @Test("the legacy VMLINUX_ prefix spelling is honoured")
    func legacyPrefixHonoured() throws {
        let cfg = try decode(hiddenSize: 2560)
        #expect(
            LagunaMoE.shouldCompileSeparatedDecode(
                cfg, jangtq: nil,
                environment: ["VMLINUX_LAGUNA_COMPILE_DECODE_REGIONS": "1"])
                == true)
    }

    /// Builds a tiny bf16 affine SwitchGLU whose projections are replaced
    /// with the supplied quantized modules — the same loader mechanism
    /// (`Module.update(modules:)`) real checkpoints use.
    private func tinySwitchGLU(
        compileSeparatedDecode: Bool,
        gate: QuantizedSwitchLinear, up: QuantizedSwitchLinear,
        down: QuantizedSwitchLinear
    ) throws -> SwitchGLU {
        let glu = SwitchGLU(
            inputDims: 64, hiddenDims: 32, numExperts: 8,
            compileSeparatedDecode: compileSeparatedDecode)
        try glu.update(
            modules: ModuleChildren.unflattened([
                ("gate_proj", gate), ("up_proj", up), ("down_proj", down),
            ]),
            verify: [])
        return glu
    }

    @Test(
        "compileSeparatedDecode output matches the generic path on a tiny bf16 affine switch"
    )
    func separatedDecodeParity() throws {
        try MLXMetalTestLock.withLock {
            // 4-bit affine group-32 projections sourced from f16 weights with
            // bf16 metadata — the exact contract the compiled region guards
            // on. Both instances share the same projection modules.
            func projection(_ outputDims: Int, inputDims: Int, seed: UInt64)
                -> QuantizedSwitchLinear
            {
                let source = MLXRandom.uniform(
                    low: -0.5, high: 0.5, [8, outputDims, inputDims],
                    key: MLXRandom.key(seed)
                ).asType(.float16)
                let (weight, scales, biases) = MLX.quantized(
                    source, groupSize: 32, bits: 4, mode: .affine)
                return QuantizedSwitchLinear(
                    inputDims: inputDims,
                    outputDims: outputDims,
                    numExperts: 8,
                    weight: weight,
                    scales: scales.asType(.bfloat16),
                    biases: biases?.asType(.bfloat16),
                    groupSize: 32,
                    bits: 4,
                    mode: .affine)
            }

            let gate = projection(32, inputDims: 64, seed: 1)
            let up = projection(32, inputDims: 64, seed: 2)
            let down = projection(64, inputDims: 32, seed: 3)

            let compiled = try tinySwitchGLU(
                compileSeparatedDecode: true, gate: gate, up: up, down: down)
            let generic = try tinySwitchGLU(
                compileSeparatedDecode: false, gate: gate, up: up, down: down)

            let x = MLXRandom.uniform(
                low: -1.0, high: 1.0, [1, 1, 64],
                key: MLXRandom.key(7)
            ).asType(.bfloat16)
            let indices = MLXArray(
                [UInt32(0), 2, 4, 6, 1, 3, 5, 7], [1, 1, 8])

            let compiledOut = compiled(x, indices)
            let genericOut = generic(x, indices)

            #expect(compiledOut.shape == genericOut.shape)
            #expect(compiledOut.shape == [1, 1, 8, 64])
            #expect(compiledOut.dtype == .bfloat16)
            #expect(MLX.allClose(compiledOut, genericOut).item(Bool.self))
        }
    }
}
