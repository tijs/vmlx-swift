// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

@Suite("Qwen3.5 Hadamard routing", .serialized)
struct Qwen35HadamardRoutingTests {
    private static let config = """
        {
          "model_type": "qwen3_5", "text_config": {
            "model_type": "qwen3_5_text", "hidden_size": 512, "num_hidden_layers": 1,
            "intermediate_size": 512, "num_attention_heads": 8, "num_key_value_heads": 2,
            "linear_num_value_heads": 4, "linear_num_key_heads": 2,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
            "head_dim": 64, "vocab_size": 128, "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false
          }
        }
        """

    @Test("wrapped text and VLM routes opt in, tied and unported routes refuse")
    func routeSupport() throws {
        try MLXMetalTestLock.withLock {
            let data = Data(Self.config.utf8)
            let vlmConfig = try JSONDecoder().decode(MLXVLM.Qwen35Configuration.self, from: data)
            let llmConfig = try JSONDecoder().decode(MLXLLM.Qwen35Configuration.self, from: data)
            let vlm = MLXVLM.Qwen35(vlmConfig)
            let llm = MLXLLM.Qwen35Model(llmConfig)
            let contract = JangHadamardRuntimeContract(
                blockSize: 512,
                forward: ["language_model.lm_head"],
                inverse: ["language_model.model.embed_tokens"])
            try contract.validateRoute(model: vlm)
            try contract.validateRoute(model: llm)
            #expect(throws: JangLoaderError.self) {
                try contract.validateRoute(model: MLXLLM.Qwen35TextModel(llmConfig.textConfig))
            }

            let tied = Data(
                Self.config.replacingOccurrences(
                    of: "\"tie_word_embeddings\": false", with: "\"tie_word_embeddings\": true"
                ).utf8)
            let tiedVLM = MLXVLM.Qwen35(
                try JSONDecoder().decode(MLXVLM.Qwen35Configuration.self, from: tied))
            let tiedLLM = MLXLLM.Qwen35Model(
                try JSONDecoder().decode(MLXLLM.Qwen35Configuration.self, from: tied))
            #expect(throws: JangLoaderError.self) { try contract.validateRoute(model: tiedVLM) }
            #expect(throws: JangLoaderError.self) { try contract.validateRoute(model: tiedLLM) }
        }
    }

    @Test("Qwen sanitize keeps the zero-centered +1 convention and stored F32/F16 dtypes")
    func normSanitizeUnchanged() throws {
        try MLXMetalTestLock.withLock {
            let data = Data(Self.config.utf8)
            let models: [any LanguageModel] = [
                MLXVLM.Qwen35(
                    try JSONDecoder().decode(MLXVLM.Qwen35Configuration.self, from: data)),
                MLXLLM.Qwen35Model(
                    try JSONDecoder().decode(MLXLLM.Qwen35Configuration.self, from: data)),
            ]
            let normPaths = [
                "language_model.model.layers.0.input_layernorm.weight",
                "language_model.model.layers.0.post_attention_layernorm.weight",
                "language_model.model.norm.weight",
                "language_model.model.layers.1.self_attn.q_norm.weight",
                "language_model.model.layers.1.self_attn.k_norm.weight",
            ]
            let raw = MLXArray((0 ..< 512).map { Float($0 % 7 - 3) * 0.0137 })
            let gamma = raw + MLXArray(1, dtype: .float32)
            let linearNorm = "language_model.model.layers.0.linear_attn.norm.weight"
            let linearA = "language_model.model.layers.0.linear_attn.in_proj_a.weight"
            let scalesKey = "language_model.model.layers.0.linear_attn.in_proj_qkv.scales"
            let signsKey = "language_model.model.layers.0.linear_attn.in_proj_qkv.signs"
            let passthrough: [String: MLXArray] = [
                linearNorm: raw,
                linearA: raw.reshaped(1, 512),
                scalesKey: MLXArray([Float(0.0137), 0.0219]).asType(.float16).reshaped(1, 2),
                signsKey: MLXArray(JangHadamardFixture.signs),
            ]
            // Bonsai stores gamma-1. Already-shifted Qwen bundles must continue
            // to take the other existing path instead of receiving +1 twice.
            for storedNorm in [raw, gamma] {
                var weights = passthrough
                for path in normPaths { weights[path] = storedNorm }
                for model in models {
                    let sanitized = model.sanitize(weights: weights, metadata: ["format": "mlx"])
                    #expect(sanitized.count == weights.count)
                    for path in normPaths {
                        let actual = try #require(sanitized[path])
                        #expect(actual.dtype == .float32)
                        #expect(MLX.all(actual .== gamma).item(Bool.self))
                    }
                    for (path, source) in passthrough {
                        let actual = try #require(sanitized[path])
                        #expect(actual.dtype == source.dtype)
                        #expect(MLX.all(actual .== source).item(Bool.self))
                    }
                }
            }
        }
    }

    @Test("VLM GDN raw-array input and output fusions cannot bypass rotation")
    func vlmFusionPreservesTransforms() throws {
        try MLXMetalTestLock.withLock {
            let args = try JSONDecoder().decode(
                MLXVLM.Qwen35Configuration.self, from: Data(Self.config.utf8)
            )
            .textConfiguration
            let baseline = Qwen35Language.GatedDeltaNet(args, fuseDecodeInputProjections: false)
            let candidate = Qwen35Language.GatedDeltaNet(args, fuseDecodeInputProjections: true)
            let contract = JangHadamardRuntimeContract(
                blockSize: 512, forward: ["in_proj_qkv", "in_proj_z", "out_proj"], inverse: [])
            for layer in [baseline, candidate] {
                let bf16 = layer.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }
                try layer.update(parameters: ModuleParameters.unflattened(bf16), verify: [])
                quantize(
                    model: layer,
                    filter: { _, module in
                        module is Linear ? (groupSize: 128, bits: 2, mode: .affine) : nil
                    })
                try contract.install(model: layer)
                for (_, module) in layer.namedModules() {
                    if let rotated = module as? HadamardQuantizedLinear {
                        rotated.update(
                            parameters: ModuleParameters.unflattened([
                                ("signs", MLXArray(JangHadamardFixture.signs))
                            ]))
                    }
                }
            }
            // Identical loaded arrays, not merely an equal random seed.
            try candidate.update(parameters: baseline.parameters(), verify: [])
            let baselineCache = MambaCache()
            let candidateCache = MambaCache()
            for length in [3, 1, 1] {
                let input = MLXArray(
                    (0 ..< length * 512).map { Float($0 % 23 - 11) / 32 }, [1, length, 512]
                )
                .asType(.bfloat16)
                let expected = baseline(input, cache: baselineCache)
                let actual = candidate(input, cache: candidateCache)
                MLX.eval(expected, actual)
                #expect(MLX.all(actual .== expected).item(Bool.self))
                #expect(candidateCache.offset == baselineCache.offset)
                #expect(candidateCache.state.count == baselineCache.state.count)
                for (lhs, rhs) in zip(candidateCache.state, baselineCache.state) {
                    #expect(MLX.all(lhs .== rhs).item(Bool.self))
                }
            }
        }
    }

    @Test("text GDN grouped fusion cannot bypass rotated inputs")
    func textFusionPreservesTransforms() throws {
        try MLXMetalTestLock.withLock {
            let args = try JSONDecoder().decode(
                MLXLLM.Qwen35Configuration.self, from: Data(Self.config.utf8)
            ).textConfig
            let baseline = Qwen35GatedDeltaNet(args)
            let candidate = Qwen35GatedDeltaNet(args)
            let contract = JangHadamardRuntimeContract(
                blockSize: 512, forward: ["in_proj_qkv", "in_proj_z", "out_proj"], inverse: [])
            for layer in [baseline, candidate] {
                quantize(
                    model: layer,
                    filter: { _, module in
                        module is Linear ? (groupSize: 128, bits: 2, mode: .affine) : nil
                    })
                try contract.install(model: layer)
                for (_, module) in layer.namedModules() {
                    if let rotated = module as? HadamardQuantizedLinear {
                        rotated.update(
                            parameters: ModuleParameters.unflattened([
                                ("signs", MLXArray(JangHadamardFixture.signs))
                            ]))
                    }
                }
            }
            try candidate.update(parameters: baseline.parameters(), verify: [])
            let previous = ProcessInfo.processInfo.environment["VMLX_GDN_FUSE_DECODE_INPUTS"]
            defer {
                if let previous {
                    setenv("VMLX_GDN_FUSE_DECODE_INPUTS", previous, 1)
                } else {
                    unsetenv("VMLX_GDN_FUSE_DECODE_INPUTS")
                }
            }
            let input = MLXArray((0 ..< 512).map { Float($0 % 23 - 11) / 32 }, [1, 1, 512])
            setenv("VMLX_GDN_FUSE_DECODE_INPUTS", "0", 1)
            let expected = baseline(input, cache: MambaCache())
            MLX.eval(expected)
            setenv("VMLX_GDN_FUSE_DECODE_INPUTS", "1", 1)
            let actual = candidate(input, cache: MambaCache())
            MLX.eval(actual)
            #expect(MLX.all(actual .== expected).item(Bool.self))
        }
    }
}
