// Copyright © 2026

import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXLMCommon
@testable import MLXLLM

/// Focused tests for the Bonsai 2 Prism-Hadamard FACTORY/LOAD handoff:
/// the exact sequence `LLMModelFactory._load` now runs when
/// `VMLX_BONSAI_PRISM_HADAMARD=1` and the manifest validates — decide →
/// pinned-shape guard → text-decoder size probe → plan build →
/// explicit `qwen3_5_text` decoder construction → `loadWeights`
/// `bonsaiTransform:` install → final `update(verify: [.noUnusedKeys])`.
///
/// These are REAL target tests: a genuine `Qwen35TextModel` (small dims,
/// all-attention so the plan's `model.layers.0.self_attn.q_proj` leaf
/// exists) is constructed and the internal `installBonsaiPrismHadamard`
/// seam swaps its leaves in-memory with the pack's own key naming
/// (`language_model.*` checkpoint namespace). No weights, no network, no
/// Metal compute beyond the quantize helper the existing install tests
/// already use.
@Suite("Prism Bonsai factory/load handoff")
struct PrismBonsaiFactoryLoadHandoffTests {

    // MARK: Fixtures (mirror the portability-gate fixtures, plus the root
    // qwen3_5_text decoder dimensions the factory requires)

    private let modules: [[String: Any]] = [
        ["path": "lm_head", "block": 1024, "embedding": false, "dtype": "float16"],
        [
            "path": "model.layers.0.self_attn.q_proj", "block": 1024,
            "embedding": false, "dtype": "float16",
        ],
        [
            "path": "model.embed_tokens", "block": 1024, "embedding": true,
            "dtype": "float16",
        ],
    ]

    private func bonsaiConfig(
        _ overrides: [String: Any] = [:],
        textDecoder: String = "qwen3_5_text"
    ) -> Data {
        var dict: [String: Any] = [
            "schema_version": 2,
            "model_type": "prism_hadamard_qwen35",
            "requires_runtime": "runtime/artifact.py",
            "hadamard_config": "hadamard.json",
            "tensor_namespace": "mlx-vlm-qwen3_5",
            "gdn_activation_layout": "grouped",
            "base_model_type": "qwen3_5",
            "quantization": ["bits": 2, "group_size": 128, "mode": "affine"],
            "modules": modules,
            "text_config": ["model_type": textDecoder],
            // qwen3_5_text decoder dimensions: the factory refuses to
            // construct the decoder when these are absent at the root.
            "hidden_size": 1024,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 256,
            "full_attention_interval": 1,
            "tie_word_embeddings": false,
            "vocab_size": 16,
        ]
        for (key, value) in overrides { dict[key] = value }
        return try! JSONSerialization.data(withJSONObject: dict)
    }

    private func hadamardData(_ overrides: [String: Any] = [:]) -> Data {
        var dict: [String: Any] = [
            "prism.hadamard.version": 1,
            "prism.hadamard.block_size": 1024,
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension",
            "prism.hadamard.sign_mode": "explicit",
            "prism.hadamard.weight_names": [
                "language_model.lm_head.weight",
                "language_model.model.layers.0.self_attn.q_proj.weight",
            ],
            "prism.hadamard.inverse_weight_names": [
                "language_model.model.embed_tokens.weight",
            ],
            "prism.hadamard.sign_widths": [4],
            "prism.hadamard.sign_values": [1.0, -1.0, 1.0, -1.0],
            "prism.hadamard.gdn_v_grouped": true,
        ]
        for (key, value) in overrides { dict[key] = value }
        return try! JSONSerialization.data(withJSONObject: dict)
    }

    /// The factory's gate-on-valid resolution: decide() must approve, and the
    /// plan builder must re-validate — mirroring `_load`'s `.gateOnManifestValid`
    /// branch.
    private func factoryPlan() throws -> PrismBonsaiHadamardPlan {
        let decision = PrismBonsaiPortability.decide(
            configData: bonsaiConfig(), hadamardData: hadamardData(), gateEnabled: true)
        guard case .gateOnManifestValid = decision else {
            Issue.record("expected .gateOnManifestValid, got \(decision)")
            throw PrismBonsaiInstall.Error("handoff precondition failed")
        }
        return try PrismBonsaiHadamardPlan(
            configData: bonsaiConfig(), hadamardData: hadamardData())
    }

    /// Build the real (small) qwen3_5_text decoder exactly like `_load` does.
    private func makeRealDecoder() throws -> Qwen35TextModel {
        let configuration = try JSONDecoder.json5().decode(
            Qwen35TextConfiguration.self, from: bonsaiConfig())
        return Qwen35TextModel(configuration)
    }

    private func packedTensors(out: Int, inputWidth: Int, seed: UInt64)
        -> (weight: MLXArray, scales: MLXArray, biases: MLXArray, signs: MLXArray)
    {
        MLXRandom.seed(seed)
        let weight = MLXRandom.normal([out, inputWidth])
        let quantized = MLX.quantized(
            weight, groupSize: 128, bits: 2, mode: .affine)
        let signsValues: [Float] = (0..<inputWidth).map {
            $0.isMultiple(of: 2) ? Float(1) : Float(-1)
        }
        return (quantized.0, quantized.1, quantized.2!, MLXArray(signsValues))
    }

    private func leafTypes(_ model: Qwen35TextModel) -> [String: String] {
        Dictionary(
            uniqueKeysWithValues: model.leafModules().flattened().map {
                ($0.0, String(describing: type(of: $0.1)))
            })
    }

    // MARK: 1. Handoff decision chain

    @Test("valid manifest: decide → plan → explicit qwen3_5_text decoder")
    func handoffChainBuildsPlanAndRealDecoder() throws {
        let plan = try factoryPlan()
        #expect(plan.entries.count == 3)
        #expect(plan.block == 1024)
        #expect(plan.bits == 2)
        #expect(plan.groupSize == 128)
        #expect(
            plan.entry(forCheckpointBase: "lm_head")?.role
                == .forwardPackedLinear)
        #expect(
            plan.entry(forCheckpointBase: "model.embed_tokens")?.role
                == .inversePackedEmbedding)

        // The plan must resolve against the REAL decoder tree and the pack's
        // language_model.* checkpoint key naming (the exact inputs the
        // install seam resolves).
        let model = try makeRealDecoder()
        let leaves = Set(model.leafModules().flattened().map(\.0))
        #expect(leaves.contains("lm_head"))
        #expect(leaves.contains("model.embed_tokens"))
        #expect(leaves.contains("model.layers.0.self_attn.q_proj"))
        let resolved = try PrismBonsaiInstall.resolve(
            plan: plan,
            leafModulePaths: leaves,
            checkpointKeys: Set([
                "language_model.lm_head.weight",
                "language_model.lm_head.scales",
                "language_model.lm_head.biases",
                "language_model.lm_head.signs",
                "language_model.model.layers.0.self_attn.q_proj.weight",
                "language_model.model.layers.0.self_attn.q_proj.scales",
                "language_model.model.layers.0.self_attn.q_proj.biases",
                "language_model.model.layers.0.self_attn.q_proj.signs",
                "language_model.model.embed_tokens.weight",
                "language_model.model.embed_tokens.scales",
                "language_model.model.embed_tokens.biases",
                "language_model.model.embed_tokens.signs",
            ]))
        #expect(resolved.updates.count == 3)
        #expect(resolved.updates[0].weightKey == "language_model.lm_head.weight")
        #expect(
            resolved.updates[1].weightKey
                == "language_model.model.layers.0.self_attn.q_proj.weight")
        #expect(
            resolved.updates[2].weightKey
                == "language_model.model.embed_tokens.weight")
    }

    @Test("bare manifest (no decoder dimensions) fails the factory size probe")
    func bareManifestFailsDecoderSizeProbe() {
        // The manifest itself is valid (decide approves) ...
        let decision = PrismBonsaiPortability.decide(
            configData: bonsaiConfig([
                "hidden_size": NSNull(), "num_hidden_layers": NSNull(),
                "vocab_size": NSNull(),
            ]),
            hadamardData: hadamardData(), gateEnabled: true)
        #expect(decision == .gateOnManifestValid)
        // ... but the factory's default-parameter trap refuses to construct
        // a Qwen35TextModel without explicit root size parameters.
        #expect(
            PrismBonsaiPortability.textDecoderRootSizeParameters(
                configData: bonsaiConfig([
                    "hidden_size": NSNull(), "num_hidden_layers": NSNull(),
                    "vocab_size": NSNull(),
                ])) == nil)
        #expect(
            PrismBonsaiPortability.textDecoderRootSizeParameters(
                configData: bonsaiConfig()) != nil)
        // Non-positive sizes also fail closed.
        #expect(
            PrismBonsaiPortability.textDecoderRootSizeParameters(
                configData: bonsaiConfig(["num_hidden_layers": 0])) == nil)
    }

    @Test("root prism with a wrong nested decoder fails closed (gate + plan)")
    func wrongNestedDecoderFailsClosed() {
        let config = bonsaiConfig(textDecoder: "qwen3_5_moe")
        guard case .gateOnManifestInvalid(let error) =
            PrismBonsaiPortability.decide(
                configData: config, hadamardData: hadamardData(), gateEnabled: true)
        else {
            Issue.record("expected gateOnManifestInvalid for qwen3_5_moe decoder")
            return
        }
        #expect(error.reason.contains("text_config.model_type must be"))
        do {
            _ = try PrismBonsaiHadamardPlan(
                configData: config, hadamardData: hadamardData())
            Issue.record("plan build must reject a wrong nested decoder")
        } catch let error as PrismBonsaiHadamardPlan.BuildError {
            #expect(error.reason.contains("text_config.model_type must be"))
        } catch {
            Issue.record("unexpected error \(error)")
        }
    }

    @Test("nested-only prism identity stays decision-valid but is refused by the factory shape guard")
    func nestedOnlyPrismIsNotARootTransformLoad() {
        // decide() keeps the nested shape valid (a VLM factory concern) ...
        let config = bonsaiConfig(
            ["model_type": "some_unknown_vlm_type"],
            textDecoder: "prism_hadamard_qwen35")
        #expect(
            PrismBonsaiPortability.decide(
                configData: config, hadamardData: hadamardData(), gateEnabled: true)
                == .gateOnManifestValid)
        // ... and the LLM factory's pinned-shape guard refuses to absorb it:
        // only the ROOT model_type is an LLM-factory transform load.
        #expect(
            PrismBonsaiPortability.isRootPrismIdentity(
                rootModelType: "prism_hadamard_qwen35"))
        #expect(
            !PrismBonsaiPortability.isRootPrismIdentity(
                rootModelType: "some_unknown_vlm_type"))
    }

    // MARK: 2. Real install on the qwen3_5_text decoder + no-unused-key tail

    @Test("install swaps the real decoder's packed leaves, consumes keys, final noUnusedKeys passes")
    func installOnRealDecoderConsumesKeysAndPassesNoUnusedKeys() throws {
        let plan = try factoryPlan()
        let model = try makeRealDecoder()

        // Pack-published checkpoint naming: language_model.* namespace.
        // q_proj fuses query+gate -> out = heads * headDim * 2 = 4*256*2.
        let lmHead = packedTensors(out: 16, inputWidth: 1024, seed: 41)
        let qProj = packedTensors(out: 2048, inputWidth: 1024, seed: 42)
        let embed = packedTensors(out: 16, inputWidth: 1024, seed: 43)
        var weights: [String: MLXArray] = [
            "language_model.lm_head.weight": lmHead.weight,
            "language_model.lm_head.scales": lmHead.scales,
            "language_model.lm_head.biases": lmHead.biases,
            "language_model.lm_head.signs": lmHead.signs,
            "language_model.model.layers.0.self_attn.q_proj.weight": qProj.weight,
            "language_model.model.layers.0.self_attn.q_proj.scales": qProj.scales,
            "language_model.model.layers.0.self_attn.q_proj.biases": qProj.biases,
            "language_model.model.layers.0.self_attn.q_proj.signs": qProj.signs,
            "language_model.model.embed_tokens.weight": embed.weight,
            "language_model.model.embed_tokens.scales": embed.scales,
            "language_model.model.embed_tokens.biases": embed.biases,
            "language_model.model.embed_tokens.signs": embed.signs,
            // Ordinary keys the plan does not cover: the final update still
            // applies them (and .noUnusedKeys stays meaningful).
            "model.norm.weight": MLXRandom.normal([1024]),
        ]

        try installBonsaiPrismHadamard(plan, model: model, weights: &weights)

        let leaves = leafTypes(model)
        #expect(leaves["lm_head"] == "HadamardPackedLinear")
        #expect(
            leaves["model.layers.0.self_attn.q_proj"] == "HadamardPackedLinear")
        #expect(leaves["model.embed_tokens"] == "HadamardPackedEmbedding")
        // The ordinary attention projections stay untouched (k/v/o are not in
        // the plan and carry no packed companions).
        #expect(leaves["model.layers.0.self_attn.k_proj"] == "Linear")
        #expect(leaves["model.layers.0.self_attn.v_proj"] == "Linear")
        #expect(leaves["model.layers.0.self_attn.o_proj"] == "Linear")

        for key in [
            "language_model.lm_head.weight",
            "language_model.lm_head.scales",
            "language_model.lm_head.biases",
            "language_model.lm_head.signs",
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "language_model.model.layers.0.self_attn.q_proj.scales",
            "language_model.model.layers.0.self_attn.q_proj.biases",
            "language_model.model.layers.0.self_attn.q_proj.signs",
            "language_model.model.embed_tokens.weight",
            "language_model.model.embed_tokens.scales",
            "language_model.model.embed_tokens.biases",
            "language_model.model.embed_tokens.signs",
        ] {
            #expect(weights[key] == nil, "key \(key) must be consumed")
        }
        #expect(weights["model.norm.weight"] != nil)

        // Tail of loadWeights: the final noUnusedKeys update over the
        // remaining (ordinary) keys passes only because the packed/signs keys
        // were consumed transactionally by the install seam.
        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: [.noUnusedKeys])
    }

    @Test("install on the real decoder fails closed before any swap when a sign vector is missing")
    func installOnRealDecoderFailsBeforeSwapOnMissingSigns() throws {
        let plan = try factoryPlan()
        let model = try makeRealDecoder()

        let lmHead = packedTensors(out: 16, inputWidth: 1024, seed: 51)
        let qProj = packedTensors(out: 2048, inputWidth: 1024, seed: 52)
        let embed = packedTensors(out: 16, inputWidth: 1024, seed: 53)
        var weights: [String: MLXArray] = [
            "language_model.lm_head.weight": lmHead.weight,
            "language_model.lm_head.scales": lmHead.scales,
            "language_model.lm_head.biases": lmHead.biases,
            "language_model.lm_head.signs": lmHead.signs,
            "language_model.model.layers.0.self_attn.q_proj.weight": qProj.weight,
            "language_model.model.layers.0.self_attn.q_proj.scales": qProj.scales,
            "language_model.model.layers.0.self_attn.q_proj.biases": qProj.biases,
            "language_model.model.layers.0.self_attn.q_proj.signs": qProj.signs,
            "language_model.model.embed_tokens.weight": embed.weight,
            "language_model.model.embed_tokens.scales": embed.scales,
            "language_model.model.embed_tokens.biases": embed.biases,
            // embed_tokens.signs deliberately missing.
        ]

        do {
            try installBonsaiPrismHadamard(plan, model: model, weights: &weights)
            Issue.record("expected install to fail on the missing sign vector")
        } catch let error as PrismBonsaiInstall.Error {
            #expect(error.reason.contains("missing sign vector"))
        }

        // No partial load: nothing was swapped, nothing was consumed.
        let leaves = leafTypes(model)
        #expect(leaves["lm_head"] == "Linear")
        #expect(leaves["model.embed_tokens"] == "Embedding")
        #expect(weights["language_model.lm_head.signs"] != nil)
    }
}