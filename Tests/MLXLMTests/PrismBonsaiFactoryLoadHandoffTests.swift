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
/// pinned-shape guard → nested text_config decoder extraction → plan build →
/// explicit `qwen3_5_text` decoder construction → `loadWeights`
/// `bonsaiTransform:` install → final `update(verify: [.noUnusedKeys])`.
///
/// The fixtures mirror the PINNED pack layout (`prism-ml/
/// Ternary-Bonsai-2-27B-mlx-2bit` at
/// `3f926b415992eaa2ae9dd7b573706494d6bbf787`): root `model_type` is
/// `prism_hadamard_qwen35` with the manifest fields at the root, and the
/// `qwen3_5_text` decoder architecture (incl. `hidden_size`/
/// `num_hidden_layers`/`vocab_size`) lives under `text_config` — the root
/// carries NO decoder dimensions.
///
/// These are REAL target tests: a genuine `Qwen35TextModel` (small dims,
/// all-attention so the plan's `model.layers.0.self_attn.q_proj` leaf
/// exists) is constructed from the nested text_config and the internal
/// `installBonsaiPrismHadamard` seam swaps its leaves in-memory with the
/// pack's own key naming (`language_model.*` checkpoint namespace). No
/// weights, no network, no Metal compute beyond the quantize helper the
/// existing install tests already use.
@Suite("Prism Bonsai factory/load handoff")
struct PrismBonsaiFactoryLoadHandoffTests {

    // MARK: Fixtures (mirror the portability-gate fixtures, plus the nested
    // qwen3_5_text decoder dimensions in text_config, exactly like the pinned
    // pack)

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
            "tie_word_embeddings": false,
            // Mirror the pinned pack: the decoder architecture (sizes, heads,
            // rope, ...) lives under text_config; the ROOT carries no
            // hidden_size/num_hidden_layers/vocab_size.
            "text_config": [
                "model_type": textDecoder,
                "hidden_size": 1024,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 256,
                "full_attention_interval": 1,
                "tie_word_embeddings": false,
                "vocab_size": 16,
            ],
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

    /// Build the real (small) qwen3_5_text decoder exactly like `_load` does:
    /// decode `Qwen35TextConfiguration` from the NESTED `text_config` object
    /// (the factory never decodes from the root, which carries no decoder
    /// dims in the pinned layout).
    private func makeRealDecoder() throws -> Qwen35TextModel {
        guard let textDecoderData =
            PrismBonsaiPortability.textDecoderConfigurationData(
                configData: bonsaiConfig())
        else {
            Issue.record("expected nested text_config decoder data")
            throw PrismBonsaiInstall.Error("handoff precondition failed")
        }
        let configuration = try JSONDecoder.json5().decode(
            Qwen35TextConfiguration.self, from: textDecoderData)
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

    @Test("bare nested decoder (no dimensions) fails the factory extraction")
    func bareNestedDecoderFailsClosed() {
        // decide() still approves: the manifest itself is valid ...
        let bare = bonsaiConfig([
            "text_config": ["model_type": "qwen3_5_text"],
        ])
        let decision = PrismBonsaiPortability.decide(
            configData: bare, hadamardData: hadamardData(), gateEnabled: true)
        #expect(decision == .gateOnManifestValid)
        // ... but the factory's default-parameter trap refuses to decode a
        // Qwen35TextConfiguration without explicit NESTED size parameters.
        #expect(
            PrismBonsaiPortability.textDecoderConfigurationData(
                configData: bare) == nil)
        // Missing text_config entirely also fails closed.
        #expect(
            PrismBonsaiPortability.textDecoderConfigurationData(
                configData: bonsaiConfig(["text_config": NSNull()])) == nil)
        // A root-flattened decoder (dims at the root, no text_config) fails
        // closed too — the pinned layout is nested, never root-flattened.
        #expect(
            PrismBonsaiPortability.textDecoderConfigurationData(
                configData: bonsaiConfig([
                    "text_config": NSNull(),
                    "hidden_size": 1024,
                    "num_hidden_layers": 1,
                    "vocab_size": 16,
                ])) == nil)
        // The full nested decoder extracts.
        #expect(
            PrismBonsaiPortability.textDecoderConfigurationData(
                configData: bonsaiConfig()) != nil)
        // Non-positive nested sizes fail closed.
        let nonPositive: [String: Any] = [
            "model_type": "qwen3_5_text",
            "hidden_size": 1024,
            "num_hidden_layers": 0,
            "vocab_size": 16,
        ]
        #expect(
            PrismBonsaiPortability.textDecoderConfigurationData(
                configData: bonsaiConfig(["text_config": nonPositive])) == nil)
    }

    @Test("nested text_config builds the explicit qwen3_5_text decoder, never the default 4096/32/151936")
    func nestedTextConfigBuildsExplicitDecoder() throws {
        // Pre-fix behavior (decoding from the ROOT configData) silently yields
        // the default-parameter decoder: the root carries no decoder dims.
        let rootDecoded = try JSONDecoder.json5().decode(
            Qwen35TextConfiguration.self, from: bonsaiConfig())
        #expect(rootDecoded.hiddenSize == 4096)
        #expect(rootDecoded.hiddenLayers == 32)
        #expect(rootDecoded.vocabularySize == 151_936)

        // The factory now decodes the actual nested text_config object: the
        // explicit small decoder, not the default one.
        guard let textDecoderData =
            PrismBonsaiPortability.textDecoderConfigurationData(
                configData: bonsaiConfig())
        else {
            Issue.record("expected nested text_config decoder data")
            throw PrismBonsaiInstall.Error("handoff precondition failed")
        }
        let nestedDecoded = try JSONDecoder.json5().decode(
            Qwen35TextConfiguration.self, from: textDecoderData)
        #expect(nestedDecoded.modelType == "qwen3_5_text")
        #expect(nestedDecoded.hiddenSize == 1024)
        #expect(nestedDecoded.hiddenLayers == 1)
        #expect(nestedDecoded.vocabularySize == 16)

        // The model the factory constructs (via makeRealDecoder) therefore
        // carries the explicit nested dimensions.
        let model = try makeRealDecoder()
        #expect(model.configuration.hiddenSize == 1024)
        #expect(model.configuration.hiddenLayers == 1)
        #expect(model.configuration.vocabularySize == 16)
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

    // MARK: 3. Text-only decoder key normalization (VLM wrapper + vision sidecar)

    /// Distinct shape per tensor, so a mis-bind is visible as the wrong
    /// extent without forcing an evaluation (same tagging convention as
    /// `WeightsTests`).
    private static func tagged(_ tag: Int) -> MLXArray {
        MLXArray.zeros([tag])
    }

    @Test("text-only Bonsai normalization flattens language_model body/head, drops vision sidecar, preserves flat keys")
    func normalizesPrefixedBodyHeadAndDropsVisionSidecar() {
        // The pinned pack's post-install remainder: ordinary decoder keys
        // under the VLM `language_model.` wrapper (`model.*` body,
        // `lm_head.*` head) plus `vision_tower.*` sidecar tensors, with a
        // few already-flat keys mixed in.
        let weights: [String: MLXArray] = [
            "language_model.model.norm.weight": Self.tagged(1),
            "language_model.model.layers.0.input_layernorm.weight": Self.tagged(2),
            "language_model.lm_head.weight": Self.tagged(3),
            // Already-flat keys: pass through untouched, same key, same tensor.
            "model.embed_tokens.weight": Self.tagged(4),
            "model.layers.0.post_attention_layernorm.weight": Self.tagged(5),
            // vision_tower sidecar: dropped entirely.
            "vision_tower.visual.blocks.0.attn.qkv.weight": Self.tagged(6),
            "vision_tower.visual.positional_embedding": Self.tagged(7),
        ]

        let normalized = normalizeBonsaiTextDecoderWeights(weights)

        // Body/head wrapper flattened onto the decoder's module paths.
        #expect(normalized["model.norm.weight"]?.shape == [1])
        #expect(normalized["model.layers.0.input_layernorm.weight"]?.shape == [2])
        #expect(normalized["lm_head.weight"]?.shape == [3])
        // No spelling of the wrapper survives.
        #expect(!normalized.keys.contains { $0.hasPrefix("language_model") })
        // Flat keys preserved under their own key names.
        #expect(normalized["model.embed_tokens.weight"]?.shape == [4])
        #expect(normalized["model.layers.0.post_attention_layernorm.weight"]?.shape == [5])
        // vision sidecar gone.
        #expect(!normalized.keys.contains { $0.hasPrefix("vision_tower") })
        #expect(normalized.count == 5)
    }

    @Test("text-only Bonsai normalization resolves duplicate spellings deterministically")
    func normalizesDuplicateSpellingsDeterministically() {
        // Mixed-provenance re-bake: the checkpoint carries BOTH spellings of
        // the same destination. The documented rule in
        // `Weights.stripLanguageModelPrefix` is that an unprefixed key
        // already at the destination always wins — never dictionary
        // iteration order.
        let weights: [String: MLXArray] = [
            "language_model.model.norm.weight": Self.tagged(11),
            "model.norm.weight": Self.tagged(22),  // the one that must win
        ]

        let normalized = normalizeBonsaiTextDecoderWeights(weights)

        #expect(normalized.count == 1)
        #expect(normalized["model.norm.weight"]?.shape == [22])
        #expect(normalized["language_model.model.norm.weight"] == nil)
    }

    @Test("install → normalization → final noUnusedKeys: VLM-wrapped ordinary keys + vision sidecar bind cleanly")
    func installThenNormalizeThenNoUnusedKeysPasses() throws {
        // Admission replay: the real pack resolves/consumes every packed
        // tensor under its own `language_model.*` key names, then leaves the
        // ORDINARY decoder keys still VLM-wrapped plus a `vision_tower.*`
        // sidecar. Before this fix the final update on the bare
        // Qwen35TextModel failed with `Unhandled keys ["language_model",
        // "vision_tower"]`.
        let plan = try factoryPlan()
        let model = try makeRealDecoder()

        let lmHead = packedTensors(out: 16, inputWidth: 1024, seed: 61)
        let qProj = packedTensors(out: 2048, inputWidth: 1024, seed: 62)
        let embed = packedTensors(out: 16, inputWidth: 1024, seed: 63)
        var weights: [String: MLXArray] = [
            // Packed tensors, consumed by the install seam.
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
            // Ordinary decoder keys the plan does NOT cover — VLM-wrapped.
            "language_model.model.norm.weight": MLXRandom.normal([1024]),
            "language_model.model.layers.0.input_layernorm.weight": MLXRandom.normal([1024]),
            // Already-flat ordinary key: survives untouched.
            "model.layers.0.post_attention_layernorm.weight": MLXRandom.normal([1024]),
            // vision_tower sidecar (333 tensors in the real pack).
            "vision_tower.visual.blocks.0.attn.qkv.weight": MLXRandom.normal([1]),
            "vision_tower.visual.positional_embedding": MLXRandom.normal([2]),
        ]

        try installBonsaiPrismHadamard(plan, model: model, weights: &weights)
        // Packed tensors are gone; the ordinary VLM-wrapped keys remain.
        #expect(weights["language_model.model.norm.weight"] != nil)
        #expect(weights["vision_tower.visual.positional_embedding"] != nil)
        #expect(!weights.keys.contains { $0.contains("lm_head.weight") })

        // Pre-normalization replay of the observed blocker: the final update
        // on the bare text decoder rejects the VLM wrapper + vision sidecar
        // containers.
        do {
            try model.update(
                parameters: ModuleParameters.unflattened(weights),
                verify: [.noUnusedKeys])
            Issue.record(
                "expected the observed unhandled-keys rejection before normalization")
        } catch let error as UpdateError {
            guard case .unhandledKeys(let path, let modules, let keys) = error else {
                Issue.record("unexpected update error \(error)")
                return
            }
            #expect(keys == ["language_model", "vision_tower"])
            #expect(path.isEmpty)
            #expect(modules.contains("Qwen35TextModel"))
        } catch {
            Issue.record("unexpected error \(error)")
        }

        // Post-normalization: wrapper flattened onto the decoder module
        // paths, vision sidecar gone, every remaining key binds.
        let normalized = normalizeBonsaiTextDecoderWeights(weights)
        #expect(!normalized.keys.contains { $0.hasPrefix("language_model") })
        #expect(!normalized.keys.contains { $0.hasPrefix("vision_tower") })
        #expect(normalized["model.norm.weight"] != nil)
        #expect(normalized["model.layers.0.input_layernorm.weight"] != nil)
        #expect(normalized["model.layers.0.post_attention_layernorm.weight"] != nil)
        #expect(normalized.count == 3)

        // The exact tail of loadWeights: noUnusedKeys must pass with no
        // unhandled keys.
        let parameters = ModuleParameters.unflattened(normalized)
        try model.update(parameters: parameters, verify: [.noUnusedKeys])
    }
}