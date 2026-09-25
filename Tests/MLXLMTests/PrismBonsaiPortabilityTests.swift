// Copyright © 2026

import Foundation
import Testing
@testable import MLXLLM

/// Deterministic tests for the Bonsai 2 Prism-Hadamard portability gate
/// (`PrismBonsaiPortability`). No model weights, no network, no Metal, no
/// server: every test feeds in-memory JSON payloads shaped after the pinned
/// pack `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` @
/// `3f926b415992eaa2ae9dd7b573706494d6bbf787` (verified 2026-09-18 against
/// the raw config.json / hadamard.json at that revision).
@Suite("Prism Bonsai portability gate")
struct PrismBonsaiPortabilityTests {

    // MARK: Fixtures

    private let defaultModules: [Any] = [
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
        textConfigOverride: Any? = nil
    ) -> Data {
        var textConfig: [String: Any] = ["model_type": "qwen3_5_text"]
        if let textConfigOverride {
            textConfig = (textConfigOverride as? [String: Any])
                ?? ["model_type": textConfigOverride]
        }
        var dict: [String: Any] = [
            "schema_version": 2,
            "model_type": "prism_hadamard_qwen35",
            "requires_runtime": "runtime/artifact.py",
            "hadamard_config": "hadamard.json",
            "tensor_namespace": "mlx-vlm-qwen3_5",
            "gdn_activation_layout": "grouped",
            "base_model_type": "qwen3_5",
            "quantization": ["bits": 2, "group_size": 128, "mode": "affine"],
            "modules": defaultModules,
            "text_config": textConfig,
        ]
        for (key, value) in overrides {
            dict[key] = value
        }
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
                "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
            ],
            "prism.hadamard.inverse_weight_names": [
                "language_model.model.embed_tokens.weight",
            ],
            "prism.hadamard.sign_widths": [4],
            "prism.hadamard.sign_values": [1.0, -1.0, 1.0, -1.0],
            "prism.hadamard.gdn_v_grouped": true,
        ]
        for (key, value) in overrides {
            dict[key] = value
        }
        return try! JSONSerialization.data(withJSONObject: dict)
    }

    private func expectInvalid(
        _ decision: PrismBonsaiPortability.Decision,
        containing fragment: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        guard case .gateOnManifestInvalid(let error) = decision else {
            Issue.record(
                "expected .gateOnManifestInvalid, got \(decision)",
                sourceLocation: sourceLocation)
            return
        }
        #expect(error.reason.contains(fragment))
    }

    // MARK: 1. Gate OFF preserves the ordinary path (incl. text_config fallback)

    @Test("gate OFF: unknown root + registered text_config stays on the ordinary path")
    func ordinaryTextConfigFallbackPreservedWhenOff() {
        // Root type unknown to the LLM registry; text_config.model_type is a
        // registered type. With the gate OFF, the ordinary (fallback-eligible)
        // path — NOT the Bonsai gate — must decide.
        let config = bonsaiConfig(
            ["model_type": "some_unknown_vlm_type"],
            textConfigOverride: "qwen3_5_text")
        let decision = PrismBonsaiPortability.decide(
            configData: config, hadamardData: nil, gateEnabled: false)
        #expect(decision == .notBonsai)
    }

    @Test("gate ON: ordinary configs are still notBonsai")
    func ordinaryConfigsNotBonsaiEvenWhenGateEnabled() {
        let config = bonsaiConfig(
            ["model_type": "qwen3_5_text"],
            textConfigOverride: "qwen3_5_text")
        let decision = PrismBonsaiPortability.decide(
            configData: config, hadamardData: nil, gateEnabled: true)
        #expect(decision == .notBonsai)
    }

    @Test("nil or unreadable config data is notBonsai")
    func nilConfigIsNotBonsai() {
        #expect(
            PrismBonsaiPortability.decide(
                configData: nil, hadamardData: nil, gateEnabled: true) == .notBonsai)
        #expect(
            PrismBonsaiPortability.decide(
                configData: Data("not json".utf8), hadamardData: nil,
                gateEnabled: true) == .notBonsai)
    }

    // MARK: 2. prism_hadamard_qwen35 never falls through to text_config

    @Test("gate OFF: prism root type rejects — the text_config trap is closed")
    func prismRootRejectsWhenOff() {
        // The exact trap shape: root is prism_hadamard_qwen35 and
        // text_config.model_type is the registered qwen3_5_text. The decision
        // must be gateOffReject (clean unsupported reject), never notBonsai —
        // i.e. the text_config fallback must NOT be reachable.
        let config = bonsaiConfig()
        let decision = PrismBonsaiPortability.decide(
            configData: config, hadamardData: nil, gateEnabled: false)
        #expect(decision == .gateOffReject)
    }

    @Test("gate OFF: prism as text_config.model_type also rejects")
    func prismTextConfigRejectsWhenOff() {
        // Reverse direction: root is unregistered, nested text_config is
        // prism. The fallback target must be protected too.
        let config = bonsaiConfig(
            ["model_type": "some_unknown_vlm_type"],
            textConfigOverride: "prism_hadamard_qwen35")
        let decision = PrismBonsaiPortability.decide(
            configData: config, hadamardData: nil, gateEnabled: false)
        #expect(decision == .gateOffReject)
    }

    @Test("gate ON + valid manifest: gateOnManifestValid, never a fake model")
    func validManifestIsExplicitRefusal() {
        let decision = PrismBonsaiPortability.decide(
            configData: bonsaiConfig(), hadamardData: hadamardData(),
            gateEnabled: true)
        #expect(decision == .gateOnManifestValid)
    }

    @Test("gate ON + valid manifest with prism nested in text_config")
    func validManifestNestedTextConfig() {
        let config = bonsaiConfig(
            ["model_type": "some_unknown_vlm_type"],
            textConfigOverride: "prism_hadamard_qwen35")
        let decision = PrismBonsaiPortability.decide(
            configData: config, hadamardData: hadamardData(), gateEnabled: true)
        #expect(decision == .gateOnManifestValid)
    }

    // MARK: 3. Gate ON manifest/sign validation matrix

    @Test("gate ON: missing requires_runtime is invalid")
    func missingRequiresRuntimeInvalid() {
        let config = bonsaiConfig(["requires_runtime": NSNull()])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: config, hadamardData: hadamardData(), gateEnabled: true),
            containing: "requires_runtime")
    }

    @Test("gate ON: non-artifact requires_runtime is invalid")
    func wrongRequiresRuntimeInvalid() {
        let config = bonsaiConfig(["requires_runtime": "runtime/other.py"])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: config, hadamardData: hadamardData(), gateEnabled: true),
            containing: "requires_runtime must be \"runtime/artifact.py\"")
    }

    @Test("gate ON: schema_version must be 2")
    func schemaVersionInvalid() {
        let config = bonsaiConfig(["schema_version": 1])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: config, hadamardData: hadamardData(), gateEnabled: true),
            containing: "schema_version")
    }

    @Test("gate ON: hadamard_config must be hadamard.json")
    func hadamardConfigInvalid() {
        let config = bonsaiConfig(["hadamard_config": "other.json"])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: config, hadamardData: hadamardData(), gateEnabled: true),
            containing: "hadamard_config")
        let missing = bonsaiConfig(["hadamard_config": NSNull()])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: missing, hadamardData: hadamardData(), gateEnabled: true),
            containing: "hadamard_config")
    }

    @Test("gate ON: base_model_type must be qwen3_5")
    func baseModelTypeInvalid() {
        let config = bonsaiConfig(["base_model_type": "qwen3_5_vl"])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: config, hadamardData: hadamardData(), gateEnabled: true),
            containing: "base_model_type")
    }

    @Test("gate ON: tensor_namespace must be present")
    func tensorNamespaceInvalid() {
        let config = bonsaiConfig(["tensor_namespace": NSNull()])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: config, hadamardData: hadamardData(), gateEnabled: true),
            containing: "tensor_namespace")
    }

    @Test("gate ON: gdn_activation_layout must be grouped")
    func gdnLayoutInvalid() {
        let config = bonsaiConfig(["gdn_activation_layout": "sequential"])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: config, hadamardData: hadamardData(), gateEnabled: true),
            containing: "gdn_activation_layout")
    }

    @Test("gate ON: quantization must be affine 2-bit / 128 group")
    func quantizationInvalid() {
        let missing = bonsaiConfig(["quantization": NSNull()])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: missing, hadamardData: hadamardData(), gateEnabled: true),
            containing: "quantization")

        let bits = bonsaiConfig(["quantization": ["bits": 4, "group_size": 128, "mode": "affine"]])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: bits, hadamardData: hadamardData(), gateEnabled: true),
            containing: "quantization.bits")

        let group = bonsaiConfig(["quantization": ["bits": 2, "group_size": 64, "mode": "affine"]])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: group, hadamardData: hadamardData(), gateEnabled: true),
            containing: "quantization.group_size")

        let mode = bonsaiConfig(["quantization": ["bits": 2, "group_size": 128, "mode": "symmetric"]])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: mode, hadamardData: hadamardData(), gateEnabled: true),
            containing: "quantization.mode")
    }

    @Test("gate ON: modules manifest must be non-empty with valid paths/blocks")
    func modulesInvalid() {
        let empty = bonsaiConfig(["modules": []])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: empty, hadamardData: hadamardData(), gateEnabled: true),
            containing: "modules")

        let badBlock = bonsaiConfig([
            "modules": [
                ["path": "lm_head", "block": 256, "embedding": false],
                ["path": "model.embed_tokens", "block": 1024, "embedding": true],
            ]
        ])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: badBlock, hadamardData: hadamardData(), gateEnabled: true),
            containing: "block")

        let emptyPath = bonsaiConfig([
            "modules": [
                ["path": "", "block": 1024, "embedding": true]
            ]
        ])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: emptyPath, hadamardData: hadamardData(), gateEnabled: true),
            containing: "non-empty path")

        let mixedBlocks = bonsaiConfig([
            "modules": [
                ["path": "lm_head", "block": 512, "embedding": false],
                ["path": "model.embed_tokens", "block": 1024, "embedding": true],
            ]
        ])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: mixedBlocks, hadamardData: hadamardData(), gateEnabled: true),
            containing: "same block size")

        let noEmbedding = bonsaiConfig([
            "modules": [
                ["path": "lm_head", "block": 1024, "embedding": false]
            ]
        ])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: noEmbedding, hadamardData: hadamardData(), gateEnabled: true),
            containing: "embedding module")
    }

    @Test("gate ON: missing or unreadable hadamard.json is invalid")
    func missingHadamardInvalid() {
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: bonsaiConfig(), hadamardData: nil, gateEnabled: true),
            containing: "hadamard.json is missing or not JSON")
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: bonsaiConfig(), hadamardData: Data("garbage".utf8),
                gateEnabled: true),
            containing: "hadamard.json is missing or not JSON")
    }

    @Test("gate ON: hadamard version/block/transform/sign-mode must match the pin")
    func hadamardContractInvalid() {
        let version = hadamardData(["prism.hadamard.version": 2])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: bonsaiConfig(), hadamardData: version, gateEnabled: true),
            containing: "prism.hadamard.version")

        let block = hadamardData(["prism.hadamard.block_size": 2048])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: bonsaiConfig(), hadamardData: block, gateEnabled: true),
            containing: "prism.hadamard.block_size")

        let transform = hadamardData(["prism.hadamard.transform": "fast-walsh"])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: bonsaiConfig(), hadamardData: transform, gateEnabled: true),
            containing: "prism.hadamard.transform")

        let signMode = hadamardData(["prism.hadamard.sign_mode": "implicit"])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: bonsaiConfig(), hadamardData: signMode, gateEnabled: true),
            containing: "prism.hadamard.sign_mode")
    }

    @Test("gate ON: weight/inverse-weight name lists must be non-empty")
    func hadamardNameListsInvalid() {
        let noWeights = hadamardData(["prism.hadamard.weight_names": []])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: bonsaiConfig(), hadamardData: noWeights, gateEnabled: true),
            containing: "weight_names")

        let noInverse = hadamardData(["prism.hadamard.inverse_weight_names": []])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: bonsaiConfig(), hadamardData: noInverse, gateEnabled: true),
            containing: "inverse_weight_names")
    }

    @Test("gate ON: sign widths must sum to the sign value count")
    func signWidthCountMismatchInvalid() {
        let mismatch = hadamardData(["prism.hadamard.sign_widths": [3]])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: bonsaiConfig(), hadamardData: mismatch, gateEnabled: true),
            containing: "must sum")
    }

    @Test("gate ON: sign values must be exactly ±1")
    func signValuesInvalid() {
        let badValues = hadamardData(
            ["prism.hadamard.sign_values": [1.0, -1.0, 2.0, -1.0]])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: bonsaiConfig(), hadamardData: badValues, gateEnabled: true),
            containing: "±1")
    }

    @Test("gate ON: gdn_v_grouped must be true")
    func gdnVGroupedInvalid() {
        let ungrouped = hadamardData(["prism.hadamard.gdn_v_grouped": false])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: bonsaiConfig(), hadamardData: ungrouped, gateEnabled: true),
            containing: "gdn_v_grouped")
    }

    @Test("gate ON: manifest probe type corruption fails as manifest invalid")
    func probeTypeCorruptionInvalid() {
        // Identity (root model_type) is readable, but the Bonsai probe cannot
        // decode → gate ON must fail as manifest invalid, never fall through.
        let config = bonsaiConfig(["schema_version": "two"])
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: config, hadamardData: hadamardData(), gateEnabled: true),
            containing: "did not decode")
    }

    // MARK: 4. Environment gate contract

    @Test("gate env var is unambiguously the documented VMLX_BONSAI_PRISM_HADAMARD=1")
    func environmentGateContract() {
        #expect(
            PrismBonsaiPortability.environmentVariableName
                == "VMLX_BONSAI_PRISM_HADAMARD")
        // The gate is read exactly like the repo's other `VMLX_*`/`_FORCE_`
        // env gates: present and exactly "1" enables, everything else keeps
        // it off. The pure decide() arms cover the semantics; this documents
        // the wiring.
        #expect(
            PrismBonsaiPortability.isEnabled
                == (ProcessInfo.processInfo.environment[
                    PrismBonsaiPortability.environmentVariableName] == "1"))
    }
}