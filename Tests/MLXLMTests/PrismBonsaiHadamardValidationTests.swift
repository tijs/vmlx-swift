// Copyright © 2026
//
// Deterministic hardening tests for the Bonsai 2 Prism-Hadamard
// manifest/sign/shape validation and the real install/no-unused-key path
// (Source/MLXNN/PrismBonsaiHadamard.swift + MLXLMCommon/Load.swift
// bonsaiTransform install seam). No model weights, no network, no Metal
// server, no Bonsai payload: everything runs in memory against synthetic
// tensors shaped after the pinned pack
// `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` @ 3f926b415992eaa2ae9dd7b573706494d6bbf787.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import MLXLLM
@testable import MLXLMCommon

@Suite("Prism Bonsai Hadamard validation + install path")
struct PrismBonsaiHadamardValidationTests {

    // MARK: Helpers

    private func validConfigData(_ overrides: [String: Any] = [:]) -> Data {
        var dict: [String: Any] = [
            "schema_version": 2,
            "model_type": "prism_hadamard_qwen35",
            "requires_runtime": "runtime/artifact.py",
            "hadamard_config": "hadamard.json",
            "tensor_namespace": "mlx-vlm-qwen3_5",
            "gdn_activation_layout": "grouped",
            "base_model_type": "qwen3_5",
            "quantization": ["bits": 2, "group_size": 128, "mode": "affine"],
            "modules": [
                [
                    "path": "lm_head", "block": 1024, "embedding": false,
                    "dtype": "float16",
                ],
                [
                    "path": "model.layers.0.self_attn.q_proj", "block": 1024,
                    "embedding": false, "dtype": "float16",
                ],
                [
                    "path": "model.embed_tokens", "block": 1024,
                    "embedding": true, "dtype": "float16",
                ],
            ],
        ]
        for (key, value) in overrides { dict[key] = value }
        return try! JSONSerialization.data(withJSONObject: dict)
    }

    private func validHadamardData(_ overrides: [String: Any] = [:]) -> Data {
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

    private func expectBuildError(
        _ build: () throws -> PrismBonsaiHadamardPlan,
        containing fragment: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        do {
            _ = try build()
            Issue.record(
                "expected plan build to throw containing \(fragment)",
                sourceLocation: sourceLocation)
        } catch let error as PrismBonsaiHadamardPlan.BuildError {
            #expect(
                error.reason.contains(fragment),
                "reason \(error.reason) misses \(fragment)",
                sourceLocation: sourceLocation)
        } catch {
            Issue.record(
                "unexpected error type \(error)", sourceLocation: sourceLocation)
        }
    }

    /// A valid module triplet: `lm_head` (packed linear), `q_proj` (packed
    /// linear), `embed_tokens` (packed embedding), plus an ordinary
    /// `gate_proj` linear outside the manifest so the install path also
    /// exercises the final noUnusedKeys update over untouched weights.
    /// All projection widths are 1024 so the plan's block-1024 transform
    /// contract is satisfied by the packed tensors.
    private func makeMiniModel(qProjInput: Int = 1024) -> BonsaiMiniModel {
        BonsaiMiniModel(
            lmHead: Linear(1024, 64),
            body: BonsaiMiniBody(
                layers: [
                    BonsaiMiniLayer(
                        selfAttn: BonsaiMiniSelfAttn(
                            qProj: Linear(qProjInput, 256)),
                        mlp: BonsaiMiniMLP(
                            gateProj: Linear(1024, 256)))],
                embedTokens: Embedding(embeddingCount: 16, dimensions: 1024)))
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

    /// The install-valid weight dictionary (stripped post-sanitize forms).
    private func makeInstallWeights() -> [String: MLXArray] {
        let lmHead = packedTensors(out: 64, inputWidth: 1024, seed: 21)
        let qProj = packedTensors(out: 256, inputWidth: 1024, seed: 22)
        let embed = packedTensors(out: 16, inputWidth: 1024, seed: 23)
        return [
            "lm_head.weight": lmHead.weight,
            "lm_head.scales": lmHead.scales,
            "lm_head.biases": lmHead.biases,
            "lm_head.signs": lmHead.signs,
            "model.layers.0.self_attn.q_proj.weight": qProj.weight,
            "model.layers.0.self_attn.q_proj.scales": qProj.scales,
            "model.layers.0.self_attn.q_proj.biases": qProj.biases,
            "model.layers.0.self_attn.q_proj.signs": qProj.signs,
            "model.embed_tokens.weight": embed.weight,
            "model.embed_tokens.scales": embed.scales,
            "model.embed_tokens.biases": embed.biases,
            "model.embed_tokens.signs": embed.signs,
            "model.layers.0.mlp.gate_proj.weight": MLXRandom.normal([256, 1024]),
            "model.layers.0.mlp.gate_proj.bias": MLXRandom.normal([256]),
        ]
    }

    private func leafTypes(_ model: BonsaiMiniModel) -> [String: String] {
        Dictionary(
            uniqueKeysWithValues: model.leafModules().flattened().map {
                ($0.0, String(describing: type(of: $0.1)))
            })
    }

    private func expectInstallToThrow(_ install: () throws -> Void) {
        do {
            try install()
            Issue.record("expected install to throw")
        } catch {
            #expect(error is PrismBonsaiInstall.Error)
        }
    }

    // MARK: 1. Manifest metadata boundaries (plan builder)

    @Test("plan build rejects duplicate modules[] paths")
    func manifestRejectsDuplicateModulePaths() {
        let modules: [[String: Any]] = [
            ["path": "lm_head", "block": 1024, "embedding": false, "dtype": "float16"],
            ["path": "lm_head", "block": 1024, "embedding": false, "dtype": "float16"],
            ["path": "model.embed_tokens", "block": 1024, "embedding": true, "dtype": "float16"],
        ]
        expectBuildError({
            try PrismBonsaiHadamardPlan(
                configData: validConfigData(["modules": modules]),
                hadamardData: validHadamardData())
        }, containing: "unique")
    }

    @Test("plan build rejects multiple embedding modules")
    func manifestRejectsMultipleEmbeddings() {
        let modules: [[String: Any]] = [
            ["path": "lm_head", "block": 1024, "embedding": false, "dtype": "float16"],
            ["path": "model.embed_tokens", "block": 1024, "embedding": true, "dtype": "float16"],
            ["path": "model.embed_tokens2", "block": 1024, "embedding": true, "dtype": "float16"],
        ]
        expectBuildError({
            try PrismBonsaiHadamardPlan(
                configData: validConfigData(["modules": modules]),
                hadamardData: validHadamardData())
        }, containing: "exactly one embedding")
    }

    @Test("plan build rejects non-positive sign widths")
    func manifestRejectsNonPositiveSignWidths() {
        // [0, 4] and [-2, 6] both sum to 4 == values.count, so the sum check
        // alone would accept them; positivity must fail deterministically.
        let unitValues = [1.0, -1.0, 1.0, -1.0]
        for widths in [[0, 4], [-2, 6]] {
            expectBuildError({
                try PrismBonsaiHadamardPlan(
                    configData: validConfigData(),
                    hadamardData: validHadamardData([
                        "prism.hadamard.sign_widths": widths,
                        "prism.hadamard.sign_values": unitValues,
                    ]))
            }, containing: "positive widths")
        }
    }

    @Test("plan build rejects overlapping weight_names / inverse_weight_names")
    func manifestRejectsOverlappingWeightNames() {
        expectBuildError({
            try PrismBonsaiHadamardPlan(
                configData: validConfigData(),
                hadamardData: validHadamardData([
                    "prism.hadamard.weight_names": [
                        "language_model.lm_head.weight",
                        "language_model.model.layers.0.self_attn.q_proj.weight",
                        "language_model.model.embed_tokens.weight",
                    ]
                ]))
        }, containing: "must not overlap")
    }

    @Test("plan build does not cross-check modules[] paths against weight_names")
    func planDoesNotRequireNameListPartition() throws {
        // The pinned pack's modules[] paths and hadamard.json weight_names
        // use different names for the same modules (self_attn.q_proj vs
        // linear_attn.in_proj_qkv), so a valid pack must still build
        // even when the two lists do not line up name-for-name.
        let plan = try PrismBonsaiHadamardPlan(
            configData: validConfigData(),
            hadamardData: validHadamardData([
                "prism.hadamard.weight_names": [
                    "language_model.lm_head.weight",
                    "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
                ]
            ]))
        #expect(plan.entries.count == 3)
    }

    @Test("namespace- and suffix-tolerant normalization is exact")
    func normalizationIsTolerantAndExact() {
        #expect(
            PrismBonsaiHadamardPlan.normalizedModuleBase(
                "language_model.model.embed_tokens.weight") == "embed_tokens")
        #expect(
            PrismBonsaiHadamardPlan.normalizedModuleBase(
                "language_model.lm_head.weight") == "lm_head")
        #expect(
            PrismBonsaiHadamardPlan.normalizedModuleBase(
                "model.layers.0.self_attn.q_proj") == "layers.0.self_attn.q_proj")
        #expect(
            PrismBonsaiHadamardPlan.normalizedModuleBase(
                "embed_tokens") == "embed_tokens")
    }

    // MARK: 2. Factory gate mirror

    private func expectInvalid(
        _ decision: PrismBonsaiPortability.Decision,
        containing fragment: String
    ) {
        guard case .gateOnManifestInvalid(let error) = decision else {
            Issue.record("expected gateOnManifestInvalid, got \(decision)")
            return
        }
        #expect(
            error.reason.contains(fragment),
            "reason \(error.reason) misses \(fragment)")
    }

    @Test("factory gate mirrors the hardened manifest checks")
    func gateMirrorsHardenedChecks() {
        let duplicatePaths: [[String: Any]] = [
            ["path": "lm_head", "block": 1024, "embedding": false],
            ["path": "lm_head", "block": 1024, "embedding": false],
            ["path": "model.embed_tokens", "block": 1024, "embedding": true],
        ]
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: validConfigData(["modules": duplicatePaths]),
                hadamardData: validHadamardData(), gateEnabled: true),
            containing: "unique")

        let twoEmbeddings: [[String: Any]] = [
            ["path": "lm_head", "block": 1024, "embedding": false],
            ["path": "model.embed_tokens", "block": 1024, "embedding": true],
            ["path": "model.embed_tokens2", "block": 1024, "embedding": true],
        ]
        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: validConfigData(["modules": twoEmbeddings]),
                hadamardData: validHadamardData(), gateEnabled: true),
            containing: "exactly one embedding")

        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: validConfigData(),
                hadamardData: validHadamardData([
                    "prism.hadamard.sign_widths": [0, 4]
                ]),
                gateEnabled: true),
            containing: "positive widths")

        expectInvalid(
            PrismBonsaiPortability.decide(
                configData: validConfigData(),
                hadamardData: validHadamardData([
                    "prism.hadamard.weight_names": [
                        "language_model.lm_head.weight",
                        "language_model.model.layers.0.self_attn.q_proj.weight",
                        "language_model.model.embed_tokens.weight",
                    ]
                ]),
                gateEnabled: true),
            containing: "must not overlap")
    }

    // MARK: 3. Packed tensor shape validation

    @Test("packed tensor validation accepts the contract shapes")
    func packedTensorValidationAccepts() throws {
        let width = try HadamardPackedCheck.validatePackedTensors(
            weight: MLXArray.zeros([64, 8]),  // 2-bit: in = 8 * 16 = 128
            scales: MLXArray.zeros([64, 1]),
            biases: MLXArray.zeros([64, 1]),
            groupSize: 128, bits: 2, block: 4)
        #expect(width == 128)
        // Full-size block-1024 contract.
        let width1024 = try HadamardPackedCheck.validatePackedTensors(
            weight: MLXArray.zeros([64, 64]),  // in = 1024
            scales: MLXArray.zeros([64, 8]),
            biases: nil,
            groupSize: 128, bits: 2, block: 1024)
        #expect(width1024 == 1024)
    }

    @Test("packed tensor validation rejects malformed shapes")
    func packedTensorValidationRejects() {
        func expectReject(
            _ weight: MLXArray, _ scales: MLXArray, _ biases: MLXArray?,
            block: Int = 4, containing fragment: String,
            sourceLocation: SourceLocation = #_sourceLocation
        ) {
            do {
                _ = try HadamardPackedCheck.validatePackedTensors(
                    weight: weight, scales: scales, biases: biases,
                    groupSize: 128, bits: 2, block: block)
                Issue.record(
                    "expected rejection containing \(fragment)",
                    sourceLocation: sourceLocation)
            } catch let error as PrismBonsaiInstall.Error {
                #expect(
                    error.reason.contains(fragment),
                    "reason \(error.reason) misses \(fragment)",
                    sourceLocation: sourceLocation)
            } catch {
                Issue.record(
                    "unexpected error type \(error)",
                    sourceLocation: sourceLocation)
            }
        }
        expectReject(MLXArray.zeros([2, 8, 8]), .zeros([2, 8, 1]), nil,
            containing: "2-D")
        expectReject(MLXArray.zeros([64, 8]), .zeros([64, 2]), .zeros([64, 1]),
            containing: "scales shape")
        expectReject(MLXArray.zeros([64, 8]), .zeros([64, 1]), .zeros([64, 3]),
            containing: "biases shape")
        // input 96 is not a multiple of group 128
        expectReject(MLXArray.zeros([64, 6]), .zeros([64, 1]), nil,
            containing: "divisible by group")
        // block 1024 does not divide the 128-wide activation
        expectReject(MLXArray.zeros([64, 8]), .zeros([64, 1]), nil,
            block: 1024, containing: "does not divide activation width")
    }

    // MARK: 4. Resolve boundaries

    @Test("resolve rejects entries collapsing onto the same model leaf")
    func resolveRejectsCollidingModulePaths() throws {
        // A hand-built (cross-module constructible) plan whose two entries
        // normalize to the same leaf — the install must refuse the ambiguity
        // even when the config builder is bypassed.
        let plan = PrismBonsaiHadamardPlan(
            block: 1024, groupSize: 128, bits: 2,
            entries: [
                PrismBonsaiHadamardPlan.Entry(
                    checkpointBase: "model.embed_tokens",
                    role: .inversePackedEmbedding, block: 1024, dtypeName: nil),
                PrismBonsaiHadamardPlan.Entry(
                    checkpointBase: "embed_tokens",
                    role: .inversePackedEmbedding, block: 1024, dtypeName: nil),
            ],
            foldedWeightNames: [],
            inverseWeightNames: ["model.embed_tokens.weight"],
            signWidths: [128], signValuesCount: 128)
        do {
            _ = try PrismBonsaiInstall.resolve(
                plan: plan,
                leafModulePaths: ["model.embed_tokens"],
                checkpointKeys: [
                    "model.embed_tokens.weight",
                    "model.embed_tokens.scales",
                    "model.embed_tokens.biases",
                    "model.embed_tokens.signs",
                ])
            Issue.record("expected resolve to reject colliding module paths")
        } catch let error as PrismBonsaiInstall.Error {
            #expect(error.reason.contains("distinct module paths"))
        }
    }

    // MARK: 5. Real install path + no-unused-key coverage

    @Test("install replaces leaves, consumes packed keys and passes noUnusedKeys")
    func installReplacesLeavesConsumesKeysAndPassesNoUnusedKeys() throws {
        let plan = try PrismBonsaiHadamardPlan(
            configData: validConfigData(), hadamardData: validHadamardData())
        let model = makeMiniModel()
        var weights = makeInstallWeights()

        try installBonsaiPrismHadamard(plan, model: model, weights: &weights)

        let leaves = leafTypes(model)
        #expect(leaves["lm_head"] == "HadamardPackedLinear")
        #expect(leaves["model.layers.0.self_attn.q_proj"] == "HadamardPackedLinear")
        #expect(leaves["model.embed_tokens"] == "HadamardPackedEmbedding")
        // The ordinary leaf outside the manifest is untouched.
        #expect(leaves["model.layers.0.mlp.gate_proj"] == "Linear")

        // Every packed + signs key was consumed; ordinary keys remain.
        for key in [
            "lm_head.weight", "lm_head.scales", "lm_head.biases", "lm_head.signs",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.scales",
            "model.layers.0.self_attn.q_proj.biases",
            "model.layers.0.self_attn.q_proj.signs",
            "model.embed_tokens.weight", "model.embed_tokens.scales",
            "model.embed_tokens.biases", "model.embed_tokens.signs",
        ] {
            #expect(weights[key] == nil, "key \(key) must be consumed")
        }
        #expect(weights["model.layers.0.mlp.gate_proj.weight"] != nil)
        #expect(weights["model.layers.0.mlp.gate_proj.bias"] != nil)

        // Tail of loadWeights: the final update over the remaining keys with
        // verify: [.noUnusedKeys] — the packed/consumed keys are gone, so
        // this passes while the ordinary keys are still applied.
        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: [.noUnusedKeys])
    }

    @Test("install fails before any swap when a sign vector is missing")
    func installFailsBeforeSwapOnMissingSigns() throws {
        let plan = try PrismBonsaiHadamardPlan(
            configData: validConfigData(), hadamardData: validHadamardData())
        let model = makeMiniModel()
        var weights = makeInstallWeights()
        weights["model.embed_tokens.signs"] = nil

        expectInstallToThrow({
            try installBonsaiPrismHadamard(plan, model: model, weights: &weights)
        })

        // No partial load: every leaf is still the original module type and
        // no key was consumed.
        let leaves = leafTypes(model)
        #expect(leaves["lm_head"] == "Linear")
        #expect(leaves["model.layers.0.self_attn.q_proj"] == "Linear")
        #expect(leaves["model.embed_tokens"] == "Embedding")
        #expect(weights["lm_head.signs"] != nil)
    }

    @Test("install fails before any swap on an affine tensor outside the manifest")
    func installFailsBeforeSwapOnStrayAffineTensor() throws {
        let plan = try PrismBonsaiHadamardPlan(
            configData: validConfigData(), hadamardData: validHadamardData())
        let model = makeMiniModel()
        var weights = makeInstallWeights()
        weights["model.layers.0.mlp.gate_proj.scales"] = MLXArray.zeros([256, 1])

        expectInstallToThrow({
            try installBonsaiPrismHadamard(plan, model: model, weights: &weights)
        })

        let leaves = leafTypes(model)
        #expect(leaves["lm_head"] == "Linear")
        #expect(leaves["model.layers.0.self_attn.q_proj"] == "Linear")
        #expect(leaves["model.embed_tokens"] == "Embedding")
    }

    @Test("install fails before any swap on a tensor/module width mismatch")
    func installFailsBeforeSwapOnLeafWidthMismatch() throws {
        let plan = try PrismBonsaiHadamardPlan(
            configData: validConfigData(), hadamardData: validHadamardData())
        // q_proj leaf input width 64, but its packed tensors unpack to 128.
        let model = makeMiniModel(qProjInput: 64)
        var weights = makeInstallWeights()

        do {
            try installBonsaiPrismHadamard(plan, model: model, weights: &weights)
            Issue.record("expected install to throw on leaf width mismatch")
        } catch let error as PrismBonsaiInstall.Error {
            #expect(error.reason.contains("does not match module leaf width"))
        }

        let leaves = leafTypes(model)
        #expect(leaves["lm_head"] == "Linear")
        #expect(leaves["model.embed_tokens"] == "Embedding")
    }

    @Test("install fails before any swap when the packed output count is wrong but input width is valid")
    func installFailsBeforeSwapOnPackedOutputMismatch() throws {
        let plan = try PrismBonsaiHadamardPlan(
            configData: validConfigData(), hadamardData: validHadamardData())
        let model = makeMiniModel()
        var weights = makeInstallWeights()
        // lm_head leaf is Linear(1024, 64) -> weight [64, 1024]. Swap in a
        // pack whose 1024-wide input is valid but whose row count (32) does
        // not match the leaf's 64 outputs: the input-width-only check would
        // accept these tensors and install a wrong-shaped lm_head.
        let wrong = packedTensors(out: 32, inputWidth: 1024, seed: 31)
        weights["lm_head.weight"] = wrong.weight
        weights["lm_head.scales"] = wrong.scales
        weights["lm_head.biases"] = wrong.biases

        do {
            try installBonsaiPrismHadamard(plan, model: model, weights: &weights)
            Issue.record("expected install to throw on packed output mismatch")
        } catch let error as PrismBonsaiInstall.Error {
            #expect(
                error.reason.contains("does not match module leaf output count"))
        }

        let leaves = leafTypes(model)
        #expect(leaves["lm_head"] == "Linear")
        #expect(leaves["model.layers.0.self_attn.q_proj"] == "Linear")
        #expect(leaves["model.embed_tokens"] == "Embedding")
        // No key was consumed, including lm_head's own packed keys.
        #expect(weights["lm_head.weight"] != nil)
        #expect(weights["lm_head.scales"] != nil)
        #expect(weights["lm_head.biases"] != nil)
        #expect(weights["lm_head.signs"] != nil)
    }

    @Test("install fails before any swap when the packed embedding row count is wrong")
    func installFailsBeforeSwapOnPackedEmbeddingRowMismatch() throws {
        let plan = try PrismBonsaiHadamardPlan(
            configData: validConfigData(), hadamardData: validHadamardData())
        let model = makeMiniModel()
        var weights = makeInstallWeights()
        // embed_tokens leaf is Embedding(16, 1024) -> weight [16, 1024].
        // Swap in a pack whose 1024-wide input is valid but which stores
        // only 8 rotated rows: the vocabulary/row count must still match.
        let wrong = packedTensors(out: 8, inputWidth: 1024, seed: 32)
        weights["model.embed_tokens.weight"] = wrong.weight
        weights["model.embed_tokens.scales"] = wrong.scales
        weights["model.embed_tokens.biases"] = wrong.biases

        do {
            try installBonsaiPrismHadamard(plan, model: model, weights: &weights)
            Issue.record("expected install to throw on embedding row mismatch")
        } catch let error as PrismBonsaiInstall.Error {
            #expect(
                error.reason.contains("does not match module leaf output count"))
        }

        let leaves = leafTypes(model)
        #expect(leaves["lm_head"] == "Linear")
        #expect(leaves["model.embed_tokens"] == "Embedding")
        #expect(weights["model.embed_tokens.weight"] != nil)
        #expect(weights["model.embed_tokens.scales"] != nil)
        #expect(weights["model.embed_tokens.biases"] != nil)
    }

    @Test("a later-entry failure keeps earlier checkpoint keys and swaps nothing")
    func laterEntryFailurePreservesEarlierKeysAndSwapsNothing() throws {
        let plan = try PrismBonsaiHadamardPlan(
            configData: validConfigData(), hadamardData: validHadamardData())
        let model = makeMiniModel()
        var weights = makeInstallWeights()
        // The last resolved entry (model.embed_tokens, modules[2]) carries a
        // complete, resolvable tensor set whose output row count (8) does not
        // match the leaf's 16-row vocabulary: resolution succeeds, the two
        // earlier entries (lm_head, q_proj) fully pass their in-loop
        // validation, and only then does the row-count check fail here —
        // inside the install loop — on the wrong packed output count.
        let wrongEmbed = packedTensors(out: 8, inputWidth: 1024, seed: 32)
        weights["model.embed_tokens.weight"] = wrongEmbed.weight
        weights["model.embed_tokens.scales"] = wrongEmbed.scales
        weights["model.embed_tokens.biases"] = wrongEmbed.biases

        expectInstallToThrow({
            try installBonsaiPrismHadamard(plan, model: model, weights: &weights)
        })

        let leaves = leafTypes(model)
        #expect(leaves["lm_head"] == "Linear")
        #expect(leaves["model.layers.0.self_attn.q_proj"] == "Linear")
        #expect(leaves["model.embed_tokens"] == "Embedding")
        // Transactional consumption: every key of the earlier entries and of
        // the failing last entry is still available on failure.
        for key in [
            "lm_head.weight", "lm_head.scales", "lm_head.biases", "lm_head.signs",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.scales",
            "model.layers.0.self_attn.q_proj.biases",
            "model.layers.0.self_attn.q_proj.signs",
            "model.embed_tokens.weight", "model.embed_tokens.scales",
            "model.embed_tokens.biases", "model.embed_tokens.signs",
        ] {
            #expect(weights[key] != nil, "key \(key) must still be available")
        }
    }

    @Test("install fails before any swap on non-unit sign values")
    func installFailsBeforeSwapOnNonUnitSigns() throws {
        let plan = try PrismBonsaiHadamardPlan(
            configData: validConfigData(), hadamardData: validHadamardData())
        let model = makeMiniModel()
        var weights = makeInstallWeights()
        weights["lm_head.signs"] = MLXArray((0..<1024).map { _ in Float(2) })

        expectInstallToThrow({
            try installBonsaiPrismHadamard(plan, model: model, weights: &weights)
        })

        let leaves = leafTypes(model)
        #expect(leaves["lm_head"] == "Linear")
        #expect(leaves["model.embed_tokens"] == "Embedding")
    }

    @Test("install fails before any swap on a malformed scales shape")
    func installFailsBeforeSwapOnBadScalesShape() throws {
        let plan = try PrismBonsaiHadamardPlan(
            configData: validConfigData(), hadamardData: validHadamardData())
        let model = makeMiniModel()
        var weights = makeInstallWeights()
        weights["lm_head.scales"] = MLXArray.zeros([64, 2])

        expectInstallToThrow({
            try installBonsaiPrismHadamard(plan, model: model, weights: &weights)
        })

        let leaves = leafTypes(model)
        #expect(leaves["lm_head"] == "Linear")
        #expect(leaves["model.embed_tokens"] == "Embedding")
    }

    @Test("a leftover non-module key fails the final noUnusedKeys update")
    func leftoverKeyFailsFinalNoUnusedKeysUpdate() throws {
        let plan = try PrismBonsaiHadamardPlan(
            configData: validConfigData(), hadamardData: validHadamardData())
        let model = makeMiniModel()
        var weights = makeInstallWeights()
        try installBonsaiPrismHadamard(plan, model: model, weights: &weights)

        // Simulate a consumption miss: a key the plan does not cover (here
        // under an installed leaf) must be rejected by the final noUnusedKeys
        // update instead of silently ignored. Note a `.signs` key would NOT
        // be rejected — `signs` is a registered parameter of the packed
        // modules — which is exactly why the install must consume those keys.
        weights["model.embed_tokens.junk"] = MLXArray.zeros([1])

        let parameters = ModuleParameters.unflattened(weights)
        do {
            try model.update(parameters: parameters, verify: [.noUnusedKeys])
            Issue.record("expected noUnusedKeys to reject the leftover junk key")
        } catch {
            #expect(error is UpdateError)
        }
    }
}

// MARK: - In-memory mini model (Module + LanguageModel, no weights on disk)

private final class BonsaiMiniModel: Module, LanguageModel {

    @ModuleInfo(key: "lm_head") var lmHead: Linear
    @ModuleInfo(key: "model") var body: BonsaiMiniBody

    init(lmHead: Linear, body: BonsaiMiniBody) {
        self._lmHead.wrappedValue = lmHead
        self._body.wrappedValue = body
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([1, 1, 1])
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(LMInput.Text(tokens: input.text.tokens))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        []
    }
}

private final class BonsaiMiniBody: Module {

    @ModuleInfo(key: "layers") var layers: [BonsaiMiniLayer]
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    init(layers: [BonsaiMiniLayer], embedTokens: Embedding) {
        self._layers.wrappedValue = layers
        self._embedTokens.wrappedValue = embedTokens
        super.init()
    }
}

private final class BonsaiMiniLayer: Module {

    @ModuleInfo(key: "self_attn") var selfAttn: BonsaiMiniSelfAttn
    @ModuleInfo(key: "mlp") var mlp: BonsaiMiniMLP

    init(selfAttn: BonsaiMiniSelfAttn, mlp: BonsaiMiniMLP) {
        self._selfAttn.wrappedValue = selfAttn
        self._mlp.wrappedValue = mlp
        super.init()
    }
}

private final class BonsaiMiniSelfAttn: Module {

    @ModuleInfo(key: "q_proj") var qProj: Linear

    init(qProj: Linear) {
        self._qProj.wrappedValue = qProj
        super.init()
    }
}

private final class BonsaiMiniMLP: Module {

    @ModuleInfo(key: "gate_proj") var gateProj: Linear

    init(gateProj: Linear) {
        self._gateProj.wrappedValue = gateProj
        super.init()
    }
}