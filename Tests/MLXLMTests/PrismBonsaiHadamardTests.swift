// Copyright © 2026

import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXLLM

/// Deterministic tests for the Bonsai 2 Prism-Hadamard transform seam
/// (`Source/MLXNN/PrismBonsaiHadamard.swift` + the `MLXLMCommon/Load.swift`
/// `bonsaiTransform:` install path). No model weights, no network, no Metal
/// server, no Bonsai payload: transform math is checked against explicit
/// reference matrices, packed modules against the public MLX quantized
/// primitives, and plan/install resolution against in-memory fixtures shaped
/// after the pinned pack `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` @
/// `3f926b415992eaa2ae9dd7b573706494d6bbf787`.
@Suite("Prism Bonsai Hadamard transform seam")
struct PrismBonsaiHadamardTests {

    // MARK: Helpers

    private func approx(_ a: MLXArray, _ b: MLXArray, tol: Float = 1e-4) -> Bool {
        MLX.max(MLX.abs(a - b)).item(Float.self) < tol
    }

    /// Explicit H4/2 Sylvester–Walsh–Hadamard reference (orthonormal rows).
    private func hadamard4Matrix() -> MLXArray {
        MLXArray([
            1.0, 1.0, 1.0, 1.0,
            1.0, -1.0, 1.0, -1.0,
            1.0, 1.0, -1.0, -1.0,
            1.0, -1.0, -1.0, 1.0,
        ] as [Float]).reshaped([4, 4]) * Float(0.5)
    }

    private func referenceFWHT(_ x: MLXArray, block: Int, signs: [Float], inverse: Bool)
        -> MLXArray
    {
        let h = hadamard4Matrix()
        var y = x.reshaped([-1, block]).asType(.float32)
        if !inverse {
            y = y * MLXArray(signs).asType(.float32)
        }
        y = y.matmul(h.T)
        if inverse {
            y = y * MLXArray(signs).asType(.float32)
        }
        return y.reshaped(x.shape).asType(x.dtype)
    }

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
            "text_config": ["model_type": "qwen3_5_text"],
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

    // MARK: 1. Transform math (explicit reference matrices)

    @Test("forward FWHT matches the explicit H4 reference")
    func forwardFWHTMatchesReference() {
        let x = MLXArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] as [Float]).reshaped([2, 4])
        let signs: [Float] = [1.0, -1.0, 1.0, -1.0]
        let actual = hadamardFWHT(
            x, block: 4, signs: MLXArray(signs), inverse: false)
        let expected = referenceFWHT(x, block: 4, signs: signs, inverse: false)
        #expect(actual.shape == x.shape)
        #expect(approx(actual, expected))
    }

    @Test("inverse FWHT matches the explicit H4 reference")
    func inverseFWHTMatchesReference() {
        let x = MLXArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] as [Float]).reshaped([2, 4])
        let signs: [Float] = [1.0, -1.0, 1.0, -1.0]
        let actual = hadamardFWHT(
            x, block: 4, signs: MLXArray(signs), inverse: true)
        let expected = referenceFWHT(x, block: 4, signs: signs, inverse: true)
        #expect(actual.shape == x.shape)
        #expect(approx(actual, expected))
    }

    @Test("forward then inverse FWHT is the identity")
    func forwardInverseRoundTrip() {
        let x = MLXArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] as [Float]).reshaped([2, 4])
        let signs = MLXArray([1.0, -1.0, -1.0, 1.0] as [Float])
        let roundTripped = hadamardFWHT(
            hadamardFWHT(x, block: 4, signs: signs, inverse: false),
            block: 4, signs: signs, inverse: true)
        #expect(approx(roundTripped, x, tol: 1e-5))
    }

    @Test("transform of all-ones with unit signs is a single spike")
    func allOnesTransform() {
        // H4/2 applied to [1,1,1,1] gives [2,0,0,0].
        let x = MLXArray([1.0, 1.0, 1.0, 1.0] as [Float])
        let out = hadamardFWHT(
            x, block: 4, signs: MLXArray([1.0, 1.0, 1.0, 1.0] as [Float]), inverse: false)
        #expect(approx(out, MLXArray([2.0, 0.0, 0.0, 0.0] as [Float]), tol: 1e-5))
    }

    @Test("transform preserves dtype and multi-row shapes")
    func dtypeAndShapePreserved() {
        let x = MLXArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] as [Float]).reshaped([2, 4])
            .asType(.float16)
        let signs = MLXArray([1.0, -1.0, -1.0, 1.0] as [Float]).asType(.float16)
        let out = hadamardFWHT(x, block: 4, signs: signs, inverse: false)
        #expect(out.dtype == .float16)
        #expect(out.shape == [2, 4])
        // Batch + sequence dims (2, 2, 4).
        let batched = hadamardFWHT(
            MLXArray.zeros([2, 2, 4]), block: 4, signs: signs, inverse: false)
        #expect(batched.shape == [2, 2, 4])
    }

    @Test("HadamardActivation forward and inverse match the free function")
    func activationModuleMatchesFunction() {
        let activation = HadamardActivation(
            block: 4, signs: MLXArray([1.0, -1.0, -1.0, 1.0] as [Float]))
        let x = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float]).reshaped([1, 4])
        #expect(
            approx(
                activation(x),
                hadamardFWHT(
                    x, block: 4, signs: activation.signs!, inverse: false)))
        #expect(
            approx(
                activation.inverse(x),
                hadamardFWHT(
                    x, block: 4, signs: activation.signs!, inverse: true)))
        // block 0 → identity
        let identity = HadamardActivation(block: 0, signs: nil)
        #expect(approx(identity(x), x))
    }

    // MARK: 2. Packed linear semantics

    private func packedFixtures(
        inputWidth: Int = 128, outputWidth: Int = 256, seed: UInt64 = 7
    ) -> (packed: MLXArray, scales: MLXArray, biases: MLXArray?) {
        MLXRandom.seed(seed)
        let weight = MLXRandom.normal([outputWidth, inputWidth])
        let quantized = MLX.quantized(
            weight, groupSize: 128, bits: 2, mode: .affine)
        return (quantized.wq, quantized.scales, quantized.biases)
    }

    @Test("packed linear with block 0 equals the plain affine QuantizedLinear")
    func packedLinearWithoutTransformMatchesQuantizedLinear() {
        let (packed, scales, biases) = packedFixtures()
        let packedLinear = HadamardPackedLinear(
            weight: packed, scales: scales, biases: biases,
            groupSize: 128, bits: 2, block: 0, signs: nil)
        let plain = QuantizedLinear(
            weight: packed, scales: scales, biases: biases,
            groupSize: 128, bits: 2)
        let x = MLXRandom.normal([1, 128])
        #expect(approx(packedLinear(x), plain(x), tol: 1e-4))
    }

    @Test("packed linear applies fwht before the quantized matmul")
    func packedLinearTransformsActivations() {
        let (packed, scales, biases) = packedFixtures()
        let signsValues: [Float] = (0..<128).map { $0.isMultiple(of: 2) ? 1.0 : -1.0 }
        let signs = MLXArray(signsValues).asType(.float32)
        let packedLinear = HadamardPackedLinear(
            weight: packed, scales: scales, biases: biases,
            groupSize: 128, bits: 2, block: 4, signs: signs)
        let plain = QuantizedLinear(
            weight: packed, scales: scales, biases: biases,
            groupSize: 128, bits: 2)
        let x = MLXRandom.normal([1, 128])
        let expected = plain(
            hadamardFWHT(x, block: 4, signs: signs, inverse: false))
        #expect(approx(packedLinear(x), expected, tol: 1e-4))
        // shape: (1, 128) → (1, 256)
        #expect(packedLinear(x).shape == [1, 256])
    }

    @Test("packed linear shape reports the unpacked input width")
    func packedLinearUnpackedShape() {
        let (packed, scales, biases) = packedFixtures(inputWidth: 128, outputWidth: 256)
        let packedLinear = HadamardPackedLinear(
            weight: packed, scales: scales, biases: biases,
            groupSize: 128, bits: 2, block: 4,
            signs: MLXArray((0..<128).map { _ in Float(1) }).asType(.float32))
        #expect(packedLinear.shape == (256, 128))
        // Stored packed width is in/16 for 2-bit.
        #expect(packedLinear.weight.shape == [256, 8])
    }

    // MARK: 3. Packed embedding semantics

    @Test("packed embedding applies the inverse fwht after the lookup")
    func packedEmbeddingInverseTransform() {
        MLXRandom.seed(11)
        let table = MLXRandom.normal([64, 128])
        let quantized = MLX.quantized(
            table, groupSize: 128, bits: 2, mode: .affine)
        let signsValues: [Float] = (0..<128).map { $0.isMultiple(of: 3) ? -1.0 : 1.0 }
        let signs = MLXArray(signsValues).asType(.float32)

        let packed = HadamardPackedEmbedding(
            weight: quantized.0, scales: quantized.1, biases: quantized.2,
            groupSize: 128, bits: 2, block: 4, signs: signs)
        let plain = QuantizedEmbedding(
            weight: quantized.0, scales: quantized.1, biases: quantized.2,
            groupSize: 128, bits: 2)

        let indices = MLXArray([0, 7, 13, 42])  // [Int32]
        let expected = hadamardFWHT(
            plain(indices), block: 4, signs: signs, inverse: true)
        #expect(approx(packed(indices), expected, tol: 1e-4))
        #expect(packed(indices).shape == [4, 128])
    }

    @Test("packed embedding with block 0 equals QuantizedEmbedding")
    func packedEmbeddingWithoutTransformMatchesQuantized() {
        MLXRandom.seed(13)
        let table = MLXRandom.normal([64, 128])
        let quantized = MLX.quantized(
            table, groupSize: 128, bits: 2, mode: .affine)
        let packed = HadamardPackedEmbedding(
            weight: quantized.0, scales: quantized.1, biases: quantized.2,
            groupSize: 128, bits: 2, block: 0, signs: nil)
        let plain = QuantizedEmbedding(
            weight: quantized.0, scales: quantized.1, biases: quantized.2,
            groupSize: 128, bits: 2)
        let indices = MLXArray([0, 7, 13])
        #expect(approx(packed(indices), plain(indices), tol: 1e-4))
    }

    // MARK: 4. Sign validation

    @Test("sign validation accepts unit signs of the right width")
    func signValidationAccepts() throws {
        let signs = MLXArray(
            (0..<128).map { $0.isMultiple(of: 2) ? Float(1) : Float(-1) })
        let width = try HadamardPackedCheck.validateSigns(
            signs, packedWeightWidth: 8, bits: 2, block: 4)
        #expect(width == 128)
    }

    @Test("sign validation rejects wrong widths and non-unit values")
    func signValidationRejects() {
        let wrongWidth = MLXArray([1.0, -1.0, 1.0] as [Float])  // 3 != 8*16
        #expect(
            throws: PrismBonsaiInstall.Error.self
        ) {
            try HadamardPackedCheck.validateSigns(
                wrongWidth, packedWeightWidth: 8, bits: 2, block: 4)
        }
        let nonUnit = MLXArray((0..<128).map { _ in Float(0.5) })
        #expect(
            throws: PrismBonsaiInstall.Error.self
        ) {
            try HadamardPackedCheck.validateSigns(
                nonUnit, packedWeightWidth: 8, bits: 2, block: 4)
        }
        let blockMismatch = MLXArray((0..<128).map { _ in Float(1) })
        #expect(
            throws: PrismBonsaiInstall.Error.self
        ) {
            try HadamardPackedCheck.validateSigns(
                blockMismatch, packedWeightWidth: 8, bits: 2, block: 1024)
        }
    }

    // MARK: 5. Manifest / plan building

    @Test("valid manifest builds the plan with roles and block 1024")
    func planBuildsFromValidManifest() throws {
        let plan = try PrismBonsaiHadamardPlan(
            configData: validConfigData(), hadamardData: validHadamardData())
        #expect(plan.block == 1024)
        #expect(plan.groupSize == 128)
        #expect(plan.bits == 2)
        #expect(plan.entries.count == 3)
        #expect(
            plan.entry(forCheckpointBase: "lm_head")?.role
                == .forwardPackedLinear)
        #expect(
            plan.entry(forCheckpointBase: "model.embed_tokens")?.role
                == .inversePackedEmbedding)
        #expect(
            plan.entry(forCheckpointBase: "model.layers.0.self_attn.q_proj")?
                .role == .forwardPackedLinear)
        #expect(plan.signWidths == [4])
        #expect(plan.signValuesCount == 4)
        #expect(plan.foldedWeightNames.count == 2)
        #expect(plan.inverseWeightNames == ["language_model.model.embed_tokens.weight"])
    }

    @Test("plan build rejects a downgraded quant manifest")
    func planRejectsQuantizationMismatch() {
        expectBuildError({
            try PrismBonsaiHadamardPlan(
                configData: validConfigData([
                    "quantization": ["bits": 4, "group_size": 128, "mode": "affine"]
                ]),
                hadamardData: validHadamardData())
        }, containing: "quantization.bits must be 2")
    }

    @Test("plan build rejects non-uniform module blocks")
    func planRejectsMixedBlocks() {
        var modules = validConfigData()
        let dict = try! JSONSerialization.jsonObject(with: modules)
            as! [String: Any]
        var override = dict
        override["modules"] = [
            ["path": "lm_head", "block": 1024, "embedding": false, "dtype": "float16"],
            ["path": "model.embed_tokens", "block": 512, "embedding": true, "dtype": "float16"],
        ]
        modules = try! JSONSerialization.data(withJSONObject: override)
        expectBuildError({
            try PrismBonsaiHadamardPlan(
                configData: modules, hadamardData: validHadamardData())
        }, containing: "share the same block size")
    }

    @Test("plan build rejects a manifest without an embedding module")
    func planRejectsMissingEmbedding() {
        let noEmbedding = [
            ["path": "lm_head", "block": 1024, "embedding": false, "dtype": "float16"]
        ]
        expectBuildError({
            try PrismBonsaiHadamardPlan(
                configData: validConfigData(["modules": noEmbedding]),
                hadamardData: validHadamardData())
        }, containing: "must contain an embedding module")
    }

    @Test("plan build rejects non-±1 sign values and bad widths")
    func planRejectsBadSigns() {
        expectBuildError({
            try PrismBonsaiHadamardPlan(
                configData: validConfigData(),
                hadamardData: validHadamardData([
                    "prism.hadamard.sign_values": [1.0, 0.5, 1.0, -1.0]
                ]))
        }, containing: "±1")
        expectBuildError({
            try PrismBonsaiHadamardPlan(
                configData: validConfigData(),
                hadamardData: validHadamardData([
                    "prism.hadamard.sign_widths": [3],
                    "prism.hadamard.sign_values": [1.0, -1.0, 1.0, -1.0],
                ]))
        }, containing: "must sum")
    }

    @Test("plan build rejects schema, runtime and block-size drift")
    func planRejectsContractDrift() {
        expectBuildError({
            try PrismBonsaiHadamardPlan(
                configData: validConfigData(["schema_version": 1]),
                hadamardData: validHadamardData())
        }, containing: "schema_version must be 2")
        expectBuildError({
            try PrismBonsaiHadamardPlan(
                configData: validConfigData([
                    "requires_runtime": "runtime/other.py"
                ]),
                hadamardData: validHadamardData())
        }, containing: "requires_runtime must be")
        expectBuildError({
            try PrismBonsaiHadamardPlan(
                configData: validConfigData(),
                hadamardData: validHadamardData([
                    "prism.hadamard.block_size": 512
                ]))
        }, containing: "must match the modules[] block size")
    }

    // MARK: 6. Install-plan resolution and sign-key consumption

    private func resignFixturePlan() throws -> PrismBonsaiHadamardPlan {
        try PrismBonsaiHadamardPlan(
            configData: validConfigData(), hadamardData: validHadamardData())
    }

    @Test("resolve maps namespaced checkpoint keys to module leaves")
    func resolveMapsCheckpointBasesToLeaves() throws {
        let plan = try resignFixturePlan()
        let leafPaths: Set<String> = [
            "lm_head", "model.layers.0.self_attn.q_proj", "model.embed_tokens",
        ]
        let checkpointKeys: Set<String> = [
            // Published with the pack's VLM namespace.
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
        ]
        let resolved = try PrismBonsaiInstall.resolve(
            plan: plan, leafModulePaths: leafPaths, checkpointKeys: checkpointKeys)
        let byPath = Dictionary(
            uniqueKeysWithValues: resolved.updates.map {
                ($0.modulePath, $0.checkpointBase)
            })
        #expect(byPath["lm_head"] == "lm_head")
        #expect(
            byPath["model.layers.0.self_attn.q_proj"]
                == "model.layers.0.self_attn.q_proj")
        #expect(byPath["model.embed_tokens"] == "model.embed_tokens")
        // Every update consumes exactly its four packed keys (signs included).
        for update in resolved.updates {
            #expect(update.weightKeysToConsume.count == 4)
            #expect(update.weightKeysToConsume.contains(update.signsKey))
        }
        #expect(
            resolved.updates.flatMap(\.weightKeysToConsume).count == 12)
    }

    @Test("resolve accepts post-sanitize stripped key forms")
    func resolveAcceptsBothKeyForms() throws {
        let plan = try resignFixturePlan()
        let leafPaths: Set<String> = [
            "lm_head", "model.layers.0.self_attn.q_proj", "model.embed_tokens",
        ]
        // Fully stripped keys (what an LLM sanitize would produce).
        let stripped: Set<String> = [
            "lm_head.weight", "lm_head.scales", "lm_head.biases", "lm_head.signs",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.scales",
            "model.layers.0.self_attn.q_proj.biases",
            "model.layers.0.self_attn.q_proj.signs",
            "model.embed_tokens.weight", "model.embed_tokens.scales",
            "model.embed_tokens.biases", "model.embed_tokens.signs",
        ]
        let resolved = try PrismBonsaiInstall.resolve(
            plan: plan, leafModulePaths: leafPaths, checkpointKeys: stripped)
        #expect(resolved.updates.count == 3)
        let embedding = resolved.updates.first {
            $0.modulePath == "model.embed_tokens"
        }
        #expect(embedding?.signsKey == "model.embed_tokens.signs")
    }

    @Test("resolve fails when a .signs vector is missing")
    func resolveFailsOnMissingSigns() throws {
        let plan = try resignFixturePlan()
        let leafPaths: Set<String> = [
            "lm_head", "model.layers.0.self_attn.q_proj", "model.embed_tokens",
        ]
        let noSigns: Set<String> = [
            "lm_head.weight", "lm_head.scales", "lm_head.biases", "lm_head.signs",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.scales",
            "model.layers.0.self_attn.q_proj.biases",
            "model.layers.0.self_attn.q_proj.signs",
            "model.embed_tokens.weight", "model.embed_tokens.scales",
            "model.embed_tokens.biases",
        ]
        #expect(
            throws: PrismBonsaiInstall.Error.self
        ) {
            try PrismBonsaiInstall.resolve(
                plan: plan, leafModulePaths: leafPaths, checkpointKeys: noSigns)
        }
    }

    @Test("resolve fails on an affine tensor outside the manifest")
    func resolveFailsOnStrayAffineTensor() throws {
        let plan = try resignFixturePlan()
        let leafPaths: Set<String> = [
            "lm_head", "model.layers.0.self_attn.q_proj", "model.embed_tokens",
        ]
        let stray: Set<String> = [
            "lm_head.weight", "lm_head.scales", "lm_head.biases", "lm_head.signs",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.scales",
            "model.layers.0.self_attn.q_proj.biases",
            "model.layers.0.self_attn.q_proj.signs",
            "model.embed_tokens.weight", "model.embed_tokens.scales",
            "model.embed_tokens.biases", "model.embed_tokens.signs",
            // An affine companion the manifest never mentions.
            "model.layers.0.mlp.gate_proj.scales",
        ]
        #expect(
            throws: PrismBonsaiInstall.Error.self
        ) {
            try PrismBonsaiInstall.resolve(
                plan: plan, leafModulePaths: leafPaths, checkpointKeys: stray)
        }
    }

    @Test("resolve fails when a plan entry matches no model leaf")
    func resolveFailsOnUnresolvableEntry() throws {
        let plan = try resignFixturePlan()
        let leafPaths: Set<String> = [
            "lm_head", "model.layers.0.self_attn.q_proj",
        ]  // embed_tokens missing
        let keys: Set<String> = [
            "lm_head.weight", "lm_head.scales", "lm_head.biases", "lm_head.signs",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.scales",
            "model.layers.0.self_attn.q_proj.biases",
            "model.layers.0.self_attn.q_proj.signs",
        ]
        #expect(
            throws: PrismBonsaiInstall.Error.self
        ) {
            try PrismBonsaiInstall.resolve(
                plan: plan, leafModulePaths: leafPaths, checkpointKeys: keys)
        }
    }

    @Test("module path candidates cover the namespace variants")
    func modulePathCandidatesCoverVariants() {
        let namespaced = PrismBonsaiInstall.modulePathCandidates(
            forCheckpointBase: "language_model.model.embed_tokens")
        #expect(namespaced.contains("language_model.model.embed_tokens"))
        #expect(namespaced.contains("embed_tokens"))
        #expect(namespaced.contains("model.embed_tokens"))
        #expect(namespaced.count == 3)
        #expect(
            PrismBonsaiInstall.modulePathCandidates(forCheckpointBase: "lm_head")
                .contains("lm_head"))
        #expect(
            PrismBonsaiInstall.modulePathCandidates(forCheckpointBase: "lm_head")
                .contains("model.lm_head"))
        #expect(
            PrismBonsaiInstall.modulePathCandidates(forCheckpointBase: "lm_head")
                .contains("language_model.lm_head"))
        #expect(
            PrismBonsaiInstall.checkpointKeyCandidates(
                moduleLeafPath: "lm_head", suffix: ".signs")
                .contains("language_model.lm_head.signs"))
        #expect(
            PrismBonsaiInstall.checkpointKeyCandidates(
                moduleLeafPath: "model.embed_tokens", suffix: ".weight")
                .contains("language_model.model.embed_tokens.weight"))
    }

    // MARK: 7. Flag-off rollback preservation

    @Test("gate OFF still rejects prism types before any transform path")
    func gateOffRejectsPrismIdentity() {
        // Byte-identical to the accepted gate suite: the decision layer is the
        // wall — with the gate OFF the transform seam can never be reached.
        let config = validConfigData()
        let decision = PrismBonsaiPortability.decide(
            configData: config, hadamardData: validHadamardData(),
            gateEnabled: false)
        #expect(decision == .gateOffReject)
    }

    @Test("gate ON with a valid manifest still reports gateOnManifestValid")
    func gateOnValidManifestDecision() {
        let decision = PrismBonsaiPortability.decide(
            configData: validConfigData(), hadamardData: validHadamardData(),
            gateEnabled: true)
        #expect(decision == .gateOnManifestValid)
    }
}