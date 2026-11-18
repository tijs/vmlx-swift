import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import XCTest
@testable import MLXLLM

final class SenseNovaLoopTests: XCTestCase {
    private func fixture() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["SENSENOVA_FIXTURE_DIR"] else {
            throw XCTSkip("Generate the pinned authored CPU fixture with tools/sensenova_loop/reference.py")
        }
        return URL(fileURLWithPath: path)
    }

    private func configuration() throws -> SenseNovaLoopConfiguration {
        try JSONDecoder().decode(SenseNovaLoopConfiguration.self,
                                 from: Data(contentsOf: fixture().appendingPathComponent("config.json")))
    }

    private func loaded() throws -> (SenseNovaLoopModel, [String: MLXArray]) {
        let directory = try fixture()
        let model = SenseNovaLoopModel(try configuration())
        let weights = try MLX.loadArrays(url: directory.appendingPathComponent("weights.safetensors"))
        try model.update(parameters: ModuleParameters.unflattened(model.sanitize(weights: weights)), verify: .all)
        let expected = try MLX.loadArrays(url: directory.appendingPathComponent("expected.safetensors"))
        return (model, expected)
    }

    private func difference(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.max(abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
    }

    func testConfigurationRejectsUnsupportedExecution() throws {
        let data = try Data(contentsOf: fixture().appendingPathComponent("config.json"))
        let original = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        for (group, field, value) in [
            ("loop_settings", "eval_count", 1 as Any),
            ("loop_settings", "eval_count", true as Any),
            ("loop_settings", "counts", [1] as Any),
            ("loop_settings", "start", -1 as Any),
            ("loop_settings", "end", 33 as Any),
            ("loop_settings", "dynamic", true as Any),
            ("mhc", "streams", 4 as Any),
            ("mhc", "iterations", 19 as Any),
            ("mhc", "fixed_identity", true as Any),
            ("mhc", "start", 5 as Any),
            ("mhc", "pre_scale", 2 as Any),
            ("text_config", "num_attention_heads", 0 as Any),
            ("text_config", "linear_num_key_heads", 0 as Any),
            ("text_config", "mtp_num_hidden_layers", 1 as Any),
        ] {
            var changed = original
            var settings = changed[group] as! [String: Any]
            settings[field] = value
            changed[group] = settings
            XCTAssertThrowsError(try JSONDecoder().decode(SenseNovaLoopConfiguration.self,
                from: JSONSerialization.data(withJSONObject: changed)), "\(group).\(field)=\(value)")
        }
        let config = try configuration()
        XCTAssertEqual(config.executionPlan.count, 56)
        XCTAssertEqual(config.executionPlan, Array(0..<4) + Array(4..<28) + Array(4..<28) + Array(28..<32))
    }

    func testFactoryRejectsInvalidLoopWithoutTryingNestedQwen() async throws {
        struct UnusedTokenizerLoader: TokenizerLoader {
            struct UnexpectedLoad: Error {}

            func load(from directory: URL) async throws -> any Tokenizer {
                throw UnexpectedLoad()
            }
        }

        let data = try Data(contentsOf: fixture().appendingPathComponent("config.json"))
        var config = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        var loop = try XCTUnwrap(config["loop_settings"] as? [String: Any])
        loop["eval_count"] = 1
        config["loop_settings"] = loop
        var text = try XCTUnwrap(config["text_config"] as? [String: Any])
        text["model_type"] = "qwen3_5_text"
        config["text_config"] = text
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try JSONSerialization.data(withJSONObject: config)
            .write(to: directory.appendingPathComponent("config.json"))

        do {
            _ = try await LLMModelFactory.shared.load(from: directory, using: UnusedTokenizerLoader())
            XCTFail("Invalid SenseNova loop configuration loaded")
        } catch is SenseNovaLoopConfiguration.ContractError {
            // Preserve the architecture's contract error before any weight or tokenizer load.
        } catch {
            XCTFail("Expected the SenseNova contract error, received \(error)")
        }
    }

    func testRouterMatchesIndependentAuthoredFixture() throws {
        let (model, expected) = try loaded()
        let router = try XCTUnwrap(model.model.layers[4].mhcAttn)
        let (branch, post, mix) = router(expected["router_residual"]!)
        XCTAssertLessThan(difference(branch, expected["router_branch"]!), 1e-5)
        XCTAssertLessThan(difference(post, expected["router_post"]!), 1e-5)
        XCTAssertLessThan(difference(mix, expected["router_mix"]!), 1e-5)
        XCTAssertLessThan(difference(mix.sum(axis: -1), MLXArray.ones([1, 3, 2])), 1e-5)
        XCTAssertLessThan(difference(mix.sum(axis: -2), MLXArray.ones([1, 3, 2])), 1e-5)
        XCTAssertEqual(router.weight.dtype, .float32)
        XCTAssertEqual(router.bias.dtype, .float32)
    }

    func testFusedSinkhornMatchesEagerAcrossAdjacentTokenMatrices() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("The fused path requires a Metal device")
        }
        let logits = MLXArray([
            1.1 as Float, -2.7, 3.4, 4.5,
            -32.0, -28.4, -44.2, -27.2,
            37.2, 7.1, 23.9, -10.2,
        ]).reshaped(1, 3, 2, 2)
        let eager = SenseNovaMHCRouter.sinkhorn(logits, fused: false)
        let fused = SenseNovaMHCRouter.sinkhorn(logits, fused: true)
        XCTAssertEqual(fused.shape, [1, 3, 2, 2])
        XCTAssertEqual(fused.dtype, .float32)
        XCTAssertLessThan(difference(fused, eager), 2e-6)
    }

    func testFullForwardAndEveryVisitMatchAuthoredReference() throws {
        let (model, expected) = try loaded()
        let (hidden, states) = model.model.forward(expected["ids"]!, cache: nil, capture: true)
        let logits = model.lmHead(hidden)
        XCTAssertEqual(states.count, 57)
        for (i, actual) in states.enumerated() {
            let delta = difference(actual, expected["hidden_\(i)"]!)
            print("SENSENOVA_VISIT \(i) maxabs=\(delta)")
            XCTAssertLessThan(difference(actual, expected["hidden_\(i)"]!), 1e-3, "visit \(i)")
        }
        XCTAssertLessThan(difference(logits, expected["logits"]!), 1e-3)
        XCTAssertEqual(argMax(logits, axis: -1).asArray(Int32.self),
                       argMax(expected["logits"]!, axis: -1).asArray(Int32.self))
    }

    func testChunkedAndTokenDecodeHaveIndependentVisitCaches() throws {
        let (model, expected) = try loaded()
        let ids = expected["ids"]!
        let cache = model.newCache(parameters: nil)
        XCTAssertEqual(cache.count, 56)
        XCTAssertEqual(Set(cache.map { ObjectIdentifier($0 as AnyObject) }).count, 56)
        XCTAssertEqual(cache.filter { $0 is MambaCache }.count, 42)
        XCTAssertEqual(model.model.layers.count, 32)
        XCTAssertTrue(model.model.layers[model.model.plan[4]] === model.model.layers[model.model.plan[28]])
        MLX.eval(model(ids[0..., 0..<3], cache: cache))
        let suffix = model(ids[0..., 3...], cache: cache)
        XCTAssertLessThan(difference(suffix, expected["cached_logits"]!), 1e-3)
        let tokenCache = model.newCache(parameters: nil)
        var outputs = [MLXArray]()
        for i in 0..<ids.dim(1) {
            let output = model(ids[0..., i..<(i + 1)], cache: tokenCache)
            MLX.eval(output)
            outputs.append(output)
        }
        XCTAssertLessThan(difference(concatenated(outputs, axis: 1), expected["logits"]!), 1e-3)
        // Mutate one visit's cache object, not a copied array of references.
        let first = cache[4] as! MambaCache
        let repeated = cache[28] as! MambaCache
        let before = repeated[1]!
        MLX.eval(before)
        first[1] = MLXArray.zeros(before.shape, dtype: before.dtype)
        XCTAssertLessThan(difference(repeated[1]!, before), 1e-8)
    }

    func testSanitizerKeepsRouterFP32AndDoesNotHideExtraWeights() throws {
        let config = try configuration()
        let model = SenseNovaLoopModel(config)
        let weights: [String: MLXArray] = [
            "model.language_model.layers.4.mhc_attn.weight": MLXArray.ones([8, 128], dtype: .bfloat16),
            "model.language_model.layers.0.input_layernorm.weight": MLXArray.zeros([64]),
            "unexpected.weight": MLXArray.ones([1]),
            "model.language_model.loop_gate.logits": MLXArray.ones([64]),
        ]
        let out = model.sanitize(weights: weights)
        XCTAssertEqual(out["model.layers.4.mhc_attn.weight"]!.dtype, .float32)
        XCTAssertEqual(out["model.layers.0.input_layernorm.weight"]!.mean().item(Float.self), 1)
        XCTAssertNotNil(out["unexpected.weight"])
        XCTAssertNil(out["model.loop_gate.logits"])
        XCTAssertFalse(model.supportsWholeForwardCompilation)
        XCTAssertEqual(model.maximumSupportedDecodeBatchSize, 1)
        XCTAssertNotNil(model.cacheComputationIdentity)
    }

    func testSenseNovaNormalizationOverrideDoesNotChangeLegacyQwen() throws {
        let config = try configuration()
        let legacy = Qwen35GatedDeltaNet(config.text)
        let model = SenseNovaLoopModel(config)
        XCTAssertNil(legacy.l2NormSumEpsilon)
        XCTAssertFalse(legacy.usesFP32GDNReference)
        XCTAssertEqual(model.model.layers[0].linearAttn!.l2NormSumEpsilon, 1e-6)
        XCTAssertTrue(model.model.layers[0].linearAttn!.usesFP32GDNReference)
        XCTAssertFalse(Qwen35Attention(config.text).preservesNormInputDType)
        XCTAssertTrue(model.model.layers[3].selfAttn!.preservesNormInputDType)
        let bf16 = MLXArray([Float](repeating: 0.3, count: config.text.hiddenSize))
            .reshaped(1, 1, config.text.hiddenSize).asType(.bfloat16)
        XCTAssertEqual(model.model.layers[0].inputNorm.weight.dtype, .float32)
        XCTAssertEqual(model.model.layers[0].inputNorm(bf16).dtype, .bfloat16)
        let small = MLXArray([Float](repeating: 0.0001, count: 16)).reshaped(1, 1, 1, 16)
        let oracle = small * rsqrt((small * small).sum(axis: -1, keepDims: true) + 1e-6)
        let weight = MLXArray.full([16], values: MLXArray(Float(0.25)))
        let sumEquivalent = MLXFast.rmsNorm(small, weight: weight, eps: 1e-6 / 16)
        XCTAssertLessThan(difference(sumEquivalent, oracle), 1e-6)
        let meanLegacy = MLXFast.rmsNorm(small, weight: weight, eps: 1e-6)
        XCTAssertGreaterThan(difference(meanLegacy, oracle), 0.05)
    }

    func testCacheNamespaceSeparatesWeightQuantizations() throws {
        let original = try Data(contentsOf: fixture().appendingPathComponent("config.json"))
        var json = try JSONSerialization.jsonObject(with: original) as! [String: Any]
        let plain = SenseNovaLoopModel(try JSONDecoder().decode(SenseNovaLoopConfiguration.self, from: original))
        json["quantization"] = ["bits": 8, "group_size": 64]
        let q8 = SenseNovaLoopModel(try JSONDecoder().decode(SenseNovaLoopConfiguration.self,
            from: JSONSerialization.data(withJSONObject: json)))
        XCTAssertNotEqual(plain.cacheComputationIdentity, q8.cacheComputationIdentity)
        json["quantization"] = ["bits": 4, "group_size": 64]
        let q4 = SenseNovaLoopModel(try JSONDecoder().decode(SenseNovaLoopConfiguration.self,
            from: JSONSerialization.data(withJSONObject: json)))
        XCTAssertNotEqual(q8.cacheComputationIdentity, q4.cacheComputationIdentity)
        json["quantization"] = ["bits": 8, "group_size": 64]
        json["activation_dtype"] = "float32"
        let fp32 = SenseNovaLoopModel(try JSONDecoder().decode(SenseNovaLoopConfiguration.self,
            from: JSONSerialization.data(withJSONObject: json)))
        XCTAssertNotEqual(q8.cacheComputationIdentity, fp32.cacheComputationIdentity)
        let sanitized = fp32.sanitize(weights: [
            "model.embed_tokens.scales": MLXArray.ones([2, 1], dtype: .bfloat16),
            "model.embed_tokens.weight": MLXArray.zeros([2, 16], dtype: .uint32),
        ])
        XCTAssertEqual(sanitized["model.embed_tokens.scales"]!.dtype, .float32)
        XCTAssertEqual(sanitized["model.embed_tokens.weight"]!.dtype, .uint32)
    }

    func testBF16FixtureMatchesAuthoredLogits() throws {
        guard let path = ProcessInfo.processInfo.environment["SENSENOVA_BF16_FIXTURE_DIR"] else {
            throw XCTSkip("Generate the additional authored BF16 tiny fixture")
        }
        let directory = URL(fileURLWithPath: path)
        let config = try JSONDecoder().decode(SenseNovaLoopConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let model = SenseNovaLoopModel(config)
        let weights = try MLX.loadArrays(url: directory.appendingPathComponent("weights.safetensors"))
        try model.update(parameters: ModuleParameters.unflattened(model.sanitize(weights: weights)), verify: .all)
        let expected = try MLX.loadArrays(url: directory.appendingPathComponent("expected.safetensors"))
        let logits = model(expected["ids"]!, cache: nil)
        XCTAssertEqual(logits.dtype, .bfloat16)
        let delta = difference(logits, expected["logits"]!)
        print("SENSENOVA_BF16 maxabs=\(delta)")
        XCTAssertLessThan(delta, 0.03)
        XCTAssertEqual(argMax(logits, axis: -1).asArray(Int32.self),
                       argMax(expected["logits"]!, axis: -1).asArray(Int32.self))
    }
}
