// Actual production prepare/store regression. No snapshot injection or altered chunk geometry.
// Source prepared; numerical proof pending.
import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest
@testable import MLXLMCommon
@testable import MLXVLM

final class FlashCanonicalStableProductionTests: XCTestCase {
    private static func withFixture(routedBits: [Int] = [], routedGroupSize: Int = 64, mtpEnabled: Bool = false,
                             inputProjectionBits: Int? = nil, gdnHeadDimension: Int = 16,
                             pleLayerIDs: [Int] = [1], additionalLinearLayer: Bool = false,
                             distinctHCNorms: Bool = false,
                             _ body: (Qwen4Exp) throws -> Void) throws {
        // Entire geometry is bounded before construction; no installed model is read.
        let layerCount = additionalLinearLayer ? 3 : 2
        let data = Data(
            """
            {
              "model_type":"qwen4_exp","eos_token_id":1,
              "text_config":{
                "model_type":"qwen4_exp_text","dtype":"float32",
                "hidden_size":64,"num_hidden_layers":2,"intermediate_size":64,
                "num_attention_heads":4,"num_key_value_heads":1,"head_dim":16,
                "linear_num_value_heads":4,"linear_num_key_heads":1,
                "linear_key_head_dim":16,"linear_value_head_dim":16,
                "linear_conv_kernel_dim":4,"vocab_size":128,
                "num_experts":8,"num_experts_per_tok":2,
                "moe_intermediate_size":16,"shared_expert_intermediate_size":16,
                "layer_types":["linear_attention","full_attention"],
                "hc_count":4,"hc_lowrank":8,"ple_layer_ids":[1],
                "ple_embed_dim":64,"ple_conv_kernel_size":4,
                "ngram_size":3,"heads_per_ngram":2,"ngram_vocab_size_base":101,
                "make_ngram_vocab_size_divisible_by":128,"seed":1234,
                "split_ngram_parts":4,"indexer_n_heads":2,"indexer_kv_heads":1,
                "indexer_head_dim":8,"indexer_budget":32,"indexer_compress_ratio":4,
                "mrope_section":[1,1,1],"mtp_num_hidden_layers":0
              }
            }
            """.utf8)
        var fixtureData = data
        if gdnHeadDimension != 16 || pleLayerIDs != [1] || additionalLinearLayer {
            try canonicalRequire([16, 128].contains(gdnHeadDimension))
            try canonicalRequire(!pleLayerIDs.isEmpty && pleLayerIDs.allSatisfy { (1...2).contains($0) })
            try canonicalRequire(!pleLayerIDs.contains(2) || additionalLinearLayer,
                         "PLE companion state requires a linear-attention layer")
            var root = try canonicalRequire(JSONSerialization.jsonObject(with: fixtureData) as? [String: Any])
            var text = try canonicalRequire(root["text_config"] as? [String: Any])
            text["linear_key_head_dim"] = gdnHeadDimension
            text["ple_layer_ids"] = pleLayerIDs
            text["num_hidden_layers"] = layerCount
            text["layer_types"] = Array(repeating: "linear_attention", count: layerCount - 1)
                + ["full_attention"]
            root["text_config"] = text
            fixtureData = try JSONSerialization.data(withJSONObject: root)
        }
        if mtpEnabled {
            var root = try canonicalRequire(JSONSerialization.jsonObject(with: fixtureData) as? [String: Any])
            var text = try canonicalRequire(root["text_config"] as? [String: Any])
            text["mtp_num_hidden_layers"] = 1
            root["text_config"] = text
            fixtureData = try JSONSerialization.data(withJSONObject: root)
        }
        var routedSpecs: [String: Int] = [:]
        if !routedBits.isEmpty {
            try canonicalRequire([32, 64].contains(routedGroupSize))
            try canonicalRequire(routedBits.count == 3 && (routedBits == [0, 0, 0] || routedBits.allSatisfy { [2, 3, 4, 6].contains($0) }))
            var root = try canonicalRequire(JSONSerialization.jsonObject(with: fixtureData) as? [String: Any])
            var text = try canonicalRequire(root["text_config"] as? [String: Any])
            text["dtype"] = "bfloat16"
            text["moe_intermediate_size"] = 64
            text["shared_expert_intermediate_size"] = 64
            root["text_config"] = text
            var quantization: [String: Any] = ["bits": 8, "group_size": routedGroupSize]
            for layer in 0..<layerCount {
                for (projection, bits) in zip(["gate_proj", "up_proj", "down_proj"], routedBits) {
                    if bits == 0 { continue } // BF16 dense control, same geometry.
                    let path = "language_model.layers.\(layer).mlp.switch_mlp.\(projection)"
                    routedSpecs[path] = bits
                    quantization[path] = ["bits": bits, "group_size": routedGroupSize]
                }
            }
            if let bits = inputProjectionBits {
                try canonicalRequire([2, 3, 4, 6, 8].contains(bits))
                for name in ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"] {
                    let path = "language_model.layers.0.linear_attn.\(name)"
                    routedSpecs[path] = bits
                    quantization[path] = ["bits": bits, "group_size": routedGroupSize]
                }
            }
            if !routedSpecs.isEmpty { root["quantization"] = quantization }
            fixtureData = try JSONSerialization.data(withJSONObject: root)
        }
        let config = try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: fixtureData)
        let text = config.base.textConfiguration
        try canonicalRequire(text.hiddenSize == 64 && text.hiddenLayers == layerCount)
        try canonicalRequire(text.vocabularySize == 128 && text.numExperts == 8)
        MLXRandom.seed(0)
        let model = Qwen4Exp(config)
        let count = model.parameters().flattened().reduce(0) { $0 + $1.1.size }
        try canonicalRequire(count < 2_000_000, "refuse fixture drift before MLX evaluation")
        if !routedBits.isEmpty {
            // Match the retained live parameter-domain contract: ordinary
            // parameters BF16, router weights FP32. This is still a tiny
            // generated fixture, not a shipped-weight numerical reference.
            model.update(parameters: ModuleParameters.unflattened(
                model.parameters().flattened().map { path, value in
                    (path, path.hasSuffix(".mlp.gate.weight") ? value : value.asType(.bfloat16))
                }))
        }
        if !routedSpecs.isEmpty {
            quantize(model: model, filter: { path, _ in
                guard let bits = routedSpecs[path] else { return nil }
                return (groupSize: routedGroupSize, bits: bits, mode: QuantizationMode.affine)
            })
            let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            for (path, bits) in routedSpecs {
                let projection = try canonicalRequire(leaves[path] as? any Quantized)
                try canonicalRequire(projection.bits == bits && projection.groupSize == routedGroupSize)
            }
            model.update(parameters: ModuleParameters.unflattened(
                model.parameters().flattened().compactMap { path, value in
                    guard path.hasSuffix(".scales") || path.hasSuffix(".biases") else { return nil }
                    return (path, value.asType(.float16))
                }))
        }

        if distinctHCNorms {
            let norms = model.parameters().flattened()
                .filter { $0.0.hasSuffix(".hc_norm.weight") }.sorted { $0.0 < $1.0 }
            try canonicalRequire(norms.count == layerCount * 2 + 1)
            model.update(parameters: ModuleParameters.unflattened(norms.enumerated().map { ordinal, leaf in
                let (path, value) = leaf
                let coordinates = MLXArray(0..<value.size).asType(.float32)
                let varied = 1 + 0.25 * sin(coordinates * 0.013 + Float(ordinal + 1))
                return (path, varied.asType(value.dtype).reshaped(value.shape))
            }))
        }

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("flash-prefill-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var arrays: [String: MLXArray] = [:]
        var map: [String: String] = [:]
        var specs: [String: [String: Int]] = [:]
        for (pleOrdinal, layerID) in pleLayerIDs.enumerated() {
          let layout = Qwen4ExpNGramHash.headVocabLayout(
              ngramHeads: 4, pleLayerIndex: pleOrdinal, ngramVocabSizeBase: 101)
          let paddedRows = ((layout.sizes.reduce(0, +) + 127) / 128) * 128
          let shardRows = paddedRows / 4
          try canonicalRequire(shardRows > 0 && shardRows <= 256)
          for shard in 0 ..< 4 {
            let base = "language_model.layers.\(layerID - 1).ple.ngram_embedding.shards.\(shard)"
            arrays[base + ".weight"] = MLXArray(
                (0 ..< (shardRows * 2)).map {
                    UInt32(truncatingIfNeeded: ($0 + shard + pleOrdinal) * 0x12345)
                }
            ).reshaped(shardRows, 2)
            arrays[base + ".scales"] = MLXArray.full([shardRows, 4], values: MLXArray(Float(0.01)))
                .asType(.float16)
            arrays[base + ".biases"] = MLXArray.full([shardRows, 4], values: MLXArray(Float(-0.05)))
                .asType(.float16)
            specs[base + ".weight"] = ["bits": 4, "group_size": 4]
            for suffix in [".weight", ".scales", ".biases"] {
                map[base + suffix] = "model.safetensors"
            }
          }
        }
        try MLX.save(arrays: arrays, url: directory.appendingPathComponent("model.safetensors"))
        try JSONSerialization.data(withJSONObject: ["weight_map": map])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        try JSONSerialization.data(withJSONObject: ["jang_config": ["bit_map": specs]])
            .write(to: directory.appendingPathComponent("config.json"))
        try model.configure(modelDirectory: directory)
        try body(model)
    }

    private struct SubmissionTensor: Equatable {
        let shape: [Int]
        let dtype: String
        let bits: [UInt32]

        init(_ value: MLXArray) {
            shape = value.shape
            dtype = String(describing: value.dtype)
            let values = value.asType(.float32).asArray(Float.self)
            XCTAssertTrue(values.allSatisfy { $0.isFinite })
            bits = values.map(\.bitPattern)
        }
    }

    private struct SubmissionCache: Equatable {
        let offset: Int
        let metadata: [String]
        let state: [SubmissionTensor]
        let pooled: SubmissionTensor?
        let pooledCount: Int

        init(_ cache: any KVCache) {
            offset = cache.offset
            metadata = cache.metaState
            state = cache.state.map(SubmissionTensor.init)
            let qsa = cache as? QSAKVCache
            pooled = qsa?.derivedPooledBlocks.map(SubmissionTensor.init)
            pooledCount = qsa?.derivedPooledBlockCount ?? 0
        }
    }

    private final class CaptureModel: Module, LanguageModel, @unchecked Sendable {
        let real: Qwen4Exp
        var preparedLengths: [Int] = []
        init(_ real: Qwen4Exp) { self.real = real; super.init() }
        func newCache(parameters: GenerateParameters?) -> [KVCache] { real.newCache(parameters: parameters) }
        func callAsFunction(_ input: MLXArray, cache: [KVCache]?) -> MLXArray { real(input, cache: cache) }
        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            preparedLengths.append(input.text.tokens.size)
            return try real.prepare(input, cache: cache, windowSize: windowSize)
        }
    }

    func testSingleStableBoundaryUsesProductionResidual() throws { try exercise(multiple: false) }
    func testMultipleStableBoundariesUseProductionResidual() throws { try exercise(multiple: true) }
    func testSmallStableBoundaryDoesNotPreventLaterCanonicalCapture() throws {
        try exercise(multiple: true, smallAndLarge: true)
    }
    func testRequiredFreshSelectionSkipsExistingDiskButCapturesOwnStableSeed() throws {
        try exercise(multiple: false, freshRequired: true)
    }
    private func exercise(multiple: Bool, smallAndLarge: Bool = false, freshRequired: Bool = false) throws {
        try MLXMetalTestLock.withLock {
            try Self.withFixture(routedBits: [2, 3, 4], inputProjectionBits: 2) { real in
                let prompt = (0..<60).map { 2 + ($0 * 17) % 113 }
                let stable = smallAndLarge ? [9, 37] : (multiple ? [33, 37] : [37])
                let parameters = GenerateParameters(maxTokens: 1, temperature: 0, prefillStepSize: 16)
                let input = LMInput(tokens: MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count),
                    mask: MLXArray.ones([1, prompt.count], dtype: .int8), tokenIds: prompt, cachePrefixTokenCounts: stable + [52], cacheStablePrefixTokenCounts: stable,
                    cacheRestorePolicy: freshRequired ? .freshRequiredToolSelection : .standard)
                let directory = FileManager.default.temporaryDirectory.appendingPathComponent("canonical-production-\(UUID().uuidString)")
                defer { try? FileManager.default.removeItem(at: directory) }
                let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
                    usePagedCache: false, enableDiskCache: true, diskCacheMaxGB: 0.1,
                    diskCacheDir: directory, modelKey: "tiny-canonical-production"))
                coordinator.setHybrid(true, requiresRecurrentSSMCompanion: true, requiresSeparateRecurrentPayload: false)
                coordinator.setGenPromptSuffixTokens(Array(prompt.suffix(8)))
                if freshRequired {
                    let prior = real.newCache(parameters: parameters)
                    _ = try feed(Array(prompt.prefix(52)), model: real, cache: prior)
                    coordinator.storeAfterGeneration(promptTokens: Array(prompt.prefix(52)), perLayerData: [],
                        ssmStates: extractSSMStates(from: prior), cache: prior,
                        mediaSalt: computeCacheSalt(for: input, parameters: parameters))
                    // Demonstrate that a usable disk restore exists, rather than
                    // mistaking an empty cache for successful restore exclusion.
                    guard case .hit(let matched, _, let detail, _, _, _) = coordinator.fetch(
                        tokens: prompt, mediaSalt: computeCacheSalt(for: input, parameters: parameters)) else {
                        XCTFail("Required-tool control needs an actual usable prior disk prefix"); return
                    }
                    XCTAssertEqual(matched, 52); XCTAssertEqual(detail, .disk)
                    XCTAssertFalse(coordinator.hasDurableDiskEntry(tokens: Array(prompt.prefix(36)),
                        mediaSalt: computeCacheSalt(for: input, parameters: parameters)))
                }
                let hitsBefore = coordinator.snapshotStats().diskStats?.hits
                let model = CaptureModel(real)
                let progress = ProgressRows()
                var iterator = try TokenIterator(input: input, model: model, parameters: parameters,
                    cacheCoordinator: coordinator, prefillProgressHandler: { progress.add($0) })
                XCTAssertEqual(model.preparedLengths, [52, 8])
                if freshRequired {
                    XCTAssertEqual(coordinator.snapshotStats().diskStats?.hits, hitsBefore,
                        "Required generation must not fetch the proven available disk prefix")
                }
                let baselineProgress = ProgressRows()
                let baselineCoordinator = CacheCoordinator(config: CacheCoordinatorConfig(
                    usePagedCache: false, enableDiskCache: true, diskCacheMaxGB: 0.1,
                    diskCacheDir: directory.appendingPathComponent("baseline"), modelKey: "tiny-canonical-production"))
                baselineCoordinator.setHybrid(true, requiresRecurrentSSMCompanion: true, requiresSeparateRecurrentPayload: false)
                baselineCoordinator.setGenPromptSuffixTokens(Array(prompt.suffix(8)))
                let baselineInput = LMInput(tokens: input.text.tokens, mask: input.text.mask, tokenIds: prompt,
                    cachePrefixTokenCounts: stable + [52], cacheStablePrefixTokenCounts: [],
                    cacheRestorePolicy: freshRequired ? .freshRequiredToolSelection : .standard)
                let baseline = try TokenIterator(input: baselineInput, model: real, parameters: parameters,
                    cacheCoordinator: baselineCoordinator, prefillProgressHandler: { baselineProgress.add($0) })
                XCTAssertEqual(progress.values, baselineProgress.values,
                    "Complete progress sequence must match same geometry with no eligible stable capture")
                XCTAssertTrue(iterator.cache.map(SubmissionCache.init) == baseline.cache.map(SubmissionCache.init))
                let live = iterator.cache.map(SubmissionCache.init)
                iterator.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
                XCTAssertEqual(Array(model.preparedLengths.dropFirst(2)), smallAndLarge ? [8, 4] : [4],
                    "Production storage must replay residual4, not full32/36; no injected snapshots")
                XCTAssertTrue(iterator.cache.map(SubmissionCache.init) == live)
                if freshRequired {
                    XCTAssertFalse(coordinator.hasDurableDiskEntry(tokens: Array(prompt.prefix(59)),
                        mediaSalt: computeCacheSalt(for: input, parameters: parameters)),
                        "Whole-prompt N-1 tool seed remains forbidden; only declared stable36 is published")
                }
                XCTAssertTrue(progress.completed.contains(16) && progress.completed.contains(32)
                    && progress.completed.contains(48), "Original progress must remain visible")
                for boundary in stable {
                    let seed = boundary - 1
                    guard case .hit(let count, _, let detail, _, _, let arrays) = coordinator.fetch(
                        tokens: Array(prompt.prefix(boundary)), mediaSalt: computeCacheSalt(for: input, parameters: parameters),
                        skipExactDiskBoundary: true, preferredDiskBoundaries: [boundary]) else {
                        XCTFail("Missing production stable boundary"); continue
                    }
                    XCTAssertEqual(count, seed); XCTAssertEqual(detail, .disk)
                    var restored = real.newCache(parameters: parameters)
                    XCTAssertEqual(restoreFromDiskArrays(try XCTUnwrap(arrays), into: &restored), seed)
                    let cold = real.newCache(parameters: parameters)
                    _ = try feed(Array(prompt.prefix(seed)), model: real, cache: cold)
                    let canonical = makePromptBoundaryCacheSnapshot(from: cold)
                    XCTAssertTrue(restored.map(SubmissionCache.init) == canonical.map(SubmissionCache.init))
                    let next = MLXArray([Int32(prompt[seed])]).reshaped(1, 1)
                    let got = real(next, cache: restored), wanted = real(next, cache: canonical)
                    MLX.eval(got, wanted, restored, canonical)
                    XCTAssertTrue(SubmissionTensor(got) == SubmissionTensor(wanted))
                    XCTAssertTrue(restored.map(SubmissionCache.init) == canonical.map(SubmissionCache.init))
                }
                print("CANONICAL_PRODUCTION multiple=\(multiple) prepares=\(model.preparedLengths) generatedTokens=0")
            }
        }
    }

    private final class ProgressRows: @unchecked Sendable {
        let lock = NSLock()
        private var rows: [PrefillProgress] = []
        func add(_ value: PrefillProgress) {
            lock.lock(); defer { lock.unlock() }; rows.append(value)
        }
        var values: [PrefillProgress] { lock.lock(); defer { lock.unlock() }; return rows }
        var completed: [Int] { lock.lock(); defer { lock.unlock() }; return rows.map(\.completedUnitCount) }
    }

    func testCollectorRejectsWrongOriginOwnersAndKeys() throws {
        try MLXMetalTestLock.withLock {
            try Self.withFixture(inputProjectionBits: 2) { real in
                let ids = Array(2..<62)
                let input = LMInput(tokens: MLXArray(ids.map(Int32.init)).reshaped(1, 60), tokenIds: ids)
                let cache = real.newCache(parameters: nil)
                let image = LMInput(text: input.text, image: .init(pixels: MLXArray.zeros([1, 1, 1, 3])))
                XCTAssertNil(CanonicalStablePrefillCapture(input: image, promptTokens: ids,
                    cache: cache, chunkSize: 16, targets: [36], salt: "request-a"))
                let video = LMInput(text: input.text,
                    video: .init(pixels: MLXArray.zeros([1, 1, 1, 3]), embeddingTokenCount: 1))
                XCTAssertTrue(video.requiresPostPrepareCacheKey)
                XCTAssertNil(CanonicalStablePrefillCapture(input: video, promptTokens: ids,
                    cache: cache, chunkSize: 16, targets: [36], salt: "request-a"))
                let auxiliary = LMInput(tokens: input.text.tokens, tokenIds: ids, cachePromptIntent: .auxiliary)
                XCTAssertNil(CanonicalStablePrefillCapture(input: auxiliary, promptTokens: ids,
                    cache: cache, chunkSize: 16, targets: [36], salt: "request-a"))
                XCTAssertNil(CanonicalStablePrefillCapture(input: input, promptTokens: ids,
                    cache: cache, chunkSize: 16, targets: [8], salt: "request-a"))
                let capture = try XCTUnwrap(CanonicalStablePrefillCapture(input: input, promptTokens: ids,
                    cache: cache, chunkSize: 16, targets: [36], salt: "request-a"))
                let wrong = real.newCache(parameters: nil)
                _ = try feed(Array(ids.prefix(32)), model: real, cache: wrong)
                capture.receive(input: input, cache: wrong, chunkSize: 16, completed: 32, beganWithEmptyCache: true)
                XCTAssertNil(capture.snapshot)
                _ = try feed(Array(ids.prefix(32)), model: real, cache: cache)
                XCTAssertNil(CanonicalStablePrefillCapture(input: input, promptTokens: ids,
                    cache: cache, chunkSize: 16, targets: [36], salt: "request-a"))
                capture.receive(input: input, cache: cache, chunkSize: 16, completed: 32, beganWithEmptyCache: false)
                XCTAssertNil(capture.snapshot)
                capture.receive(input: input, cache: cache, chunkSize: 8, completed: 32, beganWithEmptyCache: true)
                XCTAssertNil(capture.snapshot)
                let changed = LMInput(tokens: input.text.tokens, tokenIds: [99] + Array(ids.dropFirst()))
                capture.receive(input: changed, cache: cache, chunkSize: 16, completed: 32, beganWithEmptyCache: true)
                XCTAssertNil(capture.snapshot)
                capture.receive(input: input, cache: cache, chunkSize: 16, completed: 32, beganWithEmptyCache: true)
                XCTAssertNotNil(capture.snapshot)
                XCTAssertNil(capture.copySeed(for: Array(ids.prefix(36)), salt: "request-b", chunkSize: 16))
                XCTAssertNil(capture.copySeed(for: Array(ids.prefix(36)), salt: "request-a", chunkSize: 8))
                XCTAssertNil(capture.copySeed(for: [99] + Array(ids[1..<36]), salt: "request-a", chunkSize: 16))
                XCTAssertNotNil(capture.copySeed(for: Array(ids.prefix(36)), salt: "request-a", chunkSize: 16))
            }
        }
    }

    func testCancelledProductionStoreDoesNotPublishStableSeed() async throws {
        try await Task.detached { try Self.cancelledStore() }.value
    }

    private static func cancelledStore() throws {
        try MLXMetalTestLock.withLock {
            try withFixture(routedBits: [2, 3, 4], inputProjectionBits: 2) { real in
                let ids = Array(2..<62)
                let directory = FileManager.default.temporaryDirectory.appendingPathComponent("canonical-cancel-\(UUID().uuidString)")
                defer { try? FileManager.default.removeItem(at: directory) }
                let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
                    usePagedCache: false, enableDiskCache: true, diskCacheMaxGB: 0.1,
                    diskCacheDir: directory, modelKey: "tiny-canonical-cancel"))
                coordinator.setHybrid(true, requiresRecurrentSSMCompanion: true, requiresSeparateRecurrentPayload: false)
                coordinator.setGenPromptSuffixTokens(Array(ids.suffix(8)))
                let parameters = GenerateParameters(maxTokens: 1, temperature: 0, prefillStepSize: 16)
                let input = LMInput(tokens: MLXArray(ids.map(Int32.init)).reshaped(1, 60), tokenIds: ids,
                    cachePrefixTokenCounts: [37, 52], cacheStablePrefixTokenCounts: [37])
                let model = CaptureModel(real)
                var iterator = try TokenIterator(input: input, model: model, parameters: parameters, cacheCoordinator: coordinator)
                let before = iterator.cache.map(SubmissionCache.init)
                withUnsafeCurrentTask { $0?.cancel() }
                XCTAssertTrue(Task.isCancelled)
                iterator.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
                XCTAssertFalse(coordinator.hasDurableDiskEntry(tokens: Array(ids.prefix(36)),
                    mediaSalt: computeCacheSalt(for: input, parameters: parameters)))
                XCTAssertEqual(model.preparedLengths, [52, 8])
                XCTAssertTrue(iterator.cache.map(SubmissionCache.init) == before)
            }
        }
    }

    @discardableResult
    private func feed(_ ids: [Int], model: any LanguageModel, cache: [KVCache]) throws -> MLXArray {
        let prepared = try model.prepare(LMInput(tokens: MLXArray(ids.map(Int32.init)).reshaped(1, ids.count)),
            cache: cache, windowSize: 16)
        let logits: MLXArray
        switch prepared {
        case .logits(let out): logits = out.logits
        case .tokens(let tail): logits = model(tail.tokens.reshaped(1, -1), cache: cache)
        }
        MLX.eval(logits, cache)
        return logits
    }

}

private enum CanonicalFixtureError: Error { case invalidInvariant }
private func canonicalRequire(_ condition: Bool, _ message: String = "fixture invariant",
                              file: StaticString = #filePath, line: UInt = #line) throws {
    guard condition else {
        XCTFail(message, file: file, line: line)
        throw CanonicalFixtureError.invalidInvariant
    }
}
private func canonicalRequire<T>(_ value: T?, _ message: String = "required fixture value",
                                 file: StaticString = #filePath, line: UInt = #line) throws -> T {
    try XCTUnwrap(value, message, file: file, line: line)
}
