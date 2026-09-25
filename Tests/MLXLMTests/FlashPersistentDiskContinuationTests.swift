import Foundation
import MLX
import MLXNN
import MLXRandom
@testable import MLXLMCommon
@testable import MLXVLM
import Testing

// Bounded generated fixture: no installed model or experimental sampled verifier.
@Suite(.serialized)
struct FlashPersistentDiskContinuationTests {
    private final class ProgressRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [Int] = []

        func append(_ value: Int) {
            lock.lock()
            values.append(value)
            lock.unlock()
        }

        func snapshot() -> [Int] {
            lock.lock()
            defer { lock.unlock() }
            return values
        }
    }

    private func withFixture(routedBits: [Int] = [], routedGroupSize: Int = 64, mtpEnabled: Bool = false,
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
            try #require([16, 128].contains(gdnHeadDimension))
            try #require(!pleLayerIDs.isEmpty && pleLayerIDs.allSatisfy { (1...2).contains($0) })
            try #require(!pleLayerIDs.contains(2) || additionalLinearLayer,
                         "PLE companion state requires a linear-attention layer")
            var root = try #require(JSONSerialization.jsonObject(with: fixtureData) as? [String: Any])
            var text = try #require(root["text_config"] as? [String: Any])
            text["linear_key_head_dim"] = gdnHeadDimension
            text["ple_layer_ids"] = pleLayerIDs
            text["num_hidden_layers"] = layerCount
            text["layer_types"] = Array(repeating: "linear_attention", count: layerCount - 1)
                + ["full_attention"]
            root["text_config"] = text
            fixtureData = try JSONSerialization.data(withJSONObject: root)
        }
        if mtpEnabled {
            var root = try #require(JSONSerialization.jsonObject(with: fixtureData) as? [String: Any])
            var text = try #require(root["text_config"] as? [String: Any])
            text["mtp_num_hidden_layers"] = 1
            root["text_config"] = text
            fixtureData = try JSONSerialization.data(withJSONObject: root)
        }
        var routedSpecs: [String: Int] = [:]
        if !routedBits.isEmpty {
            try #require([32, 64].contains(routedGroupSize))
            try #require(routedBits.count == 3 && (routedBits == [0, 0, 0] || routedBits.allSatisfy { [2, 3, 4, 6].contains($0) }))
            var root = try #require(JSONSerialization.jsonObject(with: fixtureData) as? [String: Any])
            var text = try #require(root["text_config"] as? [String: Any])
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
                try #require([2, 3, 4, 6, 8].contains(bits))
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
        try #require(text.hiddenSize == 64 && text.hiddenLayers == layerCount)
        try #require(text.vocabularySize == 128 && text.numExperts == 8)
        MLXRandom.seed(0)
        let model = Qwen4Exp(config)
        let count = model.parameters().flattened().reduce(0) { $0 + $1.1.size }
        try #require(count < 2_000_000, "refuse fixture drift before MLX evaluation")
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
                let projection = try #require(leaves[path] as? any Quantized)
                try #require(projection.bits == bits && projection.groupSize == routedGroupSize)
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
            try #require(norms.count == layerCount * 2 + 1)
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
          try #require(shardRows > 0 && shardRows <= 256)
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

    @Test("default early submission retains opt-out and excludes batch, prefill, trace and speculative paths")
    func earlySubmissionEligibility() {
        #expect(Qwen4ExpEarlySubmission.parse(nil))
        for value: String? in ["", "0", "4", "64", "true", "garbage"] {
            #expect(!Qwen4ExpEarlySubmission.parse(value))
        }
        #expect(Qwen4ExpEarlySubmission.parse("1"))
        func permits(_ enabled: Bool = true, _ shape: [Int] = [1, 1],
                     ar: Bool = true, recording: Bool = false,
                     externalPLE: Bool = false, trace: Bool = false) -> Bool {
            Qwen4ExpEarlySubmission.allows(
                enabled: enabled, shape: shape, autoregressive: ar,
                recordingPrefix: recording, externalPLE: externalPLE, compiledTrace: trace)
        }
        #expect(permits())
        #expect(!permits(false))
        for shape in [[1], [2, 1], [1, 0], [1, 2], [1, 64], [1, 8192]] {
            #expect(!permits(true, shape))
        }
        #expect(!permits(ar: false))
        #expect(!permits(recording: true))
        #expect(!permits(externalPLE: true))
        #expect(!permits(trace: true))
    }

    private struct SubmissionTensor: Equatable {
        let shape: [Int]
        let dtype: String
        let bits: [UInt32]

        init(_ value: MLXArray) {
            shape = value.shape
            dtype = String(describing: value.dtype)
            let values = value.asType(.float32).asArray(Float.self)
            #expect(values.allSatisfy { $0.isFinite })
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

    @Test("early submission preserves exact mixed-quant logits and GDN/QSA/PLE continuation state")
    func earlySubmissionConnectedState() throws {
        try MLXMetalTestLock.withLock {
            // Existing generated fixture and real selective SSD table reader;
            // these are numerical/state checks, not installed-quant speed rows.
            for (bits, group) in [([], 64), ([2, 3, 4], 32), ([4, 4, 4], 64), ([6, 4, 6], 64)] {
                try withFixture(routedBits: bits, routedGroupSize: group) { model in
                    try Stream.withNewDefaultStream(device: .gpu) {
                        var baselineLogits: [SubmissionTensor] = []
                        var baselineStates: [[SubmissionCache]] = []
                        for enabled in [false, true] {
                            print("EARLY-SUBMIT-STATE bits=\(bits) group=\(group) enabled=\(enabled)")
                            model.earlySubmissionEnabled = enabled
                            let cache = model.newCache(parameters: nil)
                            var observedLogits: [SubmissionTensor] = []
                            var observedStates: [[SubmissionCache]] = []
                            // Cross the fixture's actual QSA budget32 during
                            // connected history, then decode with the sparse lane.
                            for ids in [Array(2..<31), [31], [37, 41, 43], [47]] {
                                let before = model.earlySubmissionCount
                                let logits = model(MLXArray(ids.map(Int32.init)).reshaped(1, ids.count), cache: cache)
                                MLX.eval(logits, cache)
                                #expect(model.earlySubmissionCount - before ==
                                    (enabled && ids.count == 1 ? model.config.base.textConfiguration.hiddenLayers : 0))
                                observedLogits.append(SubmissionTensor(logits))
                                observedStates.append(cache.map(SubmissionCache.init))
                            }
                            let qsa = try #require(cache.last as? QSAKVCache)
                            #expect(qsa.derivedPooledBlockCount > 0)

                            // Use the production serialization contract, not
                            // raw `state`/`metaState` assignment: Mamba also
                            // requires its offset and occupied-slot indices.
                            // Disk round-trip gives the restored tensors their
                            // own storage before the live cache advances.
                            let snapshotDirectory = FileManager.default.temporaryDirectory
                                .appendingPathComponent("flash-early-state-\(UUID().uuidString)")
                            try FileManager.default.createDirectory(
                                at: snapshotDirectory, withIntermediateDirectories: true)
                            defer { try? FileManager.default.removeItem(at: snapshotDirectory) }
                            let prefixIDs = Array(2..<31) + [31, 37, 41, 43, 47]
                            let diskConfig = CacheCoordinatorConfig(
                                usePagedCache: false, enableDiskCache: true, diskCacheMaxGB: 0.1,
                                diskCacheDir: snapshotDirectory, modelKey: "flash-connected-state")
                            let topology = ModelCacheTopologySnapshot(cache: cache)
                            #expect(topology.requiresRecurrentSSMCompanionState)
                            #expect(!topology.requiresSeparateRecurrentPayloadState)
                            func makeCoordinator() -> CacheCoordinator {
                                let coordinator = CacheCoordinator(config: diskConfig)
                                coordinator.setHybrid(true,
                                    requiresRecurrentSSMCompanion: topology.requiresRecurrentSSMCompanionState,
                                    requiresSeparateRecurrentPayload: topology.requiresSeparateRecurrentPayloadState)
                                return coordinator
                            }
                            let writer = makeCoordinator()
                            writer.storeAfterGeneration(promptTokens: prefixIDs, perLayerData: [],
                                ssmStates: extractSSMStates(from: cache), cache: cache)
                            #expect(writer.hasValidatedDiskEntry(tokens: prefixIDs))
                            let reader = makeCoordinator()
                            #expect(reader.hasDurableDiskEntry(tokens: prefixIDs))
                            #expect(!reader.hasValidatedDiskEntry(tokens: prefixIDs))
                            guard case .hit(let matched, _, let detail, _, let ssm, let diskArrays) =
                                reader.fetch(tokens: prefixIDs + [53]) else {
                                Issue.record("Flash GDN/QSA/PLE boundary must survive coordinator reopen")
                                return
                            }
                            #expect(matched == prefixIDs.count && detail == .disk && ssm == nil)
                            let stored = try #require(diskArrays)
                            #expect(reader.hasValidatedDiskEntry(tokens: prefixIDs))
                            var restored = model.newCache(parameters: nil)
                            #expect(restoreFromDiskArrays(stored, into: &restored) == cache.last?.offset)
                            #expect(restored.map(\.offset) == cache.map(\.offset))
                            #expect((restored.last as? QSAKVCache)?.derivedPooledBlockCount == 0)
                            let next = MLXArray([Int32(53)]).reshaped(1, 1)
                            let continued = model(next, cache: cache)
                            let resumed = model(next, cache: restored)
                            MLX.eval(continued, resumed, cache, restored)
                            let restoredLogitsMatch = SubmissionTensor(continued) == SubmissionTensor(resumed)
                            let restoredStatesMatch = cache.map(SubmissionCache.init) == restored.map(SubmissionCache.init)
                            #expect(restoredLogitsMatch, "bits=\(bits) group=\(group) enabled=\(enabled)")
                            #expect(restoredStatesMatch, "bits=\(bits) group=\(group) enabled=\(enabled)")
                            observedLogits.append(SubmissionTensor(resumed))
                            observedStates.append(restored.map(SubmissionCache.init))
                            if enabled {
                                let allLogitsMatch = observedLogits == baselineLogits
                                let allStatesMatch = observedStates == baselineStates
                                #expect(allLogitsMatch, "bits=\(bits) group=\(group)")
                                #expect(allStatesMatch, "bits=\(bits) group=\(group)")
                            } else {
                                baselineLogits = observedLogits
                                baselineStates = observedStates
                            }
                        }
                    }
                }
            }
        }
    }

    @Test("early submission runs only for explicit AR, not native seed or one-token prepare")
    func earlySubmissionNativeRouting() throws {
        try MLXMetalTestLock.withLock {
            try withFixture { model in
                model.earlySubmissionEnabled = true
                let input = MLXArray([Int32(2)]).reshaped(1, 1)
                let prepared = try model.prepare(
                    LMInput(tokens: input), cache: model.newCache(parameters: nil), windowSize: 1)
                if case .logits(let output) = prepared { MLX.eval(output.logits) }
                #expect(model.earlySubmissionCount == 0)
                let seedCache = model.newCache(parameters: nil)
                let seed = model.nativeBackboneForward(input, cache: seedCache)
                MLX.eval(seed.logits, seed.hiddenStates, seedCache)
                #expect(model.earlySubmissionCount == 0)
                let arCache = model.newCache(parameters: nil)
                let ar = model.nativeAutoregressiveBackboneForward(input, cache: arCache)
                MLX.eval(ar.logits, ar.hiddenStates, arCache)
                #expect(model.earlySubmissionCount == model.config.base.textConfiguration.hiddenLayers)
                #expect(SubmissionTensor(ar.logits) == SubmissionTensor(seed.logits))
                #expect(SubmissionTensor(ar.hiddenStates) == SubmissionTensor(seed.hiddenStates))
                #expect(arCache.map(SubmissionCache.init) == seedCache.map(SubmissionCache.init))
            }
        }
    }

    @Test("composed HC and GDN normalization preserve mixed-quant logits, PLE, cache and disk continuation",
          arguments: [[1], [2], [1, 2]])
    func gdnQKNormConnectedState(pleLayerIDs: [Int]) throws {
        try MLXMetalTestLock.withLock {
            for (bits, group) in [([2, 3, 4], 32), ([4, 4, 4], 64), ([6, 4, 6], 64)] {
                try withFixture(routedBits: bits, routedGroupSize: group,
                                gdnHeadDimension: 128, pleLayerIDs: pleLayerIDs,
                                additionalLinearLayer: true, distinctHCNorms: true) { model in
                    var referenceLogits: [SubmissionTensor] = []
                    var referenceStates: [[SubmissionCache]] = []
                    for (enabled, hcEnabled) in [(false, false), (true, false), (false, true), (true, true)] {
                        model.gdnQKNormEnabled = enabled
                        model.hcCombineNormEnabled = hcEnabled
                        let cache = model.newCache(parameters: nil)
                        var logits: [SubmissionTensor] = []
                        var states: [[SubmissionCache]] = []
                        for ids in [Array(2..<31), [31], [37, 41, 43], [47]] {
                            let before = model.gdnQKNormCallCount
                            let hcBefore = model.hcCombineNormCallCount
                            let output = model(MLXArray(ids.map(Int32.init)).reshaped(1, ids.count), cache: cache)
                            MLX.eval(output, cache)
                            // Two generated GDN layers, no prefill dispatch.
                            #expect(model.gdnQKNormCallCount - before ==
                                (enabled && ids.count == 1 ? 2 : 0))
                            // Three intra-layer pairs plus the next/final mixers. Preparing
                            // the next attention norm is forbidden across PLE.
                            let hcPairs = pleLayerIDs.contains(2) ? 5 : 6
                            #expect(model.hcCombineNormCallCount - hcBefore ==
                                (hcEnabled && ids.count == 1 ? hcPairs : 0))
                            logits.append(SubmissionTensor(output))
                            states.append(cache.map(SubmissionCache.init))
                        }
                        let directory = FileManager.default.temporaryDirectory
                            .appendingPathComponent("flash-gdn-norm-\(UUID().uuidString)")
                        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                        defer { try? FileManager.default.removeItem(at: directory) }
                        let saved = cache.map { $0.copy() }
                        MLX.eval(saved)
                        let payload = TQDiskSerializer.serialize(cache: saved)
                        MLX.eval(Array(payload.values))
                        let path = directory.appendingPathComponent("prefix.safetensors")
                        try MLX.save(arrays: payload, url: path)
                        var restored = model.newCache(parameters: nil)
                        let offset = restoreFromDiskArrays(try MLX.loadArrays(url: path), into: &restored)
                        #expect(validateRestoredCacheBoundary(restored, matchedTokens: 34, restoredTokens: offset))
                        let next = MLXArray([Int32(53)]).reshaped(1, 1)
                        let continued = model(next, cache: cache)
                        let resumed = model(next, cache: restored)
                        MLX.eval(continued, resumed, cache, restored)
                        #expect(SubmissionTensor(continued) == SubmissionTensor(resumed))
                        #expect(cache.map(SubmissionCache.init) == restored.map(SubmissionCache.init))
                        logits.append(SubmissionTensor(resumed))
                        states.append(restored.map(SubmissionCache.init))
                        if enabled || hcEnabled {
                            #expect(logits == referenceLogits, "bits=\(bits) group=\(group)")
                            #expect(states == referenceStates, "bits=\(bits) group=\(group)")
                        } else {
                            referenceLogits = logits
                            referenceStates = states
                        }
                    }
                }
            }
        }
    }

    @Test("GDN Q/K norm only runs in explicit AR, never seed, prepare or verify")
    func gdnQKNormNativeRouting() throws {
        try MLXMetalTestLock.withLock {
            try withFixture(routedBits: [2, 3, 4], gdnHeadDimension: 128,
                            distinctHCNorms: true) { model in
                model.gdnQKNormEnabled = true
                model.hcCombineNormEnabled = true
                let input = MLXArray([Int32(2)]).reshaped(1, 1)
                let prepared = try model.prepare(
                    LMInput(tokens: input), cache: model.newCache(parameters: nil), windowSize: 1)
                if case .logits(let output) = prepared { MLX.eval(output.logits) }
                let seedCache = model.newCache(parameters: nil)
                let seed = model.nativeBackboneForward(input, cache: seedCache)
                MLX.eval(seed.logits, seed.hiddenStates, seedCache)
                let verifyCache = model.newCache(parameters: nil)
                let verify = model.nativeBackboneMTPVerifyForward(input, cache: verifyCache)
                MLX.eval(verify.logits, verify.hiddenStates, verifyCache)
                #expect(model.gdnQKNormCallCount == 0)
                #expect(model.hcCombineNormCallCount == 0)
                let arCache = model.newCache(parameters: nil)
                let ar = model.nativeAutoregressiveBackboneForward(input, cache: arCache)
                MLX.eval(ar.logits, ar.hiddenStates, arCache)
                #expect(model.gdnQKNormCallCount == 1)
                #expect(model.hcCombineNormCallCount == 4)
                #expect(SubmissionTensor(ar.logits) == SubmissionTensor(seed.logits))
                #expect(SubmissionTensor(ar.hiddenStates) == SubmissionTensor(seed.hiddenStates))
                #expect(arCache.map(SubmissionCache.init) == seedCache.map(SubmissionCache.init))
            }
        }
    }

    @Test("native Flash token cap never publishes pending tokens under an emitted-prefix key",
          .enabled(if: ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0",
                   "Requires strict quantized-fixture oracle"), arguments: [2, 6])
    func nativeFinalBoundaryDiskPublication(inputBits: Int) throws {
        try MLXMetalTestLock.withLock {
            try withFixture(routedBits: [2, 3, 4], mtpEnabled: true,
                            inputProjectionBits: inputBits) { model in
                for cap in [2, 3, 4, 9, 32] {
                    let root = FileManager.default.temporaryDirectory
                        .appendingPathComponent("flash-final-boundary-\(UUID().uuidString)")
                    defer { try? FileManager.default.removeItem(at: root) }
                    let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
                        usePagedCache: false, enableDiskCache: true, diskCacheMaxGB: 1,
                        diskCacheDir: root, modelKey: "flash-final-\(inputBits)-\(cap)"))
                    // Let the production iterator derive topology from the
                    // actual PLE/GDN/QSA caches, exactly as an ordinary load does.
                    let promptIDs = [2, 3, 4]
                    let prompt = MLXArray(promptIDs.map(Int32.init)).reshaped(1, 3)
                    var parameters = GenerateParameters(maxTokens: cap, temperature: 1, topP: 0.95, topK: 20)
                    parameters.randomSeed = 829
                    parameters.draftStrategy = .nativeMTP(depth: 3)
                    parameters.nativeMTPDepthPolicy = .fixed
                    var iterator = try NativeMTPTokenIterator(
                        input: LMInput(tokens: prompt), model: model, parameters: parameters,
                        depth: 3, cacheCoordinator: coordinator)
                    let started = Date()
                    var emitted: [Int] = []
                    while let token = iterator.next() { emitted.append(token) }
                    #expect(emitted.count == cap)
                    #expect(iterator.mtpForwardCount > 0, "native MTP must actually execute")
                    iterator.storeCacheAfterGeneration(generatedTokenIds: emitted, includeGeneratedBoundary: true)
                    let disk = try #require(coordinator.diskCache)
                    // Exercise the production restore caller too; raw serializer
                    // comparisons alone could omit a legitimate companion restore.
                    let continuation = LMInput(tokens: MLXArray([Int32(2), 3, 4, 73]).reshaped(1, 4))
                    var cachedIterator = try NativeMTPTokenIterator(
                        input: continuation, model: model, parameters: parameters, depth: 3,
                        cacheCoordinator: coordinator)
                    var freshIterator = try NativeMTPTokenIterator(
                        input: continuation, model: model, parameters: parameters, depth: 3)
                    for (layer, pair) in zip(cachedIterator.cache, freshIterator.cache).enumerated() {
                        #expect(pair.0.offset == pair.1.offset)
                        #expect(pair.0.state.count == pair.1.state.count)
                        for (slot, values) in zip(pair.0.state, pair.1.state).enumerated() {
                            expectEqual(values.0, values.1,
                                        "native restored inputBits\(inputBits) cap\(cap) layer\(layer) slot\(slot)")
                        }
                    }
                    cachedIterator.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
                    freshIterator.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
                    // A missing post-answer snapshot is allowed for non-trimmable
                    // recurrent state. A snapshot published under this key must
                    // represent exactly the emitted prefix, never queued tokens.
                    var restoredCount = 0
                    for ids in [promptIDs, promptIDs + emitted] {
                        guard let arrays = disk.fetch(
                            tokens: ids,
                            mediaSalt: computeCacheSalt(for: LMInput(tokens: prompt), parameters: parameters))
                        else { continue }
                        var restored = model.newCache(parameters: parameters)
                        #expect(restoreFromDiskArrays(arrays, into: &restored) == ids.count)
                        let reference = model.newCache(parameters: parameters)
                        MLX.eval(model.nativeBackboneForward(prompt, cache: reference).logits)
                        for id in ids.dropFirst(promptIDs.count) {
                            MLX.eval(model.nativeBackboneForward(
                                MLXArray([Int32(id)]).reshaped(1, 1), cache: reference).logits)
                        }
                        for (layer, pair) in zip(restored, reference).enumerated() {
                            #expect(pair.0.offset == pair.1.offset)
                            #expect(pair.0.state.count == pair.1.state.count)
                            for (slot, values) in zip(pair.0.state, pair.1.state).enumerated() {
                                expectEqual(values.0, values.1,
                                            "disk inputBits\(inputBits) cap\(cap) boundary\(ids.count) layer\(layer) slot\(slot)")
                            }
                        }
                        let followup = MLXArray([Int32(73)]).reshaped(1, 1)
                        expectEqual(model(followup, cache: restored), model(followup, cache: reference),
                                    "disk continuation inputBits\(inputBits) cap\(cap) boundary\(ids.count)")
                        restoredCount += 1
                    }
                    #expect(restoredCount > 0, "positive disk publication control must execute")
                    print("FLASH-FINAL-DISK inputBits=\(inputBits) cap=\(cap) restored=\(restoredCount) emitted=\(emitted.count) fixtureTokS=\(Double(emitted.count) / max(Date().timeIntervalSince(started), 1e-9)) realModelSpeedProof=false")
                }
            }
        }
    }

    @Test("Flash text prepare honors its chunk budget without changing continuation state",
          .enabled(if: ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0",
                   "Requires strict fixture oracle"), arguments: [0, 2, 6])
    func textPrefillChunkBudget(inputBits: Int) throws {
        try MLXMetalTestLock.withLock {
            try withFixture(routedBits: inputBits == 0 ? [] : [2, 3, 4],
                            inputProjectionBits: inputBits == 0 ? nil : inputBits) { model in
                let tokens = MLXArray((2..<15).map(Int32.init)).reshaped(1, 13)
                let referenceCache = model.newCache(parameters: nil)
                let referenceLogits = model(tokens, cache: referenceCache)
                MLX.eval(referenceLogits, referenceCache)
                for step in [1, 2, 4, 8, 13] {
                    let cache = model.newCache(parameters: nil)
                    guard case .logits(let output) = try model.prepare(
                        LMInput(tokens: tokens), cache: cache, windowSize: step)
                    else { Issue.record("text prepare must return final-chunk logits"); continue }
                    MLX.eval(output.logits, cache)
                    #expect(output.logits.dim(1) <= step,
                            "prepare must not retain full-prompt logits beyond the supplied chunk budget")
                    expectEqual(output.logits[0..., (-1)..., 0...],
                                referenceLogits[0..., (-1)..., 0...], "chunk\(step) final logits")
                    for (layer, pair) in zip(cache, referenceCache).enumerated() {
                        #expect(pair.0.offset == pair.1.offset)
                        #expect(pair.0.state.count == pair.1.state.count)
                        for (slot, states) in zip(pair.0.state, pair.1.state).enumerated() {
                            expectEqual(states.0, states.1, "chunk\(step) layer\(layer) state\(slot)")
                        }
                    }
                    let next = MLXArray([Int32(73)]).reshaped(1, 1)
                    let referenceCopy = referenceCache.map { $0.copy() }
                    MLX.eval(referenceCopy)
                    expectEqual(model(next, cache: cache), model(next, cache: referenceCopy),
                                "chunk\(step) continuation")
                }
            }
        }
    }

    @Test("Flash default prefill budget reports real progress past the QSA budget",
          .enabled(if: ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0",
                   "Requires strict fixture oracle"))
    func defaultPrefillBudgetAndProgress() throws {
        try MLXMetalTestLock.withLock {
            try withFixture { model in
                let tokens = MLXArray((0..<513).map { Int32(2 + $0 % 100) }).reshaped(1, 513)
                let referenceCache = model.newCache(parameters: nil)
                let reference = model(tokens, cache: referenceCache)
                MLX.eval(reference, referenceCache)
                for step: Int? in [nil, 64, 0, -1] {
                    let cache = model.newCache(parameters: nil)
                    let recorder = ProgressRecorder()
                    let prepared = try PrefillProgressReporter.withHandler({ recorder.append($0) }) {
                        try model.prepare(LMInput(tokens: tokens), cache: cache, windowSize: step)
                    }
                    guard case .logits(let output) = prepared else {
                        Issue.record("expected final logits"); continue
                    }
                    MLX.eval(output.logits, cache)
                    let effectiveStep = step ?? 512
                    #expect(output.logits.dim(1) == (effectiveStep > 0 ? 1 : 513))
                    #expect(recorder.snapshot() == (effectiveStep > 0
                        ? Array(stride(from: effectiveStep, to: 513, by: effectiveStep)) : []))
                    expectEqual(output.logits[0..., (-1)..., 0...],
                                reference[0..., (-1)..., 0...], "default/long QSA final logits")
                    let referenceCopy = referenceCache.map { $0.copy() }
                    MLX.eval(referenceCopy)
                    let next = MLXArray([Int32(73)]).reshaped(1, 1)
                    expectEqual(model(next, cache: cache), model(next, cache: referenceCopy),
                                "default/long QSA continuation")
                }
            }
        }
    }

    @Test("Flash text prefill cancellation does not advance the cache")
    func cancelledPrefillDoesNotAdvanceCache() async throws {
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            try MLXMetalTestLock.withLock {
                try withFixture { model in
                    let cache = model.newCache(parameters: nil)
                    do {
                        _ = try model.prepare(
                            LMInput(tokens: MLXArray([Int32(2), 3, 4]).reshaped(1, 3)),
                            cache: cache, windowSize: 1)
                        Issue.record("cancelled prepare unexpectedly completed")
                    } catch is CancellationError {
                        #expect(cache.allSatisfy { $0.offset == 0 })
                    }
                }
            }
        }
        try await task.value
    }

    private func expectEqual(_ actual: MLXArray, _ expected: MLXArray, _ label: String) {
        #expect(actual.shape == expected.shape, "\(label) shape")
        guard actual.shape == expected.shape else { return }
        let a = actual.asType(.float32)
        let b = expected.asType(.float32)
        let error = MLX.max(abs(a - b)).item(Float.self)
        if !allClose(a, b, rtol: 1e-5, atol: 1e-5).item(Bool.self) {
            let actualValues = a.reshaped(-1).asArray(Float.self)
            let expectedValues = b.reshaped(-1).asArray(Float.self)
            if let index = actualValues.indices.first(where: {
                abs(actualValues[$0] - expectedValues[$0]) > 1e-5 + 1e-5 * abs(expectedValues[$0])
            }) {
                print("FLASH-PARITY-DIFF \(label) shape=\(a.shape) dtype=\(actual.dtype) index=\(index) actual=\(actualValues[index]) expected=\(expectedValues[index]) maxAbs=\(error)")
            }
        }
        #expect(
            allClose(a, b, rtol: 1e-5, atol: 1e-5).item(Bool.self),
            "\(label) maxAbs=\(error)")
    }

}
