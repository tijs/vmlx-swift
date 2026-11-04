// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

@Suite("experimental Qwen35 VLM affine decoder", .serialized)
struct Qwen35AffineResearchDecoderTests {
    private let prefix = "language_model.model.layers.0.mlp.switch_mlp."
    private let limits = AffineStreamingSwitchGLU.ResearchLimits(
        maxTokens: 65_536, maxRoutesPerToken: 4, maxWindowTokens: 128,
        maxGroupSize: 32, maxMappedExpertBytes: 2 * 1024 * 1024,
        maxRetainedOutputBytes: 128 * 1024 * 1024)

    private func fixture() throws -> (URL, [String: MLXArray]) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen35-research-decoder-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        do {
            var tensors: [String: MLXArray] = [:]
            for (projection, name) in ["gate_proj", "up_proj", "down_proj"].enumerated() {
                var weights: [MLXArray] = []
                var scales: [MLXArray] = []
                var biases: [MLXArray] = []
                for expert in 0 ..< 4 {
                    let values = (0 ..< 64 * 64).map { i in
                        Float((i * 13 + expert * 17 + projection * 23) % 101 - 50) / 200
                    }
                    let q = MLX.quantized(MLXArray(values, [64, 64]),
                        groupSize: 64, bits: 4, mode: .affine)
                    weights.append(q.wq)
                    scales.append(q.scales.asType(.bfloat16))
                    biases.append(q.biases!.asType(.bfloat16))
                }
                tensors[prefix + name + ".weight"] = stacked(weights, axis: 0)
                tensors[prefix + name + ".scales"] = stacked(scales, axis: 0)
                tensors[prefix + name + ".biases"] = stacked(biases, axis: 0)
            }
            MLX.eval(Array(tensors.values))
            let shard = "model.safetensors"
            let shardURL = directory.appendingPathComponent(shard)
            try MLX.save(arrays: tensors, url: shardURL)
            let saved = try Data(contentsOf: shardURL)
            let oldLength = saved.prefix(8).withUnsafeBytes {
                $0.loadUnaligned(as: UInt64.self).littleEndian
            }
            var header = saved.subdata(in: 8 ..< (8 + Int(oldLength)))
            header.append(Data(repeating: 0x20, count: (4096 - (8 + header.count) % 4096) % 4096))
            var rewritten = Data()
            var headerLength = UInt64(header.count).littleEndian
            rewritten.append(contentsOf: withUnsafeBytes(of: &headerLength) { Array($0) })
            rewritten.append(header)
            rewritten.append(saved.suffix(from: 8 + Int(oldLength)))
            try rewritten.write(to: shardURL)
            let map = Dictionary(uniqueKeysWithValues: tensors.keys.map { ($0, shard) })
            try JSONSerialization.data(withJSONObject: ["weight_map": map], options: [.sortedKeys])
                .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
            return (directory, tensors)
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    @Test("attention and routed forward match and failures require cache discard")
    func decoderParityAndFailure() throws {
        try MLXMetalTestLock.withLock {
            let (directory, tensors) = try fixture()
            defer { try? FileManager.default.removeItem(at: directory) }
            let json = """
                {"hidden_size":64,"num_hidden_layers":1,"num_attention_heads":4,
                 "num_key_value_heads":4,"head_dim":16,"full_attention_interval":1,
                 "num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":64,
                 "shared_expert_intermediate_size":64,"norm_topk_prob":true}
                """
            let config = try JSONDecoder().decode(Qwen35Configuration.TextConfiguration.self,
                from: Data(json.utf8))
            let decoder = Qwen35Language.DecoderLayer(config, layerIdx: 0)
            try decoder.update(parameters: decoder.parameters().mapValues { $0.asType(.bfloat16) }, verify: .all)
            guard let sparse = decoder.mlp as? Qwen35Language.SparseMoeBlock,
                let native = sparse.switchMLP as? SwitchGLU else {
                throw AffineStreamingExpertError.invalidInput("fixture requires affine sparse block")
            }
            func projection(_ name: String) throws -> QuantizedSwitchLinear {
                QuantizedSwitchLinear(inputDims: 64, outputDims: 64, numExperts: 4,
                    weight: try #require(tensors[prefix + name + ".weight"]),
                    scales: try #require(tensors[prefix + name + ".scales"]),
                    biases: try #require(tensors[prefix + name + ".biases"]),
                    groupSize: 64, bits: 4, mode: .affine)
            }
            try native.update(modules: ModuleChildren.unflattened([
                ("gate_proj", try projection("gate_proj")),
                ("up_proj", try projection("up_proj")),
                ("down_proj", try projection("down_proj")),
            ]), verify: .all)
            let catalog = AffineStreamingExpertCatalog(numExperts: 4, inputDims: 64,
                hiddenDims: 64, layout: .qwen35)
            try catalog.configure(modelDirectory: directory, layerCount: 1)
            let reducer = AffineStreamingSwitchGLU(inputDims: 64, hiddenDims: 64,
                numExperts: 4, layerIndex: 0, catalog: catalog)
            let values: [Float] = (0 ..< 2 * 64).map { index in
                let centered: Int = (index * 17) % 101 - 50
                return Float(centered) / Float(37)
            }
            let x = MLXArray(values, [1, 2, 64]).asType(.bfloat16)
            let referenceMode = createAttentionMask(h: x, cache: nil as KVCache?, returnArray: true)
            let referenceMask: MLXArray?
            if case .array(let array) = referenceMode { referenceMask = array } else { referenceMask = nil }
            let expected = decoder(x, attentionMask: referenceMask, ssmMask: nil, cache: nil, positionIds: nil)
            let actual = try decoder.researchAffineForward(x, attentionMask: referenceMask, ssmMask: nil,
                cache: nil, positionIds: nil, reducer: reducer, limits: limits,
                groupLimit: 16, tokenWindow: 8)
            MLX.eval(expected, actual)
            let difference = MLX.max(MLX.abs(expected.asType(.float32) - actual.asType(.float32))).item(Float.self)
            print("QWEN35_AFFINE_DECODER_PARITY max_abs=\(difference)")
            #expect(expected.shape == actual.shape && expected.dtype == actual.dtype)
            #expect(MLX.arrayEqual(expected, actual).item(Bool.self), "max_abs=\(difference)")

            let failedCache = KVCacheSimple()
            // Match Model's production ordering: create the offset-aware mask
            // before attention mutates the cache. A nil mask here would cause
            // the attention helper to ask the already-updated cache for one.
            let mode = createAttentionMask(h: x, cache: failedCache, returnArray: true)
            let attentionMask: MLXArray?
            if case .array(let array) = mode { attentionMask = array } else { attentionMask = nil }
            var checks = 0
            #expect(throws: CancellationError.self) {
                try decoder.researchAffineForward(x, attentionMask: attentionMask, ssmMask: nil,
                    cache: failedCache, positionIds: nil, reducer: reducer, limits: limits,
                    groupLimit: 16, tokenWindow: 8, shouldCancel: {
                        checks += 1
                        return checks >= 2
                    })
            }
            #expect(failedCache.offset == 2)
            let wrongLayer = AffineStreamingSwitchGLU(inputDims: 64, hiddenDims: 64,
                numExperts: 4, layerIndex: 1, catalog: catalog)
            let untouched = KVCacheSimple()
            #expect(throws: AffineStreamingExpertError.self) {
                try decoder.researchAffineForward(x, attentionMask: nil, ssmMask: nil,
                    cache: untouched, positionIds: nil, reducer: wrongLayer, limits: limits,
                    groupLimit: 16, tokenWindow: 8)
            }
            #expect(untouched.offset == 0)
            let missingMaskCache = KVCacheSimple()
            #expect(throws: AffineStreamingExpertError.self) {
                try decoder.researchAffineForward(x, attentionMask: nil, ssmMask: nil,
                    cache: missingMaskCache, positionIds: nil, reducer: reducer, limits: limits,
                    groupLimit: 16, tokenWindow: 8)
            }
            #expect(missingMaskCache.offset == 0)
        }
    }

    @Test("restored research cache requires explicit valid positions")
    func restoredCachePositionGuard() throws {
        try MLXMetalTestLock.withLock {
            let json = """
                {"model_type":"qwen3_5_moe","text_config":{
                 "hidden_size":64,"num_hidden_layers":1,"num_attention_heads":4,
                 "num_key_value_heads":4,"head_dim":16,"full_attention_interval":1,
                 "vocab_size":32,"num_experts":4,"num_experts_per_tok":2,
                 "moe_intermediate_size":64,"shared_expert_intermediate_size":64}}
                """
            let config = try JSONDecoder().decode(Qwen35Configuration.self, from: Data(json.utf8))
            let language = Qwen35Language.LanguageModel(config)
            let cache = KVCacheSimple()
            let keys = MLXArray.zeros([1, 4, 2, 16], dtype: .bfloat16)
            _ = cache.update(keys: keys, values: keys)
            let tokens = MLXArray([Int32(1)], [1, 1])
            do {
                _ = try language.researchAffineForward(tokens, cache: [cache], reducers: [],
                    limits: limits, groupLimit: 16, tokenWindow: 8)
                Issue.record("restored cache without positions was accepted")
            } catch AffineStreamingExpertError.invalidInput(let reason) {
                #expect(reason.contains("restored research cache requires explicit position IDs"))
            } catch { Issue.record("unexpected error: \(error)") }
            #expect(cache.offset == 2)
            let badPositions = MLXArray([Int32(2)], [1, 1])
            do {
                _ = try language.researchAffineForward(tokens, cache: [cache], positionIds: badPositions,
                    reducers: [], limits: limits, groupLimit: 16, tokenWindow: 8)
                Issue.record("invalid position shape was accepted")
            } catch AffineStreamingExpertError.invalidInput(let reason) {
                #expect(reason.contains("invalid research position IDs"))
            } catch { Issue.record("unexpected error: \(error)") }
            let positions = MLXArray([Int32](repeating: 2, count: 3), [3, 1, 1])
            do {
                _ = try language.researchAffineForward(tokens, cache: [cache], positionIds: positions,
                    reducers: [], limits: limits, groupLimit: 16, tokenWindow: 8)
                Issue.record("missing reducers were accepted")
            } catch AffineStreamingExpertError.invalidInput(let reason) {
                #expect(reason.contains("invalid research model inputs or caches"))
            } catch { Issue.record("unexpected error: \(error)") }
            #expect(cache.offset == 2)
        }
    }

    @Test("text language prefill and decode match with independent caches")
    func languagePrefillAndDecodeParity() throws {
        try MLXMetalTestLock.withLock {
            let (directory, tensors) = try fixture()
            defer { try? FileManager.default.removeItem(at: directory) }
            let json = """
                {"model_type":"qwen3_5_moe","text_config":{
                 "hidden_size":64,"num_hidden_layers":1,"num_attention_heads":4,
                 "num_key_value_heads":4,"head_dim":16,"full_attention_interval":1,
                 "vocab_size":32,"num_experts":4,"num_experts_per_tok":2,
                 "moe_intermediate_size":64,"shared_expert_intermediate_size":64,
                 "norm_topk_prob":true}}
                """
            let config = try JSONDecoder().decode(Qwen35Configuration.self, from: Data(json.utf8))
            let language = Qwen35Language.LanguageModel(config)
            try language.update(parameters: language.parameters().mapValues { $0.asType(.bfloat16) }, verify: .all)
            let decoder = try #require(language.model.children().flattened()
                .compactMap { $0.1 as? Qwen35Language.DecoderLayer }.first)
            let sparse = try #require(decoder.mlp as? Qwen35Language.SparseMoeBlock)
            let native = try #require(sparse.switchMLP as? SwitchGLU)
            func projection(_ name: String) throws -> QuantizedSwitchLinear {
                QuantizedSwitchLinear(inputDims: 64, outputDims: 64, numExperts: 4,
                    weight: try #require(tensors[prefix + name + ".weight"]),
                    scales: try #require(tensors[prefix + name + ".scales"]),
                    biases: try #require(tensors[prefix + name + ".biases"]),
                    groupSize: 64, bits: 4, mode: .affine)
            }
            try native.update(modules: ModuleChildren.unflattened([
                ("gate_proj", try projection("gate_proj")),
                ("up_proj", try projection("up_proj")),
                ("down_proj", try projection("down_proj")),
            ]), verify: .all)
            let catalog = AffineStreamingExpertCatalog(numExperts: 4, inputDims: 64,
                hiddenDims: 64, layout: .qwen35)
            try catalog.configure(modelDirectory: directory, layerCount: 1)
            let reducer = AffineStreamingSwitchGLU(inputDims: 64, hiddenDims: 64,
                numExperts: 4, layerIndex: 0, catalog: catalog)
            let prompt = MLXArray([Int32(1), 2, 3], [1, 3])
            let next = MLXArray([Int32(4)], [1, 1])
            let nativeCache = KVCacheSimple()
            language.resetPositionState()
            let nativePrompt = language(prompt, cache: [nativeCache]).logits
            MLX.eval(nativePrompt)
            #expect(nativeCache.offset == 3)
            let nativeNext = language(next, cache: [nativeCache]).logits
            MLX.eval(nativeNext)
            #expect(nativeCache.offset == 4)
            let researchCache = KVCacheSimple()
            language.resetPositionState()
            let researchPrompt = try language.researchAffineForward(prompt, cache: [researchCache],
                reducers: [reducer], limits: limits, groupLimit: 16, tokenWindow: 2).logits
            MLX.eval(researchPrompt)
            #expect(researchCache.offset == 3)
            let researchNext = try language.researchAffineForward(next, cache: [researchCache],
                reducers: [reducer], limits: limits, groupLimit: 16, tokenWindow: 2).logits
            MLX.eval(researchNext)
            #expect(researchCache.offset == 4)
            for (name, expected, actual) in [("prefill", nativePrompt, researchPrompt),
                                             ("decode", nativeNext, researchNext)] {
                let difference = MLX.max(MLX.abs(expected.asType(.float32) - actual.asType(.float32)))
                    .item(Float.self)
                print("QWEN35_AFFINE_LANGUAGE_PARITY phase=\(name) max_abs=\(difference)")
                #expect(expected.shape == actual.shape && expected.dtype == actual.dtype)
                #expect(MLX.arrayEqual(expected, actual).item(Bool.self),
                    "phase=\(name) max_abs=\(difference)")
            }
            // A later request failure clears per-request RoPE state. The KV
            // cache is intentionally still mutated and must be discarded.
            #expect(throws: AffineStreamingExpertError.self) {
                try language.researchAffineForward(next, cache: [researchCache], reducers: [],
                    limits: limits, groupLimit: 16, tokenWindow: 2)
            }
            #expect(researchCache.offset == 4)
            do {
                _ = try language.researchAffineForward(next, cache: [researchCache], reducers: [reducer],
                    limits: limits, groupLimit: 16, tokenWindow: 2)
                Issue.record("cache alias was reused without restored positions")
            } catch AffineStreamingExpertError.invalidInput(let reason) {
                #expect(reason.contains("restored research cache requires explicit position IDs"))
            } catch { Issue.record("unexpected post-failure error: \(error)") }
        }
    }
}
