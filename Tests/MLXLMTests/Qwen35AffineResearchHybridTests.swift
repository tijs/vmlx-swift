// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// A synthetic two-layer model, not a full checkpoint or memory admission test.
@Suite("experimental Qwen35 hybrid affine forward", .serialized)
struct Qwen35AffineResearchHybridTests {
    private enum FixtureFailure: Error, Equatable { case prefill, decode }

    /// Supplies the iterator with fresh hybrid caches while the test controls
    /// the fallible forward explicitly.
    private final class IteratorModel: Module, LanguageModel {
        let language: Qwen35Language.LanguageModel
        var kvHeads: [Int] { [1, 4] }

        init(_ language: Qwen35Language.LanguageModel) { self.language = language; super.init() }
        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            [MambaCache(), KVCacheSimple()]
        }
        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            .tokens(input.text)
        }
        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            language(inputs, cache: cache?.map { $0 as KVCache? }).logits
        }
    }

    private let limits = AffineStreamingSwitchGLU.ResearchLimits(
        maxTokens: 65_536, maxRoutesPerToken: 4, maxWindowTokens: 128,
        maxGroupSize: 64, maxMappedExpertBytes: 2 * 1024 * 1024,
        maxRetainedOutputBytes: 128 * 1024 * 1024)

    private func fixture() throws -> (URL, [String: MLXArray]) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen35-research-hybrid-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        do {
            var tensors: [String: MLXArray] = [:]
            for layer in 0 ..< 2 {
                for (projection, name) in ["gate_proj", "up_proj", "down_proj"].enumerated() {
                    var weights: [MLXArray] = []
                    var scales: [MLXArray] = []
                    var biases: [MLXArray] = []
                    for expert in 0 ..< 4 {
                        let values: [Float] = (0 ..< 64 * 64).map { index in
                            let centered = (index * 13 + expert * 17 + projection * 23 + layer * 11) % 101 - 50
                            return Float(centered) / Float(200)
                        }
                        let q = MLX.quantized(MLXArray(values, [64, 64]),
                            groupSize: 64, bits: 4, mode: .affine)
                        weights.append(q.wq)
                        scales.append(q.scales.asType(.bfloat16))
                        biases.append(q.biases!.asType(.bfloat16))
                    }
                    let prefix = "language_model.model.layers.\(layer).mlp.switch_mlp.\(name)"
                    tensors[prefix + ".weight"] = stacked(weights, axis: 0)
                    tensors[prefix + ".scales"] = stacked(scales, axis: 0)
                    tensors[prefix + ".biases"] = stacked(biases, axis: 0)
                }
            }
            MLX.eval(Array(tensors.values))
            let shardName = "model.safetensors"
            let shard = directory.appendingPathComponent(shardName)
            try MLX.save(arrays: tensors, url: shard)
            let saved = try Data(contentsOf: shard)
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
            try rewritten.write(to: shard)
            let map = Dictionary(uniqueKeysWithValues: tensors.keys.map { ($0, shardName) })
            try JSONSerialization.data(withJSONObject: ["weight_map": map], options: [.sortedKeys])
                .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
            return (directory, tensors)
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    @Test("hybrid recurrent and attention caches match for prefill and decode")
    func hybridPrefillDecodeParity() throws {
        try MLXMetalTestLock.withLock {
            let (directory, tensors) = try fixture()
            defer { try? FileManager.default.removeItem(at: directory) }
            let json = """
                {"model_type":"qwen3_5_moe","text_config":{
                 "hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,
                 "num_key_value_heads":4,"head_dim":16,"full_attention_interval":2,
                 "linear_num_value_heads":4,"linear_num_key_heads":2,
                 "linear_key_head_dim":8,"linear_value_head_dim":8,"linear_conv_kernel_dim":4,
                 "vocab_size":32,"num_experts":4,"num_experts_per_tok":2,
                 "moe_intermediate_size":64,"shared_expert_intermediate_size":64,
                 "norm_topk_prob":true}}
                """
            let config = try JSONDecoder().decode(Qwen35Configuration.self, from: Data(json.utf8))
            let language = Qwen35Language.LanguageModel(config)
            try language.update(parameters: language.parameters().mapValues { $0.asType(.bfloat16) }, verify: .all)
            let layers = language.model.children().flattened()
                .compactMap { $0.1 as? Qwen35Language.DecoderLayer }
                .sorted {
                    (($0.mlp as? Qwen35Language.SparseMoeBlock)?.researchLayerIndex ?? -1)
                        < (($1.mlp as? Qwen35Language.SparseMoeBlock)?.researchLayerIndex ?? -1)
                }
            #expect(layers.count == 2)
            guard layers.count == 2, layers[0].isLinear, !layers[1].isLinear else {
                throw AffineStreamingExpertError.invalidInput("fixture requires one recurrent and one full attention layer")
            }
            for (index, layer) in layers.enumerated() {
                let sparse = try #require(layer.mlp as? Qwen35Language.SparseMoeBlock)
                let native = try #require(sparse.switchMLP as? SwitchGLU)
                let prefix = "language_model.model.layers.\(index).mlp.switch_mlp."
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
            }
            let catalog = AffineStreamingExpertCatalog(numExperts: 4, inputDims: 64,
                hiddenDims: 64, layout: .qwen35)
            try catalog.configure(modelDirectory: directory, layerCount: 2)
            let reducers = (0 ..< 2).map { layer in
                AffineStreamingSwitchGLU(inputDims: 64, hiddenDims: 64,
                    numExperts: 4, layerIndex: layer, catalog: catalog)
            }
            let prompt = MLXArray([Int32(1), 2, 3], [1, 3])
            let next = MLXArray([Int32(4)], [1, 1])
            let nativeSSM = MambaCache()
            let nativeAttention = KVCacheSimple()
            let nativeCaches: [KVCache?] = [nativeSSM, nativeAttention]
            language.resetPositionState()
            let nativePrefill = language(prompt, cache: nativeCaches).logits
            MLX.eval(nativePrefill)
            let nativePrefillState = nativeSSM.state.map { $0 * 1 }
            MLX.eval(nativePrefillState)
            #expect(nativeSSM.offset == 3 && nativeAttention.offset == 3)
            let nativeDecode = language(next, cache: nativeCaches).logits
            MLX.eval(nativeDecode)
            let nativeDecodeState = nativeSSM.state.map { $0 * 1 }
            MLX.eval(nativeDecodeState)
            #expect(nativeSSM.offset == 4 && nativeAttention.offset == 4)

            let researchSSM = MambaCache()
            let researchAttention = KVCacheSimple()
            let researchCaches: [KVCache?] = [researchSSM, researchAttention]
            language.resetPositionState()
            let researchPrefill = try language.researchAffineForward(prompt, cache: researchCaches,
                reducers: reducers, limits: limits, groupLimit: 16, tokenWindow: 2).logits
            MLX.eval(researchPrefill)
            let researchPrefillState = researchSSM.state.map { $0 * 1 }
            MLX.eval(researchPrefillState)
            #expect(researchSSM.offset == 3 && researchAttention.offset == 3)
            let researchDecode = try language.researchAffineForward(next, cache: researchCaches,
                reducers: reducers, limits: limits, groupLimit: 16, tokenWindow: 2).logits
            MLX.eval(researchDecode)
            let researchDecodeState = researchSSM.state.map { $0 * 1 }
            MLX.eval(researchDecodeState)
            #expect(researchSSM.offset == 4 && researchAttention.offset == 4)
            for (phase, expected, actual) in [("prefill", nativePrefill, researchPrefill),
                                               ("decode", nativeDecode, researchDecode)] {
                let error = MLX.max(MLX.abs(expected.asType(.float32) - actual.asType(.float32)))
                    .item(Float.self)
                print("QWEN35_AFFINE_HYBRID_PARITY phase=\(phase) max_abs=\(error)")
                #expect(expected.shape == actual.shape && expected.dtype == actual.dtype)
                #expect(MLX.arrayEqual(expected, actual).item(Bool.self), "phase=\(phase) max_abs=\(error)")
            }
            for (phase, expected, actual) in [("prefill", nativePrefillState, researchPrefillState),
                                               ("decode", nativeDecodeState, researchDecodeState)] {
                #expect(expected.count == 2 && actual.count == 2)
                guard expected.count == actual.count else { continue }
                for (left, right) in zip(expected, actual) {
                    #expect(left.shape == right.shape && left.dtype == right.dtype)
                    #expect(MLX.arrayEqual(left, right).item(Bool.self), "recurrent state phase=\(phase)")
                }
            }
            let nativeKV = nativeAttention.state
            let researchKV = researchAttention.state
            #expect(nativeKV.count == researchKV.count && nativeKV.count == 2)
            for (left, right) in zip(nativeKV, researchKV) {
                #expect(left.shape == right.shape && left.dtype == right.dtype)
                #expect(MLX.arrayEqual(left, right).item(Bool.self), "attention KV state")
            }

            // Drive the same hybrid model through the real fallible prefill
            // and iterator boundary, comparing sampled tokens to resident
            // execution over multiple autoregressive steps.
            let model = IteratorModel(language)
            let researchLimits = limits
            let input = LMInput(tokens: prompt)
            let referenceCaches: [KVCache?] = [MambaCache(), KVCacheSimple()]
            language.resetPositionState()
            var referenceLogits = language(prompt, cache: referenceCaches).logits
            var referenceTokens: [Int] = []
            for _ in 0 ..< 3 {
                let token = argMax(referenceLogits[0..., -1, 0...], axis: -1).item(Int.self)
                referenceTokens.append(token)
                referenceLogits = language(MLXArray([Int32(token)], [1, 1]), cache: referenceCaches).logits
            }
            language.resetPositionState()
            var discardCount = 0
            var iterator = try ExperimentalThrowingTokenIterator.experimentalTextPrefill(
                input: input, model: model, processor: nil, sampler: ArgMaxSampler(), maxTokens: 3,
                prepare: { input, _, _ in .tokens(input.text) },
                forward: { text, cache, _ in
                    try language.researchAffineForward(text.tokens, cache: cache?.map { $0 as KVCache? },
                        reducers: reducers, limits: researchLimits, groupLimit: 16, tokenWindow: 2)
                }, discardOnFailure: { discardCount += 1; language.resetPositionState() })
            for expected in referenceTokens {
                #expect(try iterator.nextToken() == expected)
            }
            #expect(try iterator.nextToken() == nil)
            #expect(discardCount == 0)

            // A partially completed prefill is discarded by the factory.
            language.resetPositionState()
            #expect(throws: FixtureFailure.prefill) {
                _ = try ExperimentalThrowingTokenIterator.experimentalTextPrefill(
                    input: input, model: model, processor: nil, sampler: ArgMaxSampler(),
                    prepare: { input, cache, _ in
                        _ = try language.researchAffineForward(input.text.tokens,
                            cache: cache.map { $0 as KVCache? }, reducers: reducers,
                            limits: researchLimits, groupLimit: 16, tokenWindow: 2)
                        throw FixtureFailure.prefill
                    }, forward: { _, _, _ in throw FixtureFailure.decode },
                    discardOnFailure: { discardCount += 1; language.resetPositionState() })
            }
            #expect(discardCount == 1)

            // Decode cleanup belongs to the caller: do not reuse the iterator
            // or its caches after a forward mutates them and then fails.
            language.resetPositionState()
            var failing = try ExperimentalThrowingTokenIterator.experimentalTextPrefill(
                input: input, model: model, processor: nil, sampler: ArgMaxSampler(),
                prepare: { input, _, _ in .tokens(input.text) },
                forward: { text, cache, _ in
                    let result = try language.researchAffineForward(text.tokens,
                        cache: cache?.map { $0 as KVCache? }, reducers: reducers,
                        limits: researchLimits, groupLimit: 16, tokenWindow: 2)
                    if text.tokens.size == 1 { throw FixtureFailure.decode }
                    return result
                }, discardOnFailure: { Issue.record("unexpected prefill discard") })
            #expect(throws: FixtureFailure.decode) { try failing.nextToken() }
            #expect(throws: ExperimentalDecodeError.invalidated) { try failing.nextToken() }
            try MLX.withError { StreamOrDevice.default.stream.synchronize() }
            language.resetPositionState()
        }
    }
}
