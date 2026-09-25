// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLX
import MLXLMCommon
import MLXRandom
import Testing

@testable import MLXVLM

@Suite("Qwen dense router preserves loaded weight precision", .serialized)
struct Qwen4ExpDenseRouterTests {
    private func sample(_ shape: [Int], seed: UInt64, magnitude: Float, dtype: DType) -> MLXArray {
        (MLXRandom.uniform(low: -1, high: 1, shape, key: MLXRandom.key(seed)) * magnitude)
            .asType(dtype)
    }

    private func reference(_ x: MLXArray, _ weight: MLXArray, k: Int, normalize: Bool)
        -> (MLXArray, MLXArray)
    {
        // Match SparseMoeBlock's ordinary graph, including precise softmax
        // and normalization after selection. No sampler or quantization edits.
        let gates = softmax(matmul(x, weight.transposed()), axis: -1, precise: true)
        let kth = weight.dim(0) - k
        let indices = argPartition(gates, kth: kth, axis: -1)[.ellipsis, kth...]
        let scores = takeAlong(gates, indices, axis: -1)
        return (indices, normalize ? scores / scores.sum(axis: -1, keepDims: true) : scores)
    }

    private func bits(_ x: MLXArray) -> [UInt32] {
        x.asType(.float32).asArray(Float.self).map(\.bitPattern)
    }

    @Test("F32 loaded router and BF16 router retain exact IDs, scores, and source tensors")
    func loadedDTypeAndQueuedWeights() throws {
        try MLXMetalTestLock.withLock {
            var cases = 0
            for dtype in [DType.float32, .bfloat16] {
                for magnitude: Float in [0.001, 0.1, 1, 16] {
                    for normalize in [false, true] {
                        for k in [1, 10, 512] {
                            var pending: [(MLXArray, MLXArray, MLXArray, MLXArray)] = []
                            var sources: [([MLXArray], [[UInt32]])] = []
                            // Multiple layers share one compiled region. Queue
                            // fresh weights before evaluation; sequential eval
                            // cannot detect stale captured-weight reuse.
                            for layer in 0 ..< 3 {
                                let x = sample(
                                    [1, 1, 2560], seed: UInt64(800 + layer),
                                    magnitude: magnitude, dtype: .bfloat16)
                                let w = sample(
                                    [512, 2560], seed: UInt64(900 + layer),
                                    magnitude: 0.1, dtype: dtype)
                                let expected = reference(x, w, k: k, normalize: normalize)
                                let actual = try #require(
                                    Qwen4ExpCompiledMoE.denseRouter(
                                        x, weight: w, topK: k, normTopK: normalize))
                                sources.append(([x, w], [bits(x), bits(w)]))
                                pending.append(
                                    (expected.0, expected.1, actual.indices, actual.scores))
                            }
                            MLX.eval(pending.flatMap { [$0.0, $0.1, $0.2, $0.3] })
                            for (expectedIDs, expectedScores, ids, scores) in pending {
                                #expect(ids.dtype == expectedIDs.dtype)
                                #expect(ids.shape == expectedIDs.shape)
                                #expect(ids.asArray(Int32.self) == expectedIDs.asArray(Int32.self))
                                #expect(scores.dtype == expectedScores.dtype)
                                #expect(scores.shape == expectedScores.shape)
                                #expect(bits(scores) == bits(expectedScores))
                                #expect(isFinite(scores).all().item(Bool.self))
                                cases += 1
                            }
                            for (arrays, originals) in sources {
                                #expect(arrays.map(bits) == originals)
                            }
                        }
                    }
                }
            }
            #expect(cases == 144)
            print("[DenseRouter] exact_cases=\(cases) generation_tps=NA reason=no_generation")
        }
    }

    @Test("ties, strided inputs, batch shape changes, and non-square dimensions remain exact")
    func layoutsAndTies() throws {
        try MLXMetalTestLock.withLock {
            for batch in [1, 2] {
                for width in [64, 192, 2560] {
                    let x = sample(
                        [batch, 1, width * 2], seed: 311,
                        magnitude: 1, dtype: .bfloat16)[.ellipsis, .stride(by: 2)]
                    for zero in [false, true] {
                        let w =
                            zero
                            ? MLXArray.zeros([64, width], dtype: .float32)
                            : sample([64, width * 2], seed: 512, magnitude: 1, dtype: .float32)[
                                .ellipsis, .stride(by: 2)]
                        let expected = reference(x, w, k: 10, normalize: true)
                        let actual = try #require(
                            Qwen4ExpCompiledMoE.denseRouter(
                                x, weight: w, topK: 10, normTopK: true))
                        MLX.eval(expected.0, expected.1, actual.indices, actual.scores)
                        #expect(
                            actual.indices.asArray(Int32.self) == expected.0.asArray(Int32.self))
                        #expect(bits(actual.scores) == bits(expected.1))
                    }
                }
            }
        }
    }

    @Test("prefill, verification, outer traces, and malformed signatures use the ordinary path")
    func eligibility() throws {
        try MLXMetalTestLock.withLock {
            let weight = MLXArray.zeros([16, 64], dtype: .float32)
            for shape in [[64], [1, 64], [1, 2, 64], [1, 3, 64], [1, 4, 64], [1, 1, 32]] {
                #expect(
                    Qwen4ExpCompiledMoE.denseRouter(
                        .zeros(shape, dtype: .bfloat16), weight: weight, topK: 4, normTopK: true)
                        == nil)
            }
            let x = MLXArray.zeros([1, 1, 64], dtype: .bfloat16)
            for k in [0, -1, 17] {
                #expect(
                    Qwen4ExpCompiledMoE.denseRouter(x, weight: weight, topK: k, normTopK: true)
                        == nil)
            }
            #expect(
                Qwen4ExpCompiledMoE.denseRouter(
                    x.asType(.float32), weight: weight, topK: 4, normTopK: true) == nil)
            #expect(
                Qwen4ExpCompiledMoE.denseRouter(
                    x, weight: weight.asType(.float16), topK: 4, normTopK: true) == nil)
            #expect(
                Qwen4ExpCompiledMoE.denseRouter(
                    x, weight: weight.reshaped(-1), topK: 4, normTopK: true) == nil)
            CompiledDecodeTrace.withActive {
                #expect(
                    Qwen4ExpCompiledMoE.denseRouter(x, weight: weight, topK: 4, normTopK: true)
                        == nil)
            }
        }
    }
}
