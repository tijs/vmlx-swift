// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

@Suite("qwen4_exp forward-local rotary factors", .serialized)
struct Qwen4ExpRotaryContextTests {
    private func bits(_ array: MLXArray) -> [UInt32] {
        array.asType(.float32).asArray(Float.self).map(\.bitPattern)
    }

    @Test("independent layers reuse exact factors across short and long positions")
    func exactFactors() throws {
        try MLXMetalTestLock.withLock {
            var cases = 0
            for dtype in [DType.float16, .bfloat16, .float32] {
                for dimension in [64, 128] {
                    for fast in [false, true] {
                        for (start, count, step) in [
                            (0, 1, 1), (31, 1, 1), (8339, 1, 1),
                            (34953, 1, 1), (131073, 1, 1), (8320, 4, 4),
                        ] {
                            let rotary = Qwen35Language.RotaryEmbedding(
                                dim: dimension, base: 1_000_000, mropeSection: [11, 11, 10],
                                textPositionFastPath: fast)
                            let otherLayer = Qwen35Language.RotaryEmbedding(
                                dim: dimension, base: 1_000_000, mropeSection: [11, 11, 10],
                                textPositionFastPath: fast)
                            let query = MLXArray.zeros([1, 16, count, 256], dtype: dtype)
                            let key = MLXArray.zeros([1, 2, count, 256], dtype: dtype)
                            let positions = MLXArray(stride(
                                from: start, to: start + count * step, by: step))
                                .asType(.int32).reshaped(1, count)
                            let expected = rotary(x: query, positionIds: positions)
                            let context = Qwen4ExpRotaryContext()
                            let first = context.factors(
                                rotary: rotary, like: query, start: start,
                                end: start + count * step, step: step)
                            // Mirrors the per-layer async submission: reuse after the
                            // first consumer has already materialized the factors.
                            MLX.eval(first.0, first.1)
                            let reused = context.factors(
                                rotary: otherLayer, like: key, start: start,
                                end: start + count * step, step: step)
                            #expect(context.factorCount == 1 && context.reuseCount == 1)
                            for (actual, reference) in [(first.0, expected.0), (first.1, expected.1),
                                (reused.0, expected.0), (reused.1, expected.1)]
                            {
                                #expect(actual.shape == [1, count, dimension])
                                #expect(actual.dtype == dtype)
                                #expect(bits(actual) == bits(reference))
                            }
                            cases += 1
                        }
                    }
                }
            }
            print("[Qwen4Rotary] exact_factor_cases=\(cases) cross_layer=1 evaluated_reuse=1")
        }
    }

    @Test("all factor inputs are keyed and a new forward has no retained entries")
    func keyAndLifetimeIsolation() throws {
        try MLXMetalTestLock.withLock {
            let context = Qwen4ExpRotaryContext()
            let x = MLXArray.zeros([1, 1, 1, 128], dtype: .bfloat16)
            let configurations: [(Int, Float, [Int], Bool, DType, Int, Int, Int)] = [
                (64, 10_000, [11, 11, 10], true, .bfloat16, 8339, 8340, 1),
                (128, 10_000, [11, 11, 10], true, .bfloat16, 8339, 8340, 1),
                (64, 1_000_000, [11, 11, 10], true, .bfloat16, 8339, 8340, 1),
                (64, 10_000, [10, 11, 11], true, .bfloat16, 8339, 8340, 1),
                (64, 10_000, [11, 11, 10], false, .bfloat16, 8339, 8340, 1),
                (64, 10_000, [11, 11, 10], true, .float16, 8339, 8340, 1),
                (64, 10_000, [11, 11, 10], true, .bfloat16, 8338, 8340, 1),
                (64, 10_000, [11, 11, 10], true, .bfloat16, 8339, 8341, 1),
                (64, 10_000, [11, 11, 10], true, .bfloat16, 8339, 8340, 4),
            ]
            for (dimension, base, sections, fast, dtype, start, end, step) in configurations {
                let rotary = Qwen35Language.RotaryEmbedding(
                    dim: dimension, base: base, mropeSection: sections, textPositionFastPath: fast)
                let typed = x.asType(dtype)
                let positions = MLXArray(stride(from: start, to: end, by: step))
                    .asType(.int32).reshaped(1, -1)
                let expected = rotary(x: typed, positionIds: positions)
                let actual = context.factors(
                    rotary: rotary, like: typed, start: start, end: end, step: step)
                #expect(bits(actual.0) == bits(expected.0))
                #expect(bits(actual.1) == bits(expected.1))
            }
            #expect(context.factorCount == configurations.count)
            #expect(context.reuseCount == 0)
            let nextForward = Qwen4ExpRotaryContext()
            #expect(nextForward.factorCount == 0 && nextForward.reuseCount == 0)
        }
    }

    @Test("shared AR eligibility excludes prefill, seed, verification, capture and outer trace")
    func eligibility() {
        #expect(Qwen4ExpEarlySubmission.allows(
            enabled: true, shape: [1, 1], autoregressive: true,
            recordingPrefix: false, externalPLE: false, compiledTrace: false))
        for shape in [[1, 2], [1, 512], [2, 1]] {
            #expect(!Qwen4ExpEarlySubmission.allows(
                enabled: true, shape: shape, autoregressive: true,
                recordingPrefix: false, externalPLE: false, compiledTrace: false))
        }
        for (enabled, ar, capture, external, trace) in [
            (false, true, false, false, false), (true, false, false, false, false),
            (true, true, true, false, false), (true, true, false, true, false),
            (true, true, false, false, true),
        ] {
            #expect(!Qwen4ExpEarlySubmission.allows(
                enabled: enabled, shape: [1, 1], autoregressive: ar,
                recordingPrefix: capture, externalPLE: external, compiledTrace: trace))
        }
    }
}
