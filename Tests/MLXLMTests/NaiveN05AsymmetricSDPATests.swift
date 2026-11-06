import Foundation
import MLX
import XCTest
@testable import MLXLLM

final class NaiveN05AsymmetricSDPATests: XCTestCase {
    private func arrays(tokens: Int, group: Int, dtype: DType = .float32)
        -> (MLXArray, MLXArray, MLXArray) {
        (MLXArray.zeros([1, group, tokens, 192], dtype: dtype),
         MLXArray.zeros([1, 1, 17, 192], dtype: dtype),
         MLXArray.ones([1, 1, 17, 128], dtype: dtype))
    }

    /// Independent scalar oracle for F32 inputs. The explicit GPU reference's
    /// matmuls can use TF32 under the default process policy, so it is not an
    /// accuracy oracle for the full-precision vector SDPA output.
    private func cpuAttention(query: [Float], key: [Float], value: [Float],
                              allowed: [Bool], heads: Int, tokens: Int,
                              hasSink: Bool, valueScale: Float) -> [Float] {
        let keys = 17, width = 192, outputs = 128
        var result = [Float]()
        for batch in 0 ..< 2 {
            for head in 0 ..< heads {
                for token in 0 ..< tokens {
                    let selected = (0 ..< keys).filter { allowed[(batch * tokens + token) * keys + $0] }
                    if selected.isEmpty {
                        result += [Float](repeating: 0, count: outputs)
                        continue
                    }
                    let logits = selected.map { keyIndex -> Double in
                        var dot = 0.0
                        for d in 0 ..< width {
                            dot += Double(query[((batch * heads + head) * tokens + token) * width + d])
                                * Double(key[(batch * keys + keyIndex) * width + d])
                        }
                        return dot / sqrt(Double(width))
                    }
                    let maximum = max(logits.max()!, hasSink ? 0 : -Double.infinity)
                    let exponentials = logits.map { exp($0 - maximum) }
                    let denominator = exponentials.reduce(0, +) + (hasSink ? exp(-maximum) : 0)
                    for column in 0 ..< outputs {
                        var sum = 0.0
                        for (slot, keyIndex) in selected.enumerated() {
                            let scaled = value[(batch * keys + keyIndex) * outputs + column] * valueScale
                            sum += exponentials[slot] / denominator * Double(scaled)
                        }
                        result.append(Float(sum))
                    }
                }
            }
        }
        return result
    }

    func testNativeAsymmetricCutoffsKeepPrefillOnReference() throws {
        try MLXMetalTestLock.withLock {
            for (group, limit) in [(8, 4), (16, 2), (4, 8)] {
                for tokens in [1, limit, limit + 1] {
                    let (q, k, v) = arrays(tokens: tokens, group: group)
                    XCTAssertEqual(NaiveN05FlashMath.usesAsymmetricDecodeSDPA(query: q, key: k, value: v),
                                   tokens <= limit)
                    if tokens > limit {
                        let mask = MLXArray.ones([1, tokens, 17], dtype: .bool)
                        let result = NaiveN05FlashMath.attention(query: q, key: k, value: v,
                            allowed: mask, sink: nil, valueScale: nil)
                        let reference = NaiveN05FlashMath.referenceAttention(query: q, key: k, value: v,
                            allowed: mask, sink: nil, valueScale: nil)
                        XCTAssertEqual(result.asArray(Float.self), reference.asArray(Float.self))
                    }
                }
            }
        }
    }

    func testSinksScaleAndAllMaskedRowsAgainstAnalyticResult() throws {
        try MLXMetalTestLock.withLock {
            for dtype in [DType.float32, .float16, .bfloat16] {
                let (q, k, v) = arrays(tokens: 2, group: 16, dtype: dtype)
                // One valid query followed by one entirely padded query.
                let mask = MLXArray([Bool](repeating: true, count: 17)
                    + [Bool](repeating: false, count: 17)).reshaped(1, 2, 17)
                for withSink in [false, true] {
                    let sink = withSink ? MLXArray.zeros([16], dtype: .float32) : nil
                    let result = NaiveN05FlashMath.attention(query: q, key: k, value: v,
                        allowed: mask, sink: sink, valueScale: 0.5).asType(.float32).asArray(Float.self)
                    let expected: Float = withSink ? 0.5 * 17 / 18 : 0.5
                    // Fixed native output-rounding budget, specified before execution.
                    let accuracy: Float = dtype == .float32 ? 0.000002 : dtype == .float16 ? 0.00048828125 : 0.00390625
                    for head in 0 ..< 16 {
                        for column in 0 ..< 128 {
                            XCTAssertEqual(result[(head * 2) * 128 + column], expected, accuracy: accuracy)
                            XCTAssertEqual(result[(head * 2 + 1) * 128 + column], 0)
                        }
                    }
                }
            }
        }
    }

    func testNoncontiguousOffsetMasksMatchIndependentReference() throws {
        try MLXMetalTestLock.withLock {
            for dtype in [DType.float32, .float16, .bfloat16] {
                for group in [8, 16] {
                    let tokens = 32 / group
                    func values(_ shape: [Int], seed: Int) -> MLXArray {
                        let count = shape.reduce(1, *)
                        return MLXArray((0 ..< count).map { Float(($0 * 7 + seed) % 17 - 8) / 32 }, shape).asType(dtype)
                    }
                    // Transposed sequence/head storage exercises native stride handling.
                    let q = values([2, tokens, group, 192], seed: 3).transposed(0, 2, 1, 3)
                    let k = values([2, 17, 2, 192], seed: 7)[0..., 0..., ..<1, 0...].transposed(0, 2, 1, 3)
                    let v = values([2, 17, 2, 128], seed: 11)[0..., 0..., ..<1, 0...].transposed(0, 2, 1, 3)
                    let padding = MLXArray.ones([2, 29], dtype: .bool)
                    let mask = NaiveN05FlashMath.allowedMask(padding: padding,
                        queryOffset: 25 - tokens, length: tokens, keyOffset: 8, keyLength: 17, window: 7)
                    for withSink in [false, true] {
                        let sink = withSink ? MLXArray.zeros([group], dtype: .float32) : nil
                        let actual = NaiveN05FlashMath.attention(query: q, key: k, value: v,
                            allowed: mask, sink: sink, valueScale: 0.707)
                        let reference = NaiveN05FlashMath.referenceAttention(query: q, key: k, value: v,
                            allowed: mask, sink: sink, valueScale: 0.707)
                        let got = actual.asType(.float32).asArray(Float.self)
                        let gpuReference = reference.asType(.float32).asArray(Float.self)
                        let expected = dtype == .float32
                            ? cpuAttention(query: q.asArray(Float.self), key: k.asArray(Float.self),
                                           value: v.asArray(Float.self), allowed: mask.asArray(Bool.self),
                                           heads: group, tokens: tokens, hasSink: withSink, valueScale: 0.707)
                            : gpuReference
                        let accuracy: Float = dtype == .float32 ? 0.00001 : dtype == .float16 ? 0.001 : 0.008
                        XCTAssertEqual(actual.shape, [2, group, tokens, 128])
                        if dtype == .float32 {
                            let drift = zip(gpuReference, expected).map { abs($0.0 - $0.1) }.max() ?? 0
                            print("F32 reference vs CPU maximum error=\(drift) group=\(group) sink=\(withSink)")
                            // This second relation is exercised by an explicitly
                            // strict test process; no production setting changes.
                            if ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0" {
                                for index in got.indices {
                                    XCTAssertEqual(got[index], gpuReference[index], accuracy: accuracy)
                                }
                            }
                        }
                        for index in got.indices {
                            XCTAssertEqual(got[index], expected[index], accuracy: accuracy,
                                           "dtype=\(dtype) group=\(group) sink=\(withSink) index=\(index)")
                        }
                    }
                }
            }
        }
    }
}
