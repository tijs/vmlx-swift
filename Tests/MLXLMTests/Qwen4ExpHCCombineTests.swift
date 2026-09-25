// Copyright © 2026 Apple Inc.

import MLX
import Testing

@testable import MLXVLM

@Suite("qwen4_exp exact HC residual combine", .serialized)
struct Qwen4ExpHCCombineTests {
    @Test("default HC combine retains strict explicit opt-out semantics")
    func defaultPolicy() {
        #expect(Qwen4ExpHCCombine.parse(nil))
        #expect(Qwen4ExpHCCombine.parse("1"))
        for value in ["", "0", "true", "2", "garbage"] {
            #expect(!Qwen4ExpHCCombine.parse(value))
        }
    }

    private func reference(_ residual: MLXArray, _ block: MLXArray, _ injection: MLXArray) -> MLXArray {
        let product = expandedDimensions(block, axis: -2) * expandedDimensions(injection, axis: -1)
        MLX.eval(product)
        return (residual + product.reshaped(residual.shape)).asType(residual.dtype)
    }

    private func bits(_ array: MLXArray) -> [UInt32] {
        array.asType(.float32).asArray(Float.self).map(\.bitPattern)
    }

    @Test("random and strided activations match separate multiply and add bit-for-bit")
    func exactParity() throws {
        try MLXMetalTestLock.withLock {
            for dtype in [DType.float16, .bfloat16, .float32] {
                for streams in [1, 2, 4] {
                    for hidden in [31, 128, 2560] {
                        for strided in [false, true] {
                            MLXRandom.seed(73)
                            let stride = strided ? 2 : 1
                            let residual = MLXRandom.normal([1, 1, streams * hidden * stride])
                                .asType(dtype)[.ellipsis, .stride(by: stride)]
                            let block = MLXRandom.normal([1, 1, hidden * stride])
                                .asType(dtype)[.ellipsis, .stride(by: stride)]
                            let injection = MLXRandom.normal([1, 1, streams * stride])
                                .asType(dtype)[.ellipsis, .stride(by: stride)]
                            let expected = reference(residual, block, injection)
                            let result = try #require(Qwen4ExpHCCombine.call(
                                residual: residual, block: block, injection: injection, enabled: true))
                            #expect(result.shape == residual.shape)
                            #expect(result.dtype == dtype)
                            #expect(bits(result) == bits(expected),
                                "dtype=\(dtype) streams=\(streams) hidden=\(hidden) strided=\(strided)")
                        }
                    }
                }
            }
        }
    }

    @Test("intermediate rounding is preserved instead of using an FMA")
    func productRoundingBoundary() throws {
        try MLXMetalTestLock.withLock {
            for (dtype, epsilon) in [(DType.float16, Float(0.0009765625)),
                (.bfloat16, Float(0.0078125)), (.float32, Float(0.00000011920928955078125))]
            {
                let value = 1 + epsilon
                let residual = MLXArray([-Float(1 + 2 * epsilon)]).reshaped(1, 1, 1).asType(dtype)
                let block = MLXArray([value]).reshaped(1, 1, 1).asType(dtype)
                let injection = MLXArray([value]).reshaped(1, 1, 1).asType(dtype)
                let expected = reference(residual, block, injection)
                let result = try #require(Qwen4ExpHCCombine.call(
                    residual: residual, block: block, injection: injection, enabled: true))
                #expect(bits(expected) == [Float(0).bitPattern])
                #expect(bits(result) == bits(expected))
                #expect((-Float(1 + 2 * epsilon)).addingProduct(value, value) != 0)
            }
        }
    }

    @Test("prefill, verification, batches, mixed dtypes and opt-out retain the original graph")
    func fallbackBoundaries() throws {
        try MLXMetalTestLock.withLock {
            for (batch, sequence) in [(1, 2), (1, 64), (2, 1)] {
                #expect(Qwen4ExpHCCombine.call(
                    residual: MLXArray.zeros([batch, sequence, 64]),
                    block: MLXArray.zeros([batch, sequence, 32]),
                    injection: MLXArray.zeros([batch, sequence, 2]), enabled: true) == nil)
            }
            let residual = MLXArray.zeros([1, 1, 64], dtype: .bfloat16)
            let block = MLXArray.zeros([1, 1, 32], dtype: .bfloat16)
            let injection = MLXArray.zeros([1, 1, 2], dtype: .bfloat16)
            #expect(Qwen4ExpHCCombine.call(residual: residual, block: block,
                injection: injection, enabled: false) == nil)
            #expect(Qwen4ExpHCCombine.call(residual: residual, block: block.asType(.float16),
                injection: injection, enabled: true) == nil)
            #expect(Qwen4ExpHCCombine.call(residual: residual[.ellipsis, ..<63], block: block,
                injection: injection, enabled: true) == nil)
            #expect(Qwen4ExpHCCombine.call(residual: residual.reshaped(1, 64), block: block,
                injection: injection, enabled: true) == nil)
        }
    }
}
