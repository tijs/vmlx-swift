// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

@Suite("Flash AR HC residual and next normalization", .serialized)
struct Qwen4ExpHCCombineNormTests {
    private func bits(_ value: MLXArray) -> [UInt32] {
        value.asType(.float32).asArray(Float.self).map(\.bitPattern)
    }

    @Test("both outputs preserve MLX rounding, group reduction, and source ownership")
    func exactArithmetic() throws {
        try MLXMetalTestLock.withLock {
            var checks = 0
            for dtype in [DType.bfloat16, .float16] {
                for streams in [1, 2, 4] {
                    for width in [32, 64, 96, 128, 192, 768, 2560, 4096] {
                        let kernel = Qwen4ExpHCCombineNorm(hiddenSize: width, eps: 1e-6)
                        for stride in [1, 2] {
                            for magnitude: Float in [0, 0.0001, 1, 8] {
                                for blockType in [dtype, .float32] {
                                    let coords = MLXArray(0 ..< (streams * width * stride)).asType(
                                        .float32)
                                    let residual = (sin(coords * 0.173 + 0.27) * magnitude)
                                        .asType(dtype).reshaped(1, 1, -1)[
                                            .ellipsis, .stride(by: stride)]
                                    let block =
                                        (cos(coords[0 ..< (width * stride)] * 0.071) * magnitude)
                                        .asType(blockType).reshaped(1, 1, -1)[
                                            .ellipsis, .stride(by: stride)]
                                    let injection =
                                        (sin(coords[0 ..< (streams * stride)] * 0.37) + 1)
                                        .asType(dtype).reshaped(1, 1, -1)[
                                            .ellipsis, .stride(by: stride)]
                                    let weight = (cos(coords * 0.097) + 1.3)
                                        .asType(dtype)[.stride(by: stride)]
                                    let original = [residual, block, injection, weight].map(bits)
                                    let product =
                                        expandedDimensions(block, axis: -2)
                                        * expandedDimensions(injection, axis: -1)
                                    let expected = (residual + product.reshaped(residual.shape))
                                        .asType(dtype)
                                    let expectedNorm =
                                        MLXFast.rmsNorm(
                                            expected.reshaped(1, 1, streams, width),
                                            weight: .mlxNone, eps: 1e-6
                                        )
                                        .reshaped(expected.shape) * weight
                                    let actual = try #require(
                                        kernel(
                                            residual: residual, block: block, injection: injection,
                                            weight: weight))
                                    MLX.eval(
                                        expected, expectedNorm, actual.residual, actual.normalized)
                                    let residualMatches = bits(actual.residual) == bits(expected)
                                    let normMatches = bits(actual.normalized) == bits(expectedNorm)
                                    #expect(
                                        residualMatches,
                                        "dtype=\(dtype) block=\(blockType) streams=\(streams) width=\(width) stride=\(stride) magnitude=\(magnitude)"
                                    )
                                    #expect(
                                        normMatches,
                                        "dtype=\(dtype) block=\(blockType) streams=\(streams) width=\(width) stride=\(stride) magnitude=\(magnitude)"
                                    )
                                    #expect(actual.residual.shape == residual.shape)
                                    #expect(actual.normalized.dtype == dtype)
                                    #expect(
                                        [residual, block, injection, weight].map(bits) == original)
                                    checks += 1
                                }
                            }
                        }
                    }
                }
            }
            #expect(checks == 768)
            print("[HCCombineNorm] exact_cases=\(checks) tokens_per_second=NA reason=no_generation")
        }
    }

    @Test("mean and epsilon preserve loaded RMS rounding at narrow cast boundaries")
    func meanEpsilonRoundingBoundaries() throws {
        try MLXMetalTestLock.withLock {
            // Generated, reproducible inputs only: no model weights, installed
            // bundle, private prompt, or sampler. With AOT fast RMS the old
            // separately rounded reciprocal mean fails 11 of these cases.
            var state: UInt64 = 0x9103_401
            func next() -> Float {
                state ^= state << 13
                state ^= state >> 7
                state ^= state << 17
                return Float(Int32(truncatingIfNeeded: state)) / Float(Int32.max)
            }
            for ordinal in 0 ..< 1024 {
                let width = [96, 192, 768, 2560][ordinal % 4]
                let dtype: DType = (ordinal / 4) % 2 == 0 ? .bfloat16 : .float16
                let magnitude: Float = [0.0001, 0.01, 0.1, 1, 8, 64][(ordinal / 8) % 6]
                let values = (0 ..< (width * 4)).map { _ in next() * magnitude }
                let residual = MLXArray(values).asType(dtype).reshaped(1, 1, -1)
                let block = MLXArray.zeros([1, 1, width], dtype: dtype)
                let injection = MLXArray.ones([1, 1, 4], dtype: dtype)
                let weight = MLXArray.ones([width * 4], dtype: dtype)
                // Preserve the COMPLETE ordinary graph, even for zero block
                // and unit weight: do not assume tiny FP16 values survive an
                // omitted arithmetic operation identically under fast math.
                let product =
                    expandedDimensions(block, axis: -2)
                    * expandedDimensions(injection, axis: -1)
                let expected = (residual + product.reshaped(residual.shape)).asType(dtype)
                let normalized =
                    MLXFast.rmsNorm(
                        expected.reshaped(4, width), weight: .mlxNone, eps: 1e-6
                    ).reshaped(residual.shape) * weight
                let actual = try #require(
                    Qwen4ExpHCCombineNorm(hiddenSize: width, eps: 1e-6)(
                        residual: residual, block: block, injection: injection, weight: weight))
                #expect(bits(actual.residual) == bits(expected), "ordinal=\(ordinal)")
                #expect(bits(actual.normalized) == bits(normalized), "ordinal=\(ordinal)")
            }
            print(
                "[HCCombineNorm] mean_epsilon_boundary_cases=1024 tokens_per_second=NA reason=no_generation"
            )
        }
    }

    @Test("unsupported widths, dtypes, prefill, batch and tracing fall back")
    func fallback() throws {
        try MLXMetalTestLock.withLock {
            for (batch, rows, width, dtype) in [
                (1, 2, 128, DType.bfloat16), (2, 1, 128, .bfloat16),
                (1, 1, 31, .bfloat16), (1, 1, 8192, .bfloat16), (1, 1, 128, .float32),
            ] {
                let kernel = Qwen4ExpHCCombineNorm(hiddenSize: width, eps: 1e-6)
                #expect(
                    kernel(
                        residual: MLXArray.zeros([batch, rows, 4 * width], dtype: dtype),
                        block: MLXArray.zeros([batch, rows, width], dtype: dtype),
                        injection: MLXArray.zeros([batch, rows, 4], dtype: dtype),
                        weight: MLXArray.ones([4 * width], dtype: dtype)) == nil)
            }
            let kernel = Qwen4ExpHCCombineNorm(hiddenSize: 128, eps: 1e-6)
            let residual = MLXArray.zeros([1, 1, 512], dtype: .bfloat16)
            let block = MLXArray.zeros([1, 1, 128], dtype: .bfloat16)
            let injection = MLXArray.ones([1, 1, 4], dtype: .bfloat16)
            let weight = MLXArray.ones([512], dtype: .bfloat16)
            #expect(
                kernel(
                    residual: residual, block: block, injection: injection,
                    weight: weight.asType(.float16)) == nil)
            CompiledDecodeTrace.withActive {
                #expect(
                    kernel(residual: residual, block: block, injection: injection, weight: weight)
                        == nil)
            }
            Device.withDefaultDevice(.cpu) {
                #expect(
                    kernel(residual: residual, block: block, injection: injection, weight: weight)
                        == nil)
            }
        }
    }
}
