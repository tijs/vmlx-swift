// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

@Suite("Flash AR GDN Q/K normalization", .serialized)
struct Qwen4ExpGDNQKNormTests {
    private func reference(_ input: MLXArray, heads: Int, width: Int) -> [MLXArray] {
        let q = input[.ellipsis, ..<(heads * width)].reshaped(1, 1, heads, width)
        let k = input[.ellipsis, (heads * width)..<(2 * heads * width)]
            .reshaped(1, 1, heads, width)
        let inv = pow(Float(width), -0.5)
        return [
            MLXArray(pow(inv, 2), dtype: q.dtype)
                * MLXFast.rmsNorm(q, weight: .mlxNone, eps: 1e-6),
            MLXArray(inv, dtype: k.dtype)
                * MLXFast.rmsNorm(k, weight: .mlxNone, eps: 1e-6),
        ]
    }

    private func bits(_ input: MLXArray) -> [UInt32] {
        input.asType(.float32).asArray(Float.self).map(\.bitPattern)
    }

    @Test("eligible strided half and BF16 rows preserve both intermediate roundings")
    func exactArithmetic() throws {
        try MLXMetalTestLock.withLock {
            var checks = 0
            for dtype in [DType.bfloat16, .float16] {
                for heads in [1, 3, 16] {
                    for width in [32, 64, 128, 256, 512, 1024, 4096] {
                        let kernel = Qwen4ExpGDNQKNorm(heads: heads, headDimension: width)
                        for stride in [1, 2] {
                            for magnitude: Float in [0, 0.0001, 1, 32] {
                                // Include an unused V tail and non-contiguous source views.
                                let coords = MLXArray(0..<(5 * heads * width * stride))
                                    .asType(.float32)
                                let input = (sin(coords * 0.193 + 0.37) * magnitude)
                                    .asType(dtype).reshaped(1, 1, -1)[
                                        .ellipsis, .stride(by: stride)]
                                let expected = concatenated(
                                    reference(input, heads: heads, width: width), axis: 2)
                                let actual = try #require(kernel(input))
                                MLX.eval(expected, actual)
                                #expect(actual.shape == expected.shape)
                                #expect(actual.dtype == expected.dtype)
                                #expect(bits(actual) == bits(expected),
                                    "dtype=\(dtype) heads=\(heads) width=\(width) stride=\(stride) magnitude=\(magnitude)")
                                checks += 1
                            }
                        }
                    }
                }
            }
            #expect(checks == 336)
            print("[GDNQKNorm] exact_cases=\(checks) token_per_second=NA reason=no_generation")
        }
    }

    @Test("unqualified dtype, shape, CPU and compiled tracing keep the original graph")
    func unsupportedFallback() throws {
        try MLXMetalTestLock.withLock {
            let kernel = Qwen4ExpGDNQKNorm(heads: 3, headDimension: 128)
            for shape in [[1, 2, 768], [2, 1, 768], [1, 1, 767], [768]] {
                #expect(kernel(MLXArray.zeros(shape, dtype: .bfloat16)) == nil)
            }
            #expect(kernel(MLXArray.zeros([1, 1, 768], dtype: .float32)) == nil)
            for width in [16, 96, 192, 8192] {
                #expect(!Qwen4ExpGDNQKNorm.eligible(
                    shape: [1, 1, 6 * width], dtype: .bfloat16,
                    heads: 3, headDimension: width))
            }
            #expect(!Qwen4ExpGDNQKNorm.eligible(
                shape: [1, 1, 768], dtype: .bfloat16, heads: 0, headDimension: 128))
            let input = MLXArray.zeros([1, 1, 768], dtype: .bfloat16)
            CompiledDecodeTrace.withActive { #expect(kernel(input) == nil) }
            Device.withDefaultDevice(.cpu) { #expect(kernel(input) == nil) }
        }
    }

    @Test("isolated timing excludes artificial concatenation from the control",
          .enabled(if: ProcessInfo.processInfo.environment["VMLX_GDN_QK_NORM_BENCH"] == "1"))
    func timing() throws {
        try MLXMetalTestLock.withLock {
            let heads = 16, width = 128
            let kernel = Qwen4ExpGDNQKNorm(heads: heads, headDimension: width)
            let input = sin(MLXArray(0..<(80 * width)).asType(.float32) * 0.193)
                .asType(.bfloat16).reshaped(1, 1, -1)
            func run(_ fused: Bool) throws {
                if fused { MLX.eval(try #require(kernel(input))) }
                else { MLX.eval(reference(input, heads: heads, width: width)) }
            }
            func measure(_ fused: Bool) throws -> Double {
                for _ in 0..<10 { try run(fused) }
                let start = ProcessInfo.processInfo.systemUptime
                for _ in 0..<200 { try run(fused) }
                return (ProcessInfo.processInfo.systemUptime - start) * 1000 / 200
            }
            for pair in 0..<4 {
                let fusedFirst = pair % 2 == 1
                let first = try measure(fusedFirst), second = try measure(!fusedFirst)
                print("[GDNQKNorm] pair=\(pair) original_ms=\(fusedFirst ? second : first) fused_ms=\(fusedFirst ? first : second) model_speed_claim=false")
            }
        }
    }
}
