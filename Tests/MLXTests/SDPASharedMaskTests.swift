// Copyright © 2026 Osaurus contributors.

import Foundation
import MLX
import XCTest

final class SDPASharedMaskTests: XCTestCase {
    private func mask(length: Int, variant: Int) -> MLXArray {
        let blocks = length / 4
        let selected =
            variant == 0
            ? Set((blocks - 512) ..< blocks)
            : Set((0 ..< 512).map { ($0 * 7919) % blocks })
        let bits = (0 ..< length).map {
            variant != 3 && (variant == 2 || selected.contains($0 / 4) || $0 / 4 == blocks)
        }
        return MLXArray(bits).reshaped(1, 1, 1, length)
    }

    private func assertOriginalPartitionOrder(
        queries: MLXArray, keys: MLXArray, values: MLXArray, mask: MLXArray,
        sinks: MLXArray? = nil, label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        // Materialize identical per-head mask rows. Their nonzero head stride
        // forces the original traversal in the SAME kernel and math mode.
        // The shared mask can use packed traversal; do not compare against a
        // gathered K/V tensor, which would change the split/reduction order.
        let shape = [queries.dim(0), queries.dim(1), queries.dim(2), keys.dim(2)]
        let originalMask = logicalOr(
            broadcast(mask, to: shape), MLXArray.zeros(shape, dtype: .bool))
        eval(originalMask)
        XCTAssertGreaterThan(originalMask.strides[1], 0, file: file, line: line)
        let scale = 1 / sqrt(Float(queries.dim(3)))
        let original = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale,
            mask: .array(originalMask), sinks: sinks)
        let shared = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale,
            mask: .array(mask), sinks: sinks)
        eval(original, shared)
        XCTAssertEqual(original.dtype, shared.dtype, label, file: file, line: line)
        XCTAssertEqual(original.shape, shared.shape, label, file: file, line: line)
        // Bit patterns also compare fully masked rows, including their existing
        // NaN convention, without weakening the test to a numerical tolerance.
        XCTAssertEqual(
            original.asType(.float32).asArray(Float.self).map(\.bitPattern),
            shared.asType(.float32).asArray(Float.self).map(\.bitPattern),
            label, file: file, line: line)
    }

    func testSharedMaskRetainsExactPartitionOrder() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Metal shared-mask traversal requires a GPU")
        }
        var cases = 0
        for dtype in [DType.bfloat16, .float16, .float32] {
            for (heads, kvHeads, dim) in [(24, 2, 256), (8, 2, 128), (4, 2, 256)] {
                // Include both sides of admission, non-aligned tails, and spans
                // exceeding the packed-word bound on smaller split counts.
                for length in [8193, 32768, 32769, 34939, 131075] {
                    MLXRandom.seed(63)
                    let queries = MLXRandom.normal([1, heads, 1, dim]).asType(dtype)
                    let keys = MLXRandom.normal([1, kvHeads, length + 256, dim])
                        .asType(dtype)[.ellipsis, ..<length, 0...]
                    let values = MLXRandom.normal([1, kvHeads, length + 256, dim])
                        .asType(dtype)[.ellipsis, ..<length, 0...]
                    eval(queries, keys, values)
                    for variant in 0 ..< 4 {
                        assertOriginalPartitionOrder(
                            queries: queries, keys: keys, values: values,
                            mask: mask(length: length, variant: variant),
                            label: "\(dtype) H\(heads) D\(dim) N\(length) mask\(variant)")
                        cases += 1
                    }
                    Stream.gpu.synchronize()
                    Memory.clearCache()
                }
            }
        }
        print("SDPA_SHARED_MASK exact_partition_cases=\(cases)")
    }

    func testSharedMaskBatchSinksAndFallbackLayouts() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Metal shared-mask traversal requires a GPU")
        }
        for dtype in [DType.bfloat16, .float16, .float32] {
            for kind in ["batch", "rows", "strided", "sinks", "queryTranspose"] {
                MLXRandom.seed(83)
                let length = 34939
                let batch = kind == "batch" ? 2 : 1
                let rows = kind == "rows" || kind == "queryTranspose" ? 4 : 1
                let queries =
                    kind == "queryTranspose"
                    ? MLXRandom.normal([batch, rows, 8, 128]).asType(dtype)
                        .transposed(0, 2, 1, 3)
                    : MLXRandom.normal([batch, 8, rows, 128]).asType(dtype)
                let keys = MLXRandom.normal([batch, 2, length, 128]).asType(dtype)
                let values = MLXRandom.normal([batch, 2, length, 128]).asType(dtype)
                var sharedMask = mask(length: length, variant: 1)
                if kind == "strided" {
                    let bits = sharedMask.asArray(Bool.self).flatMap { [$0, false] }
                    sharedMask = MLXArray(bits)[.stride(by: 2)].reshaped(1, 1, 1, length)
                    eval(sharedMask)
                    XCTAssertEqual(sharedMask.strides[3], 2)
                }
                let sinks = kind == "sinks" ? MLXRandom.normal([8]).asType(dtype) : nil
                assertOriginalPartitionOrder(
                    queries: queries, keys: keys, values: values, mask: sharedMask,
                    sinks: sinks, label: "\(dtype) \(kind)")
                Stream.gpu.synchronize()
                Memory.clearCache()
            }
        }
        print("SDPA_SHARED_MASK batch_sinks_fallback_cases=15")
    }

    func testGeneratedHeaderMatchesPinnedCore() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let core = root.appendingPathComponent(
            "Source/Cmlx/mlx/mlx/backend/metal/kernels/sdpa_vector.h")
        let generated = root.appendingPathComponent("Source/Cmlx/mlx-generated/metal/sdpa_vector.h")
        XCTAssertEqual(try Data(contentsOf: core), try Data(contentsOf: generated))
    }
}
