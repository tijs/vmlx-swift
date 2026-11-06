// Copyright © 2026 Osaurus contributors.

import MLX
import XCTest

@testable import MLXVLM

final class Qwen35CompiledMoEBatchTests: XCTestCase {
    func testPostRegionRetainsDynamicBatchWidth() throws {
        try MLXMetalTestLock.withLock {
            // Match a live batch shrinking when a request reaches EOS, then
            // growing as new requests arrive. Reuse the same compiled region.
            for batch in [2, 1, 4, 3, 1, 2] {
                let routed = MLXArray((0 ..< batch * 8 * 32).map {
                    Float($0 % 5 - 2) / 4
                }, [batch, 1, 8, 32]).asType(.bfloat16)
                let scores = MLXArray((0 ..< batch * 8).map {
                    Float($0 % 3) / 8
                }, [batch, 1, 8]).asType(.bfloat16)
                let shared = MLXArray((0 ..< batch * 32).map {
                    Float($0 % 7 - 3) / 8
                }, [batch, 1, 32]).asType(.bfloat16)
                let expected = (routed * expandedDimensions(scores, axis: -1))
                    .sum(axis: -2) + shared
                let actual = try XCTUnwrap(Qwen4ExpCompiledMoE.post(
                    routed: routed, scores: scores, shared: shared))
                XCTAssertEqual(actual.shape, [batch, 1, 32])
                XCTAssertEqual(actual.dtype, .bfloat16)
                XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self),
                               "batch width \(batch)")
            }
        }
    }
}
