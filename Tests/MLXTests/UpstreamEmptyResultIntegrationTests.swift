// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Cmlx
import Foundation
import XCTest

@testable import MLX

final class UpstreamEmptyResultIntegrationTests: XCTestCase {
    func testEmptyVectorPadsWithInvalidArrays() {
        let vector = mlx_vector_array_new()
        defer { mlx_vector_array_free(vector) }
        XCTAssertTrue(mlx_vector_array_values(vector, count: 0).isEmpty)
        let padded = mlx_vector_array_values(vector, count: 3)
        XCTAssertEqual(padded.count, 3)
        XCTAssertTrue(padded.allSatisfy { !$0.isValid })
    }

    func testFailedQuantizationThrowsInsteadOfIndexingEmptyResults() throws {
        try Device.withDefaultDevice(.cpu) {
            XCTAssertThrowsError(try withError {
                _ = quantized(MLXArray.zeros([3, 3]), groupSize: 64, bits: 4)
            })
        }
    }

    func testFailedSVDThrowsInsteadOfIndexingEmptyResults() throws {
        try Device.withDefaultDevice(.cpu) {
            XCTAssertThrowsError(try withError {
                _ = MLXLinalg.svd(MLXArray(Float(1)))
            })
        }
    }

    func testFailedIndexBroadcastInvalidatesAssignmentDestination() throws {
        try Device.withDefaultDevice(.cpu) {
            let destination = MLXArray.zeros([3, 3])
            let rows = MLXArray([0, 1])
            let columns = MLXArray([0, 1, 2])
            XCTAssertThrowsError(try withError {
                destination[rows, columns] = MLXArray(Float(0))
            })
            XCTAssertFalse(destination.isValid)
        }
    }

    func testCompilePolicyBypassAlsoPadsShortResults() throws {
        let environment = ProcessInfo.processInfo.environment
        let enabled = environment["VMLX_ENABLE_UNSAFE_COMPILE"]
            ?? environment["MLXPRESS_ENABLE_UNSAFE_COMPILE"]
            ?? environment["MLX_ENABLE_UNSAFE_COMPILE"] ?? ""
        try XCTSkipIf(
            environment["MLX_DISABLE_COMPILE"] == nil
                && (enabled == "1" || enabled.lowercased() == "true"),
            "This regression specifically exercises the disabled compile-policy path")
        let state = CompiledFunction(inputs: [], outputs: [], shapeless: false) { _ in [] }
        let result = state.call([], expected: 2)
        XCTAssertEqual(result.count, 2)
        XCTAssertTrue(result.allSatisfy { !$0.isValid })
    }
}
