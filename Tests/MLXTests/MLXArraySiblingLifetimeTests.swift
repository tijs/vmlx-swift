// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import XCTest
@testable import MLX

// These assertions measure process-global allocator bytes. Run this class in
// its own serial filtered process; Swift Testing suite serialization cannot
// exclude allocations from other suites. Keep it outside the migrated suite.
final class MLXArraySiblingLifetimeTests: XCTestCase {
    private func activeBytesLeftBySiblings(
        file: StaticString = #filePath, line: UInt = #line,
        retire: (MLXArray, MLXArray, MLXArray?) -> Void
    ) -> Int {
        let bytes = 1024 * 1024 * MemoryLayout<Float>.size
        // Earlier tests in the same process may still have GPU work in flight, whose
        // buffers would be freed while this one measures.
        Stream.gpu.synchronize()
        let before = Memory.activeMemory
        do {
            let outputs: (MLXArray, MLXArray, MLXArray?)
            do {
                let w = MLXArray([Float](repeating: 1, count: 1024 * 1024), [1024, 1024])
                // premise: the input's buffer is counted as active
                XCTAssertGreaterThanOrEqual(
                    Memory.activeMemory - before, bytes / 2, file: file, line: line)
                outputs = quantized(w, groupSize: 64, bits: 4)
            }
            // `w`'s handle is gone, but the outputs' graph still holds its buffer; without
            // this the measurement below could not fail
            XCTAssertGreaterThanOrEqual(
                Memory.activeMemory - before, bytes, file: file, line: line)
            retire(outputs.0, outputs.1, outputs.2)
        }
        return Memory.activeMemory - before
    }

    func testDroppedSiblingOutputsFreeTheirGraph() {
        // control: the outputs are dropped, so `~array` breaks the sibling cycle
        let left = activeBytesLeftBySiblings { _, _, _ in }
        XCTAssertLessThan(left, 1024 * 1024)
    }

    func testUpdateInternalOnSiblingOutputsFreesTheirGraph() {
        // `Module.update(parameters:)` replaces each output of a freshly built
        // `QuantizedLinear` like this; the last replacement must break the cycle
        let left = activeBytesLeftBySiblings { wq, scales, biases in
            let replacement = MLXArray(Float(0))
            wq._updateInternal(replacement)
            scales._updateInternal(replacement)
            biases?._updateInternal(replacement)
        }
        XCTAssertLessThan(left, 1024 * 1024)
    }
}
