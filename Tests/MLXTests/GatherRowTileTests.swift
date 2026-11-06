// Copyright © 2026 Osaurus contributors.

import Foundation
import MLX
import XCTest

final class GatherRowTileTests: XCTestCase {
    override func setUp() { prepareMLXMetallibForTests() }

    func testSortedExpertBoundaries() {
        var cases = 0
        for dtype: DType in [.float32, .float16, .bfloat16] {
            for rows in [1, 7, 33, 65, 129, 261, 513] {
                for columns in [32, 67] {
                    let k = 128
                    let experts = 65
                    // Leave empty experts at both ends and between populated groups.
                    let ids = (0 ..< rows).map { i -> Int32 in
                        i < 3 ? 1 : (i < rows / 2 ? 32 : 63)
                    }.sorted()
                    let x = MLXArray(
                        (0 ..< rows * k).map {
                            Float(($0 * 7) % 13 - 6) / 16
                        }, [rows, 1, k]
                    ).asType(dtype)
                    let w = MLXArray(
                        (0 ..< experts * columns * k).map {
                            Float(($0 * 3) % 11 - 5) / 16
                        }, [experts, columns, k]
                    ).asType(dtype)
                    let indices = MLXArray(ids)
                    let reference = gatherMM(
                        x.asType(.float32), w.asType(.float32).swappedAxes(-1, -2),
                        rhsIndices: indices, stream: .cpu)
                    let actual = gatherMM(
                        x, w.swappedAxes(-1, -2), rhsIndices: indices, sortedIndices: true)
                    XCTAssertTrue(
                        allClose(
                            actual.asType(.float32), reference,
                            rtol: 0.01, atol: 0.005
                        ).item(Bool.self),
                        "float dtype=\(dtype) rows=\(rows) N=\(columns)")
                    for bits in [2, 3, 4, 6, 8] {
                        let (qw, scales, biases) = quantized(w, groupSize: 64, bits: bits)
                        let unpacked = dequantized(
                            qw, scales: scales, biases: biases,
                            groupSize: 64, bits: bits)
                        let expected = gatherMM(
                            x.asType(.float32), unpacked.asType(.float32).swappedAxes(-1, -2),
                            rhsIndices: indices, stream: .cpu)
                        let got = gatherQuantizedMM(
                            x, qw, scales: scales, biases: biases, rhsIndices: indices,
                            transpose: true, groupSize: 64, bits: bits, sortedIndices: true)
                        XCTAssertTrue(
                            allClose(
                                got.asType(.float32), expected,
                                rtol: 0.03, atol: 0.03
                            ).item(Bool.self),
                            "affine dtype=\(dtype) bits=\(bits) rows=\(rows) N=\(columns)")
                        cases += 1
                    }
                }
            }
        }
        print("GATHER_ROWS correctness_cases=\(cases)")
    }

    func testAffineFormatsAndGatherModes() {
        let experts = 9
        let rows = 97
        var cases = 0
        for group in [32, 64, 128] {
            for bits in [1, 2, 3, 4, 5, 6, 8] {
                for transpose in [false, true] {
                    // K=96 also exercises a partial 64-wide reduction tile.
                    let k = group == 32 ? 96 : 256
                    let n = transpose ? 67 : 128
                    let height = transpose ? n : k
                    let width = transpose ? k : n
                    let w = MLXArray(
                        (0 ..< experts * height * width).map {
                            Float(($0 * 3) % 11 - 5) / 16
                        }, [experts, height, width]
                    ).asType(.bfloat16)
                    let packed: MLXArray
                    let scales: MLXArray
                    let biases: MLXArray?
                    if bits == 1 {
                        // One-bit bundles are supported by matmul, but not quantize().
                        packed = MLXArray(
                            Array(
                                repeating: UInt32(0xa5a5_a5a5),
                                count: experts * height * width / 32),
                            [experts, height, width / 32])
                        scales = MLXArray.full(
                            [experts, height, width / group],
                            values: MLXArray(Float(0.125))
                        ).asType(.bfloat16)
                        biases = MLXArray.full(
                            [experts, height, width / group],
                            values: MLXArray(Float(-0.0625))
                        ).asType(.bfloat16)
                    } else {
                        (packed, scales, biases) = quantized(w, groupSize: group, bits: bits)
                    }
                    let unpacked = dequantized(
                        packed, scales: scales, biases: biases,
                        groupSize: group, bits: bits
                    ).asType(.float32)
                    for sorted in [false, true] {
                        let rawIDs = (0 ..< rows).map { Int32(($0 * 7) % (experts - 2) + 1) }
                        let ids = MLXArray(sorted ? rawIDs.sorted() : rawIDs)
                        for explicitLHS in [false, true] {
                            let x = MLXArray(
                                (0 ..< rows * k).map {
                                    Float(($0 * 7) % 13 - 6) / 16
                                }, [rows, 1, k]
                            ).asType(.bfloat16)
                            let lhs =
                                explicitLHS
                                ? MLXArray((0 ..< rows).map { Int32(rows - $0 - 1) }) : nil
                            let expected = gatherMM(
                                x.asType(.float32),
                                transpose ? unpacked.swappedAxes(-1, -2) : unpacked,
                                lhsIndices: lhs, rhsIndices: ids, stream: .cpu)
                            let actual = gatherQuantizedMM(
                                x, packed, scales: scales, biases: biases,
                                lhsIndices: lhs, rhsIndices: ids, transpose: transpose,
                                groupSize: group, bits: bits, sortedIndices: sorted)
                            XCTAssertTrue(
                                allClose(
                                    actual.asType(.float32), expected,
                                    rtol: 0.03, atol: 0.03
                                ).item(Bool.self),
                                "group=\(group) bits=\(bits) transpose=\(transpose) sorted=\(sorted) explicitLHS=\(explicitLHS)"
                            )
                            cases += 1
                        }
                    }
                }
            }
        }
        print("GATHER_ROWS format_mode_cases=\(cases)")
    }

    func testSharedNAXLoaderWithOrdinaryMatmul() {
        // The gathered tail fix also shares a loader with ordinary affine MM.
        for group in [32, 64, 128] {
            for bits in [2, 3, 4, 5, 6, 8] {
                let x = MLXArray(
                    (0 ..< 9472).map { (i: Int) -> Float in
                        Float((i * 7) % 13 - 6) / 16
                    }, [37, 256]
                ).asType(.bfloat16)
                let w = MLXArray(
                    (0 ..< 17152).map { (i: Int) -> Float in
                        Float((i * 3) % 11 - 5) / 16
                    }, [67, 256]
                ).asType(.bfloat16)
                let (packed, scales, biases) = quantized(w, groupSize: group, bits: bits)
                let unpacked = dequantized(
                    packed, scales: scales, biases: biases,
                    groupSize: group, bits: bits)
                let expected = matmul(
                    x.asType(.float32),
                    unpacked.asType(.float32).T, stream: .cpu)
                let actual = quantizedMM(
                    x, packed, scales: scales, biases: biases,
                    transpose: true, groupSize: group, bits: bits)
                XCTAssertTrue(
                    allClose(
                        actual.asType(.float32), expected,
                        rtol: 0.03, atol: 0.03
                    ).item(Bool.self),
                    "ordinary affine group=\(group) bits=\(bits)")
            }
        }
    }

    func testMeasuredSortedGather() throws {
        guard ProcessInfo.processInfo.environment["VMLX_GATHER_ROW_BENCH"] == "1" else {
            throw XCTSkip("Opt-in throughput diagnostic")
        }
        MLXRandom.seed(4567)
        // Representative routed projection sizes; no model weights loaded.
        for (experts, k, n) in [
            (128, 2816, 704), (128, 2688, 1856), (256, 2048, 512), (512, 2560, 640),
        ] {
            let weight = MLXRandom.normal([experts, n, k]).asType(.bfloat16)
            let (packed, scales, biases) = quantized(weight, groupSize: 64, bits: 4)
            eval(weight, packed, scales)
            for rows in [64, 512, 713, 2048, 4096, 4123] {
                let x = MLXRandom.normal([rows, 1, k]).asType(.bfloat16)
                let ids = MLXArray((0 ..< rows).map { Int32(($0 * 37) % experts) }.sorted())
                eval(x, ids)
                for quant in [false, true] {
                    var times: [Double] = []
                    for iteration in 0 ..< 20 {
                        let start = Date.timeIntervalSinceReferenceDate
                        let y =
                            quant
                            ? gatherQuantizedMM(
                                x, packed, scales: scales, biases: biases,
                                rhsIndices: ids, transpose: true, groupSize: 64,
                                bits: 4, sortedIndices: true)
                            : gatherMM(
                                x, weight.swappedAxes(-1, -2), rhsIndices: ids,
                                sortedIndices: true)
                        eval(y)
                        let elapsed = Date.timeIntervalSinceReferenceDate - start
                        if iteration >= 5 { times.append(elapsed * 1000) }
                    }
                    print(
                        "GATHER_ROWS E=\(experts) K=\(k) N=\(n) M=\(rows) quant=\(quant) median_ms=\(times.sorted()[7]) samples=\(times)"
                    )
                }
            }
        }
    }
}
