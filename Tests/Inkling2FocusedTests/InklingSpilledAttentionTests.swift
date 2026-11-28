#if canImport(Darwin)
import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class InklingSpilledAttentionTests: XCTestCase {
    func testPagedVectorMatchesNativeMaskedDecodeOnM1Max() throws {
        try XCTSkipUnless(GPU.deviceInfo().architecture == "applegpu_g13s")
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLX_SDPA_BLOCKS"] == nil)
        try Device.withDefaultDevice(.gpu) {
            try withError { error in
                for count in [4096, 8192, 8213, 32769] {
                    let dtype = DType.bfloat16
                    let query = sin(arange(24 * 128, dtype: .float32) * Float(0.017))
                        .reshaped([1, 24, 1, 128]).asType(dtype)
                    let positions = arange(8 * count * 128, dtype: .float32)
                    let keys = cos(positions * Float(0.003) + Float(0.2))
                        .reshaped([1, 8, count, 128]).asType(dtype)
                    let values = sin(positions * Float(0.0041))
                        .reshaped([1, 8, count, 128]).asType(dtype)
                    let bias = (sin(arange(24 * 1024, dtype: .float32) * Float(0.013)) * Float(0.4))
                        .reshaped([1, 24, 1, 1024]).asType(dtype)
                    let sinks = (cos(arange(24, dtype: .float32)) * Float(0.8))
                        .reshaped([1, 24, 1, 1]).asType(dtype)
                    let zero = MLXArray.zeros([1, 8, 1, 128], dtype: dtype)
                    let mask = concatenated([
                        MLXArray.zeros([1, 24, 1, count - 1024], dtype: dtype),
                        take(bias, arange(1023, -1, step: -1), axis: -1), sinks,
                    ], axis: -1)
                    let expected = MLXFast.scaledDotProductAttention(
                        queries: query, keys: concatenated([keys, zero], axis: 2),
                        values: concatenated([values, zero], axis: 2),
                        scale: Float(1) / Float(128).squareRoot(), mask: .array(mask))
                    let history = ActiveKVSpillHistory(
                        directory: FileManager.default.temporaryDirectory, pageTokens: 1024,
                        byteLimit: 160 << 20)
                    try history.append(keys: keys, values: values)
                    let actual = try Inkling2SpilledVector.attention(
                        queries: query, projectedBias: bias, sinks: sinks, history: history)
                    let actualBytes = actual.asData().data
                    let expectedBytes = expected.asData().data
                    try error.check()
                    let unsplit = ActiveKVSpillHistory(
                        directory: FileManager.default.temporaryDirectory, pageTokens: count + 1,
                        byteLimit: 160 << 20)
                    try unsplit.append(keys: keys, values: values)
                    let unsplitOutput = try Inkling2SpilledVector.attention(
                        queries: query, projectedBias: bias, sinks: sinks, history: unsplit)
                    let unsplitBytes = unsplitOutput.asData().data
                    try error.check()
                    XCTAssertEqual(unsplitBytes, actualBytes, "\(count) keys differ at page boundaries")
                    XCTAssertTrue(actualBytes == expectedBytes,
                                  "\(count) keys, max error \(max(abs(actual - expected)).item(Float.self))")
                    XCTAssertGreaterThan(history.bytesRead, 0)
                    XCTAssertEqual(unsplit.bytesRead, 0)
                }
            }
        }
    }

    func testBundledPagedPrefillMatchesNativeMaskedAttentionOnM1Max() throws {
        try assertPagedAttentionMatchesNative([(1024, 32), (2117, 21), (8192, 128), (8213, 21)]) {
            query, projected, sinks, history, extent in
            try Inkling2SpilledPrefill.attention(
                queries: query, projected: projected, sinks: sinks, history: history, extent: extent)
        }
    }

    func testPagedVectorMatchesNativeForShortPrefillTailsOnM1Max() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLX_SDPA_BLOCKS"] == nil)
        try assertPagedAttentionMatchesNative([(1024, 2), (4096, 3), (8213, 8), (32769, 4)]) {
            query, projected, sinks, history, _ in
            try Inkling2SpilledVector.attention(
                queries: query, projectedBias: projected, sinks: sinks, history: history)
        }
    }

    private func assertPagedAttentionMatchesNative(
        _ cases: [(count: Int, rows: Int)],
        attention: (MLXArray, MLXArray, MLXArray, ActiveKVSpillHistory, Int) throws -> MLXArray
    ) throws {
        try XCTSkipUnless(GPU.deviceInfo().architecture == "applegpu_g13s")
        try Device.withDefaultDevice(.gpu) {
            try withError { error in
                let dtype = DType.bfloat16
                let extent = 1024
                let projection = (cos(arange(2 * extent, dtype: .float32) * Float(0.013)) * Float(0.4))
                    .reshaped([2, extent]).asType(dtype)
                for (count, rows) in cases {
                    let query = sin(arange(24 * rows * 128, dtype: .float32) * Float(0.017))
                        .reshaped([1, 24, rows, 128]).asType(dtype)
                    let positions = arange(8 * count * 128, dtype: .float32)
                    let keys = cos(positions * Float(0.003) + Float(0.2))
                        .reshaped([1, 8, count, 128]).asType(dtype)
                    let values = sin(positions * Float(0.0041))
                        .reshaped([1, 8, count, 128]).asType(dtype)
                    let relative = sin(arange(rows * 24 * 2, dtype: .float32) * Float(0.02))
                        .reshaped([1, rows, 24, 2]).asType(dtype)
                    let sinks = (cos(arange(24 * rows, dtype: .float32)) * Float(0.8))
                        .reshaped([1, 24, rows, 1]).asType(dtype)
                    let mask = inklingBandedMask(
                        relative: relative, projection: projection, offset: count - rows,
                        count: count, window: 0, extent: extent)
                    let zero = MLXArray.zeros([1, 8, 1, 128], dtype: dtype)
                    let expected = MLXFast.scaledDotProductAttention(
                        queries: query, keys: concatenated([keys, zero], axis: 2),
                        values: concatenated([values, zero], axis: 2),
                        scale: Float(1) / Float(128).squareRoot(),
                        mask: .array(concatenated([mask, sinks], axis: -1)))
                    let history = ActiveKVSpillHistory(
                        directory: FileManager.default.temporaryDirectory, pageTokens: 1024,
                        byteLimit: 160 << 20)
                    try history.append(keys: keys, values: values)
                    let projected = matmul(relative, projection).transposed(0, 2, 1, 3)
                    let actual = try attention(query, projected, sinks, history, extent)
                    let actualBytes = actual.asData().data
                    let expectedBytes = expected.asData().data
                    try error.check()
                    XCTAssertTrue(actualBytes == expectedBytes,
                                  "\(count) keys and \(rows) rows, max error \(max(abs(actual - expected)).item(Float.self))")
                    XCTAssertGreaterThan(history.bytesRead, 0)
                }
            }
        }
    }

    func testAcrossPrefillAndDecodeWithGroupedQueriesAndPartialTail() throws {
        try Device.withDefaultDevice(.cpu) {
            let history = ActiveKVSpillHistory(
                directory: FileManager.default.temporaryDirectory, pageTokens: 2, byteLimit: 4096)
            let keys = MLXArray((0..<24).map { cos(Float($0) * 0.3) }, [1, 1, 6, 4])
            let values = MLXArray((0..<24).map { sin(Float($0) * 0.5) }, [1, 1, 6, 4])
            let projection = MLXArray([Float(0.2), -0.1, 0.5, -0.8, 0.9, 0.1], [2, 3])

            for range in [0..<1, 1..<5, 5..<6] {
                let rows = range.count
                let queries = MLXArray((0..<(8 * rows)).map { sin(Float($0 + range.lowerBound)) },
                                       [1, 2, rows, 4])
                let relative = MLXArray((0..<(4 * rows)).map { cos(Float($0) * 0.3) },
                                        [1, rows, 2, 2])
                let sinks = MLXArray((0..<(2 * rows)).map { Float($0) * 0.1 }, [1, 2, rows, 1])
                let actual = try inklingSpilledAttention(
                    queries: queries, keys: keys[.ellipsis, range, 0...],
                    values: values[.ellipsis, range, 0...], relative: relative,
                    projection: projection, sinks: sinks, history: history, extent: 3)
                let residentK = keys[.ellipsis, ..<range.upperBound, 0...]
                let residentV = values[.ellipsis, ..<range.upperBound, 0...]
                let projected = matmul(relative, projection).transposed(0, 2, 1, 3).asArray(Float.self)
                let biases = (0..<2).flatMap { head in
                    range.flatMap { queryPosition in
                        (0..<range.upperBound).map { keyPosition -> Float in
                            let distance = queryPosition - keyPosition
                            if distance < 0 { return -Float.infinity }
                            if distance >= 3 { return 0 }
                            let row = head * rows + queryPosition - range.lowerBound
                            return projected[row * 3 + distance]
                        }
                    }
                }
                let mask = MLXArray(biases, [1, 2, rows, range.upperBound])
                let scores = matmul(queries, residentK.swappedAxes(-1, -2)) * Float(0.5) + mask
                let weights = softmax(concatenated([scores, sinks], axis: -1), axis: -1)
                let expected = matmul(weights[.ellipsis, ..<range.upperBound], residentV)

                XCTAssertLessThanOrEqual(max(abs(actual - expected)).item(Float.self), 0.0001)
                XCTAssertEqual(history.count, range.upperBound)
                let retained = history.residentState.map { $0.dim(2) }
                XCTAssertEqual(retained, range.upperBound % 2 == 0 ? [] : [1, 1])
            }
        }
    }

    func testWiderDiskReadsPreserveTheAttentionReduction() throws {
        try Device.withDefaultDevice(.cpu) {
            let directory = FileManager.default.temporaryDirectory
            let narrow = ActiveKVSpillHistory(directory: directory, pageTokens: 128, byteLimit: 65_536)
            let wide = ActiveKVSpillHistory(directory: directory, pageTokens: 256, byteLimit: 65_536)
            let keys = MLXArray((0..<(257 * 4)).map { cos(Float($0) * 0.3) }, [1, 1, 257, 4]).asType(.bfloat16)
            let values = MLXArray((0..<(257 * 4)).map { sin(Float($0) * 0.5) }, [1, 1, 257, 4]).asType(.bfloat16)
            let projection = MLXArray([Float(0.2), -0.1, 0.5, -0.8, 0.9, 0.1], [2, 3])
            for range in [0..<96, 96..<224, 224..<256, 256..<257] {
                let rows = range.count
                let query = MLXArray((0..<(rows * 8)).map { sin(Float($0 + range.lowerBound)) },
                                     [1, 2, rows, 4])
                let relative = MLXArray((0..<(rows * 4)).map { cos(Float($0) * 0.2) },
                                        [1, rows, 2, 2])
                let sink = MLXArray.full([1, 2, rows, 1], values: MLXArray(Float(0.1)))
                func attend(_ history: ActiveKVSpillHistory) throws -> Data {
                    try inklingSpilledAttention(
                        queries: query, keys: keys[.ellipsis, range, 0...],
                        values: values[.ellipsis, range, 0...], relative: relative,
                        projection: projection, sinks: sink, history: history, extent: 3).asData().data
                }
                XCTAssertEqual(try attend(narrow), try attend(wide))
            }
            XCTAssertEqual(narrow.bytesWritten, wide.bytesWritten)
            XCTAssertGreaterThan(wide.bytesRead, 0)
        }
    }

    func testSpillFailurePropagatesToTheAttentionCaller() {
        Device.withDefaultDevice(.cpu) {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            let history = ActiveKVSpillHistory(directory: directory, pageTokens: 1, byteLimit: 4096)
            let tile = MLXArray.zeros([1, 1, 1, 4])
            do {
                _ = try inklingSpilledAttention(
                    queries: tile, keys: tile, values: tile,
                    relative: .zeros([1, 1, 1, 2]), projection: .zeros([2, 3]),
                    sinks: .zeros([1, 1, 1, 1]), history: history, extent: 3)
                XCTFail("Missing spill directory should fail the forward")
            } catch {
                XCTAssertTrue(error.localizedDescription.contains("open"))
            }
            XCTAssertEqual(history.count, 0)
        }
    }
}
#endif
