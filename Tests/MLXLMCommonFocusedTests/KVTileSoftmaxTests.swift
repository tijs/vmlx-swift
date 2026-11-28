import MLX
import XCTest

@testable import MLXLMCommon

final class KVTileSoftmaxTests: XCTestCase {
    func testTilesMatchDenseAttentionWithSinkAndPositionBias() {
        Device.withDefaultDevice(.cpu) {
            let masked = -Float.infinity
            // The rows use different causal masks and include relative bias.
            // The large score in the second tile must rescale the first tile.
            let scores = MLXArray([Float(1), 0, masked, masked, -3, 2, 95, masked], [1, 2, 4])
            let values = MLXArray([Float(1), 2, 10, 20, 100, 200, 900, 900], [1, 4, 2])
            let sink = MLXArray([Float(2), 94], [1, 2, 1])
            var accumulator = KVTileSoftmax(rowShape: [1, 2], valueWidth: 2, sink: sink)
            accumulator.accumulate(scores: scores[.ellipsis, ..<2], values: values[0..., ..<2, 0...])
            accumulator.materialize()
            accumulator.accumulate(scores: scores[.ellipsis, 2...], values: values[0..., 2..., 0...])
            accumulator.materialize()

            let denseScores = concatenated([scores, sink], axis: -1)
            let denseValues = concatenated([values, .zeros([1, 1, 2])], axis: -2)
            let expected = matmul(softmax(denseScores, axis: -1), denseValues).asArray(Float.self)
            let actual = accumulator.output(dtype: .float32).asArray(Float.self)
            XCTAssertEqual(actual.count, expected.count)
            for (streamed, dense) in zip(actual, expected) {
                XCTAssertEqual(streamed, dense, accuracy: 0.0001)
            }
        }
    }

    func testMaskedPagesBeforeVisibleKeysAndFullyMaskedRow() {
        Device.withDefaultDevice(.cpu) {
            let masked = -Float.infinity
            var accumulator = KVTileSoftmax(rowShape: [1, 2], valueWidth: 1)
            accumulator.accumulate(
                scores: MLXArray([masked, masked], [1, 2, 1]),
                values: MLXArray([Float(60)], [1, 1, 1]))
            accumulator.accumulate(
                scores: MLXArray([Float(-100), masked], [1, 2, 1]),
                values: MLXArray([Float(13)], [1, 1, 1]))
            accumulator.materialize()
            XCTAssertEqual(accumulator.output(dtype: .float32).asArray(Float.self), [13, 0])
        }
    }
}
