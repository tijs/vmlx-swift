import MLX
import XCTest
@testable import MLXLLM

final class InklingBandedMaskTests: XCTestCase {
    func testStridedMaskMatchesCausalWindowAndLearnedBias() throws {
        try Device.withDefaultDevice(.gpu) {
            let batch = 2, heads = 3, extent = 32
            for dtype: DType in [.float32, .bfloat16, .float16] {
                for (queries, keys, window) in [(1, 1, 0), (7, 17, 4), (129, 257, 0), (1, 1025, 1024)] {
                    let offset = keys - queries
                    let values = (0..<(batch * queries * heads * extent)).map { Float($0 % 127) / 8 }
                    let projected = MLXArray(values, [batch, queries, heads, extent])
                        .asType(dtype).transposed(0, 2, 1, 3)
                    let mask = try XCTUnwrap(Inkling2InferenceKernels.bandedMask(
                        projected, count: keys, offset: offset, window: window, extent: extent))
                    var expected = [Float]()
                    for b in 0..<batch {
                        for h in 0..<heads {
                            for q in 0..<queries {
                                for k in 0..<keys {
                                    let distance = offset + q - k
                                    let row = (b * queries + q) * heads + h
                                    if distance < 0 || (window > 0 && distance >= window) {
                                        expected.append(-1e30)
                                    } else if distance >= extent {
                                        expected.append(0)
                                    } else {
                                        expected.append(values[row * extent + distance])
                                    }
                                }
                            }
                        }
                    }
                    let shape = [batch, heads, queries, keys]
                    XCTAssertEqual(mask.shape, shape)
                    XCTAssertTrue(arrayEqual(mask, MLXArray(expected, shape).asType(dtype)).item(Bool.self))
                }
            }
        }
    }
}
