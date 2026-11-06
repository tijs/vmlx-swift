import MLX
@testable import MLXVLM
import XCTest

final class Glm5NextLayoutRangeARangeTests: XCTestCase {
    func testExactIntegerLayouts() throws {
        try MLXMetalTestLock.withLock {
            for count in [0, 1, 3, 4, 5, 2047, 2048, 2049, 7621, 8689, 10142, 10780, 11723, 11787, 20001, 32769] {
                let host = MLXArray(Int32(0) ..< Int32(count))
                let device = Glm5NextIndexerRuntime.layoutRange(count)
                eval(host, device)
                XCTAssertEqual(host.dtype, .int32)
                XCTAssertEqual(device.dtype, .int32)
                XCTAssertEqual(host.shape, [count])
                XCTAssertEqual(device.shape, [count])
                XCTAssertEqual(device.asArray(Int32.self), host.asArray(Int32.self))
            }
            // Bounded rows expose accidental F32 arange rounding without huge allocations.
            for start in [16_777_214, Int(Int32.max) - 3] {
                let stop = start + 3
                let values = arange(start, stop, dtype: .int32)
                eval(values)
                XCTAssertEqual(values.asArray(Int32.self), (start ..< stop).map(Int32.init))
            }
        }
    }

    func testLayoutCountBounds() {
        XCTAssertTrue(Glm5NextIndexerRuntime.layoutRangeCountIsRepresentable(0))
        XCTAssertTrue(Glm5NextIndexerRuntime.layoutRangeCountIsRepresentable(Int(Int32.max)))
        XCTAssertFalse(Glm5NextIndexerRuntime.layoutRangeCountIsRepresentable(-1))
        XCTAssertFalse(Glm5NextIndexerRuntime.layoutRangeCountIsRepresentable(Int.min))
        XCTAssertFalse(Glm5NextIndexerRuntime.layoutRangeCountIsRepresentable(Int(Int32.max) + 1))
        XCTAssertFalse(Glm5NextIndexerRuntime.layoutRangeCountIsRepresentable(Int.max))
        // Rounding up a valid token count to complete pools can exceed Int32 bounds.
        let n = Int(Int32.max)
        XCTAssertFalse(Glm5NextIndexerRuntime.layoutRangeCountIsRepresentable(((n + 3) / 4) * 4))
    }
}
