import MLX
import XCTest

@testable import MLXLMCommon

final class MambaDiskRestoreOwnershipTests: XCTestCase {
    func testSparsePersistentSlotsKeepValuesAndIndependentWrappers() throws {
        let source = MambaCache(slots: 6, persistentSlotCount: 4)
        source[0] = MLXArray([Float(1), 2]).reshaped(1, 2)
        source[1] = MLXArray([Float(3), 4]).reshaped(1, 2)
        source[3] = MLXArray([Int32(5), 6]).reshaped(1, 2)
        source.offset = 12
        let arrays = TQDiskSerializer.serialize(cache: [source])
        MLX.eval(Array(arrays.values))
        let first = MambaCache(slots: 6, persistentSlotCount: 4)
        let second = MambaCache(slots: 6, persistentSlotCount: 4)
        for cache in [first, second] {
            var target: [KVCache] = [cache]
            XCTAssertEqual(restoreFromDiskArrays(arrays, into: &target, requirePromptBoundary: true), 12)
            MLX.eval(cache)
            XCTAssertNil(cache[2])
            XCTAssertNil(cache[4])
            XCTAssertNil(cache[5])
            XCTAssertFalse(cache[0] === source[0])
            XCTAssertFalse(cache[1] === source[1])
            XCTAssertFalse(cache[3] === source[3])
        }
        first[0] = MLXArray([Float(10), 20]).reshaped(1, 2)
        first[1] = MLXArray([Float(30), 40]).reshaped(1, 2)
        first[3] = MLXArray([Int32(50), 60]).reshaped(1, 2)
        MLX.eval(first)
        for cache in [source, second] {
            XCTAssertEqual(cache[0]?.asArray(Float.self), [1, 2])
            XCTAssertEqual(cache[1]?.asArray(Float.self), [3, 4])
            XCTAssertEqual(cache[3]?.asArray(Int32.self), [5, 6])
        }
    }
}
