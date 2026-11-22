import Cmlx
import Foundation
import MLX
import MLXLMCommon
import XCTest
@testable import MLXLLM

final class InklingCacheEvaluationTests: XCTestCase {
    private func isAvailable(_ array: MLXArray) -> Bool {
        var available = false
        XCTAssertEqual(_mlx_array_is_available(&available, array.ctx), 0)
        return available
    }

    private func checkChunkBoundaryEvaluation() {
        let lhs = MLXArray([Float(1), 2, 3, 4, 5, 6, 7, 8], [2, 4])
        let rhs = MLXArray((0..<16).map { Float(($0 % 5) + 1) }, [4, 4])
        let product = matmul(lhs, rhs)
        let cache = Inkling2VisitCache()
        _ = cache.update(keys: product.reshaped([1, 1, 2, 4]),
                         values: (product + 1).reshaped([1, 1, 2, 4]))
        for i in 0..<4 {
            cache.convolutions[i] = (product[0..<1] + Float(i + 2)).reshaped([1, 1, 4])
        }
        let states = cache.state
        XCTAssertEqual(states.count, 6)
        // Inspect scheduling status without evaluating or touching data from
        // an unallocated lazy array. item/asArray would force the work itself.
        for array in states { XCTAssertFalse(isAvailable(array)) }

        // Exactly the typed collection used by LLMModel.prepare between
        // prompt chunks. Check materialization before reading any values.
        let caches: [KVCache] = [cache]
        MLX.eval(caches)
        for array in states { XCTAssertTrue(isAvailable(array)) }
        let evaluated = cache.innerState()
        XCTAssertEqual(evaluated.count, 6)
        XCTAssertEqual(evaluated.reduce(0) { $0 + $1.nbytes }, 128)
        XCTAssertEqual(cache.offset, 2)
        XCTAssertFalse(cache.isTrimmable)
        XCTAssertEqual(cache.trim(1), 0)
    }

    func testEvictionPreservesChunkKeysAndAbsolutePositionAcrossCopyAndRestore() {
        Device.withDefaultDevice(.cpu) {
            let cache = Inkling2VisitCache(window: 3)
            for i in 0..<4 { cache.convolutions[i] = MLXArray([Float(i)]) }
            func append(_ rows: [Int32], to cache: Inkling2VisitCache) -> [Int32] {
                let keys = MLXArray(rows, [1, 1, rows.count, 1])
                let (allKeys, allValues) = cache.update(keys: keys, values: keys + 1)
                XCTAssertEqual(allValues.asArray(Int32.self), allKeys.asArray(Int32.self).map { $0 + 1 })
                return allKeys.asArray(Int32.self)
            }
            XCTAssertEqual(append([0, 1, 2, 3, 4], to: cache), [0, 1, 2, 3, 4])
            XCTAssertEqual(append([5, 6], to: cache), [2, 3, 4, 5, 6])
            XCTAssertEqual(cache.state[0].asArray(Int32.self), [4, 5, 6])
            let copy = cache.copy() as! Inkling2VisitCache
            let restored = Inkling2VisitCache(window: 3)
            restored.state = cache.state
            restored.metaState = cache.metaState
            for branch in [copy, restored] {
                XCTAssertEqual(branch.offset, 7)
                XCTAssertEqual(append([7], to: branch), [4, 5, 6, 7])
                XCTAssertEqual(branch.state[0].asArray(Int32.self), [5, 6, 7])
                XCTAssertEqual(branch.offset, 8)
            }
            XCTAssertEqual(cache.offset, 7)
        }
    }

    func testChunkBoundaryMaterializesKVAndAllConvolutionStatesCPU() {
        Device.withDefaultDevice(.cpu) { checkChunkBoundaryEvaluation() }
    }

    func testChunkBoundaryMaterializesKVAndAllConvolutionStatesMetal() throws {
        guard ProcessInfo.processInfo.environment["VMLX_INKLING_METAL_FIXTURE"] == "1" else {
            throw XCTSkip("Explicit Metal fixture session required")
        }
        Device.withDefaultDevice(.gpu) { checkChunkBoundaryEvaluation() }
    }
}
