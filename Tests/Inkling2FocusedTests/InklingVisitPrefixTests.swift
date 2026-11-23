import Foundation
import MLX
import MLXLLM
@testable import MLXLMCommon
import XCTest

final class InklingVisitPrefixTests: XCTestCase {
    func testSnapshotIsolationClearAndBudget() throws {
        try Device.withDefaultDevice(.cpu) {
            let owner = UUID()
            func key(_ tokens: [Int], step: Int = 128, salt: String? = nil) -> CacheCoordinator.VisitPrefixKey {
                .init(tokens: tokens, step: step, salt: salt, model: owner)
            }
            func coordinator(limit: Int) -> CacheCoordinator {
                var config = CacheCoordinatorConfig(usePagedCache: false, enableDiskCache: false, maxCacheBytes: limit)
                config.useAlignedVisitSnapshots = true
                return CacheCoordinator(config: config)
            }
            let stored = key([1, 2, 3, 4])
            let visits = (0..<48).map { _ in
                let c = Inkling2VisitCache()
                c.state = Array(repeating: MLXArray.ones([1, 1, 4, 1]), count: 6)
                return c
            }
            let manager = coordinator(limit: 64 << 20)
            let (ticket, miss) = manager.beginVisitPrefix(stored)
            XCTAssertNil(miss)
            manager.storeVisitPrefix(stored, generation: ticket, cache: visits)
            XCTAssertGreaterThan(manager.promptSnapshotBytes, 0)
            do {
                let (_, entry) = manager.beginVisitPrefix(stored)
                let snapshot = try XCTUnwrap(entry)
                visits[0].state = Array(repeating: MLXArray.zeros([1, 1, 4, 1]), count: 6)
                XCTAssertEqual(sum(snapshot.cache[0].state[0]).item(Float.self), 4)
                withExtendedLifetime(snapshot) {
                    manager.clear()
                    XCTAssertGreaterThan(manager.promptSnapshotBytes, 0)
                }
            }
            XCTAssertEqual(manager.promptSnapshotBytes, 0)
            manager.storeVisitPrefix(stored, generation: ticket, cache: visits)
            XCTAssertNil(manager.beginVisitPrefix(stored).1)

            let reloaded = CacheCoordinator.VisitPrefixKey(tokens: stored.tokens, step: stored.step,
                                                           salt: stored.salt, model: UUID())
            for mismatch in [key([1, 2, 3, 5]), key(stored.tokens, step: 256),
                             key(stored.tokens, salt: "other"), reloaded] {
                let (ticket, _) = manager.beginVisitPrefix(stored)
                manager.storeVisitPrefix(stored, generation: ticket, cache: visits)
                XCTAssertNil(manager.beginVisitPrefix(mismatch).1)
                XCTAssertEqual(manager.promptSnapshotBytes, 0)
            }

            let tiny = coordinator(limit: 1)
            let (tinyTicket, _) = tiny.beginVisitPrefix(stored)
            tiny.storeVisitPrefix(stored, generation: tinyTicket, cache: visits)
            XCTAssertNil(tiny.beginVisitPrefix(stored).1)
            XCTAssertEqual(tiny.retainedCacheBytes, 0)
        }
    }
}
