#if canImport(Darwin)
import Foundation
import MLX
import XCTest
@testable import MLXLMCommon

final class ActiveKVSpillFileTests: XCTestCase {
    func testRoundTripAndRollbackReuseDiskSlot() throws {
        try withDirectory { directory in
            let store = try ActiveKVSpillFile(directory: directory, pageBytes: 4096, byteLimit: 8193)
            let original = Data(repeating: 0x17, count: store.pageBytes)
            let discarded = Data(repeating: 0x85, count: store.pageBytes)
            let replacement = Data(repeating: 0x34, count: store.pageBytes)
            XCTAssertEqual(try original.withUnsafeBytes(store.append), 0)
            XCTAssertEqual(try discarded.withUnsafeBytes(store.append), 1)
            XCTAssertThrowsError(try replacement.withUnsafeBytes(store.append)) {
                XCTAssertEqual($0 as? ActiveKVSpillFile.Failure, .capacityExceeded)
            }
            XCTAssertEqual(store.pageCount, 2)
            try store.truncate(toPageCount: 1)
            var page = Data(count: store.pageBytes)
            XCTAssertThrowsError(try page.withUnsafeMutableBytes { try store.read(page: 1, into: $0) }) {
                XCTAssertEqual($0 as? ActiveKVSpillFile.Failure, .invalidPage)
            }
            XCTAssertEqual(try replacement.withUnsafeBytes(store.append), 1)
            try page.withUnsafeMutableBytes { try store.read(page: 1, into: $0) }
            XCTAssertEqual(page, replacement)
            try page.withUnsafeMutableBytes { try store.read(page: 0, into: $0) }
            XCTAssertEqual(page, original)
            XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: directory.path).isEmpty)
        }
    }

    func testInvalidInputsLeaveCommittedPagesUntouched() throws {
        try withDirectory { directory in
            XCTAssertThrowsError(try ActiveKVSpillFile(directory: directory, pageBytes: 0, byteLimit: 4096))
            XCTAssertThrowsError(try ActiveKVSpillFile(directory: directory, pageBytes: 4096, byteLimit: 4095))
            let store = try ActiveKVSpillFile(directory: directory, pageBytes: 4096, byteLimit: 8192)
            XCTAssertThrowsError(try store.truncate(toPageCount: 1))
            XCTAssertThrowsError(try Data(count: 4095).withUnsafeBytes(store.append))
            XCTAssertEqual(store.pageCount, 0)
            var page = Data(count: store.pageBytes)
            XCTAssertThrowsError(try page.withUnsafeMutableBytes { try store.read(page: -1, into: $0) })
            XCTAssertThrowsError(try ActiveKVSpillFile(
                directory: directory.appendingPathComponent("missing"), pageBytes: 4096, byteLimit: 4096
            )) {
                guard case .systemCall("mkstemp", _) = $0 as? ActiveKVSpillFile.Failure else {
                    return XCTFail("Expected a file creation error, got \($0)")
                }
            }
        }
    }

    func testConcurrentAppendsCommitDistinctWholePages() throws {
        try withDirectory { directory in
            let pageBytes = 4096
            let count = 16
            let store = try ActiveKVSpillFile(directory: directory, pageBytes: pageBytes, byteLimit: count * pageBytes)
            DispatchQueue.concurrentPerform(iterations: count) { index in
                let source = Data(repeating: UInt8(index), count: pageBytes)
                do {
                    _ = try source.withUnsafeBytes(store.append)
                } catch {
                    XCTFail("Unexpected append error: \(error)")
                }
            }
            XCTAssertEqual(store.pageCount, count)
            var markers = Set<UInt8>()
            var page = Data(count: pageBytes)
            for index in 0..<count {
                try page.withUnsafeMutableBytes { try store.read(page: index, into: $0) }
                let marker = try XCTUnwrap(page.first)
                XCTAssertEqual(page, Data(repeating: marker, count: pageBytes))
                markers.insert(marker)
            }
            XCTAssertEqual(markers, Set((0..<count).map(UInt8.init)))
        }
    }

    func testCancelledAppendPublishesNoPage() async throws {
        let store = try ActiveKVSpillFile(
            directory: FileManager.default.temporaryDirectory, pageBytes: 4096, byteLimit: 4096)
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try Data(count: store.pageBytes).withUnsafeBytes(store.append)
        }
        do {
            _ = try await task.value
            XCTFail("Cancelled append succeeded")
        } catch is CancellationError {
            XCTAssertEqual(store.pageCount, 0)
        }
    }

    func testForkedHistoryChargesInheritedPagesWithoutOverwritingThem() throws {
        try withDirectory { directory in
            try Device.withDefaultDevice(.cpu) {
                let pageBytes = 2 * 2 * MemoryLayout<Float>.size
                let original = ActiveKVSpillHistory(
                    directory: directory, pageTokens: 2, byteLimit: 3 * pageBytes)
                func append(_ values: [Float], to history: ActiveKVSpillHistory) throws {
                    let tile = MLXArray(values, [1, 1, values.count, 1])
                    try history.append(keys: tile, values: -tile)
                }
                func read(_ history: ActiveKVSpillHistory, pages: IndexSet) throws -> [Float] {
                    var keys: [Float] = []
                    try history.forEachTile(selectedPages: pages) { _, tile, _ in
                        keys.append(contentsOf: tile.asArray(Float.self))
                    }
                    return keys
                }

                try append([1, 2], to: original)
                let smallerQuota = ActiveKVSpillHistory(
                    directory: directory, pageTokens: 2, byteLimit: 2 * pageBytes)
                let fork = try XCTUnwrap(original.fork(into: smallerQuota))
                XCTAssertEqual(fork.storedBytes, pageBytes)
                XCTAssertEqual(fork.bytesWritten, 0)
                try append([3, 4], to: original)
                try append([5, 6], to: fork)
                XCTAssertEqual(try read(original, pages: IndexSet(0..<2)), [1, 2, 3, 4])
                XCTAssertEqual(try read(fork, pages: IndexSet(0..<2)), [1, 2, 5, 6])
                XCTAssertThrowsError(try append([7, 8], to: fork)) {
                    XCTAssertEqual($0 as? ActiveKVSpillFile.Failure, .capacityExceeded)
                }
                XCTAssertEqual(fork.bytesWritten, UInt64(pageBytes))
                XCTAssertEqual(try read(original, pages: IndexSet(integer: 0)), [1, 2])
            }
        }
    }

    private func withDirectory(_ body: (URL) throws -> Void) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        try body(directory)
    }
}
#endif
