// Copyright © 2026 Apple Inc.
import Foundation
import XCTest
import Cmlx
@testable import MLX

final class SafetensorsLoadStatusTests: XCTestCase {
    func testStatusWithoutCallbackFails() throws {
        let url = URL(fileURLWithPath: "/tmp/status-only.safetensors")
        try checkedSafetensorsLoadStatus(url: url) { 0 }
        XCTAssertThrowsError(try checkedSafetensorsLoadStatus(url: url) { 7 }) { error in
            guard let loadError = error as? LoadSaveError,
                case .unableToOpen(let actualURL, let message) = loadError else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(actualURL, url)
            XCTAssertTrue(message.contains("7"))
        }
    }

    func testCallbackErrorWinsOverNonzeroStatus() {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
        XCTAssertThrowsError(try checkedSafetensorsLoadStatus(url: url) {
            var result = mlx_array_new()
            defer { mlx_array_free(result) }
            var shape = [Int32(1)]
            return url.withUnsafeFileSystemRepresentation { path -> Int32 in
                guard let path else { return 1 }
                return mlx_array_new_mmap_file_region(&result, path, 0, 4, &shape, 1, DType.float32.cmlxDtype)
            }
        }) { error in
            XCTAssertTrue(error is MLXError)
            XCTAssertTrue(String(describing: error).contains(url.path))
        }
    }

    func testMissingFileThrowsForFilteredExactAndUnfilteredPaths() {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
        for (keys, exact) in [(Set(["drop"]), false), (Set(["drop"]), true), (Set<String>(), false)] {
            XCTAssertThrowsError(try loadArraysAndMetadata(url: url, excludingKeys: keys, exactTensorBuffers: exact)) { error in
                XCTAssertTrue(error is MLXError)
                XCTAssertTrue(String(describing: error).contains(url.path))
            }
        }
        XCTAssertThrowsError(try loadArrays(url: url)) { error in
            XCTAssertTrue(error is MLXError)
            XCTAssertTrue(String(describing: error).contains(url.path))
        }
    }
}
