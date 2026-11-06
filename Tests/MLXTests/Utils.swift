// Copyright © 2024 Apple Inc.

import Foundation
import Cmlx
import MLX
import XCTest

/// assert two arrays have the same shape and contents
func assertEqual(
    _ array1: MLXArray, _ array2: MLXArray, rtol: Double = 1e-5, atol: Double = 1e-8,
    file: StaticString = #filePath, line: UInt = #line
) {
    XCTAssertEqual(
        array1.shape, array2.shape, "shapes differ: \(array1.shape) != \(array2.shape)",
        file: file, line: line)
    XCTAssertTrue(
        array1.allClose(array2, rtol: rtol, atol: atol).item(Bool.self),
        "contents differ:\n\(array1)\n\(array2)", file: file, line: line)
}

func assertEqual(
    _ array1: [MLXArray], _ array2: [MLXArray], rtol: Double = 1e-5, atol: Double = 1e-8,
    file: StaticString = #filePath, line: UInt = #line
) {
    XCTAssertEqual(array1.count, array2.count, file: file, line: line)
    for (e1, e2) in zip(array1, array2) {
        assertEqual(e1, e2, rtol: rtol, atol: atol, file: file, line: line)
    }
}

func assertNotEqual(
    _ array1: MLXArray, _ array2: MLXArray, rtol: Double = 1e-5, atol: Double = 1e-8,
    file: StaticString = #filePath, line: UInt = #line
) {
    XCTAssertEqual(
        array1.shape, array2.shape, "shapes differ: \(array1.shape) != \(array2.shape)",
        file: file, line: line)
    XCTAssertFalse(
        array1.allClose(array2, rtol: rtol, atol: atol).item(Bool.self),
        "contents same:\n\(array1)\n\(array2)", file: file, line: line)
}

private let mlxTestRepoRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .standardizedFileURL

private let mlxMetallibSourceDirectoryForTests: URL? = {
    let sourceDirectories = [
        mlxTestRepoRoot.appendingPathComponent(".build/arm64-apple-macosx/debug"),
        mlxTestRepoRoot.appendingPathComponent(".build/debug"),
    ]
    return sourceDirectories.first {
        FileManager.default.fileExists(atPath: $0.appendingPathComponent("default.metallib").path)
    }
}()

private final class MLXTestBundleProbe {}

func setDefaultDevice() {
    prepareMLXMetallibForTests()
    MLX.Device.setDefault(device: .gpu)
}

private let mlxMetallibPreparedForTests: Void = {
    if let path = ProcessInfo.processInfo.environment["VMLX_TEST_METALLIB_PATH"] {
        precondition(FileManager.default.isReadableFile(atPath: path))
        precondition(mlx_metal_set_metallib_path(path) == 0)
        return
    }
    let fileManager = FileManager.default
    guard let sourceDirectory = mlxMetallibSourceDirectoryForTests else {
        return
    }
    let source = sourceDirectory.appendingPathComponent("default.metallib")
    let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .standardizedFileURL

    var targetDirectories: [URL] = []
    if let executableURL = Bundle.main.executableURL {
        targetDirectories.append(executableURL.deletingLastPathComponent())
    }
    if let resourceURL = Bundle.main.resourceURL {
        targetDirectories.append(resourceURL)
    }
    let testBundle = Bundle(for: MLXTestBundleProbe.self)
    if let executableURL = testBundle.executableURL {
        targetDirectories.append(executableURL.deletingLastPathComponent())
    }
    if let resourceURL = testBundle.resourceURL {
        targetDirectories.append(resourceURL)
    }
    if let firstArgument = CommandLine.arguments.first, !firstArgument.isEmpty {
        targetDirectories.append(URL(fileURLWithPath: firstArgument).deletingLastPathComponent())
    }

    var scanned = Set<String>()
    for candidate in targetDirectories {
        var directory = candidate.standardizedFileURL
        for _ in 0 ..< 4 {
            let path = directory.path
            if scanned.insert(path).inserted {
                try? fileManager.copyMetallibIfMissing(from: source, into: directory)
            }
            directory.deleteLastPathComponent()
        }
    }
}()

func prepareMLXMetallibForTests() {
    _ = mlxMetallibPreparedForTests
}

func withMLXMetallibForTests<T>(_ body: () throws -> T) rethrows -> T {
    prepareMLXMetallibForTests()
    let originalDirectory = FileManager.default.currentDirectoryPath
    if let sourceDirectory = mlxMetallibSourceDirectoryForTests {
        FileManager.default.changeCurrentDirectoryPath(sourceDirectory.path)
    }
    defer {
        FileManager.default.changeCurrentDirectoryPath(originalDirectory)
    }
    return try body()
}

private extension FileManager {
    func copyMetallibIfMissing(from source: URL, into directory: URL) throws {
        try createDirectory(at: directory, withIntermediateDirectories: true)
        for name in ["default.metallib", "mlx.metallib"] {
            let destination = directory.appendingPathComponent(name)
            if !fileExists(atPath: destination.path) {
                try copyItem(at: source, to: destination)
            }
        }
    }
}
