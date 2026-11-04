// Copyright © 2026 Apple Inc.
#if canImport(Darwin) && canImport(CryptoKit)
import Foundation
import CryptoKit
import Testing
@testable import MLXLMCommon

struct Qwen35AffineResearchSnapshotTests {
    private func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    @Test("cloned snapshot remains independent of later source modification")
    func sourceMutation() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let source = root.appendingPathComponent("source")
        let staging = root.appendingPathComponent("staging")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let before = Data("original checkpoint".utf8)
        try before.write(to: source.appendingPathComponent("weights"))
        let snapshot = try Qwen35AffineResearchSnapshot.create(source: source, stagingParent: staging,
            files: ["weights": digest(before)])
        try Data("modified".utf8).write(to: source.appendingPathComponent("weights"))
        #expect(try Data(contentsOf: snapshot.directory.appendingPathComponent("weights")) == before)
        try snapshot.dispose()
        try snapshot.dispose()
        #expect(!FileManager.default.fileExists(atPath: snapshot.directory.path))
    }

    @Test("mismatch, missing source and escaping paths leave no staged tree")
    func failureCleanup() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let source = root.appendingPathComponent("source")
        let staging = root.appendingPathComponent("staging")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let value = Data("data".utf8)
        try value.write(to: source.appendingPathComponent("weights"))
        for files in [["weights": digest(Data("wrong".utf8))],
                      ["weights": digest(value), "z-missing": digest(value)],
                      ["../weights": digest(value)]] {
            #expect(throws: (any Error).self) {
                try Qwen35AffineResearchSnapshot.create(source: source, stagingParent: staging, files: files)
            }
            #expect(try FileManager.default.contentsOfDirectory(atPath: staging.path).isEmpty)
        }
    }
}
#endif
