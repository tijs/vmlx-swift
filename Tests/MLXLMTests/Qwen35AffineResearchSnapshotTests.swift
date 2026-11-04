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
            files: ["weights": digest(before)], diskGate: { _, _ in })
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
                try Qwen35AffineResearchSnapshot.create(source: source, stagingParent: staging, files: files, diskGate: { _, _ in })
            }
            #expect(try FileManager.default.contentsOfDirectory(atPath: staging.path).isEmpty)
        }
    }
    enum Failure: Error, Equatable { case gate, clone }

    @Test("disk gate stages are ordered and failure cleans partial snapshots")
    func diskGate() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let source = root.appendingPathComponent("source")
        let staging = root.appendingPathComponent("staging")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let data = Data("weight".utf8)
        try data.write(to: source.appendingPathComponent("a"))
        try data.write(to: source.appendingPathComponent("b"))
        var stages: [String] = []
        #expect(throws: Failure.gate) {
            try Qwen35AffineResearchSnapshot.create(source: source, stagingParent: staging,
                files: ["a": digest(data), "b": digest(data)], diskGate: { stage, path in
                    stages.append("\(path):\(stage)")
                    if path == "b" && stage == .afterClone { throw Failure.gate }
                })
        }
        #expect(stages == ["a:beforeClone", "a:afterClone", "a:beforeHash", "a:duringHash", "a:afterHash",
                           "b:beforeClone", "b:afterClone"])
        #expect(try FileManager.default.contentsOfDirectory(atPath: staging.path).isEmpty)
    }

    @Test("clone failure has no copy fallback and removes staged tree")
    func cloneFailure() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let source = root.appendingPathComponent("source")
        let staging = root.appendingPathComponent("staging")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let data = Data("weight".utf8)
        try data.write(to: source.appendingPathComponent("a"))
        var calls = 0
        #expect(throws: Failure.clone) {
            try Qwen35AffineResearchSnapshot.createForTesting(source: source, stagingParent: staging,
                files: ["a": digest(data)], diskGate: { _, _ in }, clone: { _, _ in
                    calls += 1
                    throw Failure.clone
                })
        }
        #expect(calls == 1)
        #expect(try FileManager.default.contentsOfDirectory(atPath: staging.path).isEmpty)
    }

    @Test("cancellation after clone removes the partial snapshot")
    func cancellation() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let source = root.appendingPathComponent("source")
        let staging = root.appendingPathComponent("staging")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let data = Data("weight".utf8)
        try data.write(to: source.appendingPathComponent("a"))
        let expected = digest(data)
        let cancelled = await Task.detached {
            do {
                _ = try Qwen35AffineResearchSnapshot.create(source: source, stagingParent: staging,
                    files: ["a": expected], diskGate: { stage, _ in
                        if stage == .afterClone { withUnsafeCurrentTask { $0?.cancel() } }
                    })
                return false
            } catch is CancellationError { return true }
            catch { return false }
        }.value
        #expect(cancelled)
        #expect(try FileManager.default.contentsOfDirectory(atPath: staging.path).isEmpty)
    }

    @Test("mid-hash gate failure removes a partially processed snapshot")
    func midHashFailure() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let source = root.appendingPathComponent("source")
        let staging = root.appendingPathComponent("staging")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let data = Data(repeating: 7, count: 2 * 1024 * 1024 + 1)
        try data.write(to: source.appendingPathComponent("weights"))
        var hashChecks = 0
        #expect(throws: Failure.gate) {
            try Qwen35AffineResearchSnapshot.create(source: source, stagingParent: staging,
                files: ["weights": digest(data)], diskGate: { stage, _ in
                    if stage == .duringHash {
                        hashChecks += 1
                        if hashChecks == 2 { throw Failure.gate }
                    }
                })
        }
        #expect(hashChecks == 2)
        #expect(try FileManager.default.contentsOfDirectory(atPath: staging.path).isEmpty)
    }

    @Test("arena lease prevents disposal through failed drain and recovery")
    func arenaLeaseQuarantine() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let source = root.appendingPathComponent("source")
        let staging = root.appendingPathComponent("staging")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let data = Data("weight".utf8)
        try data.write(to: source.appendingPathComponent("weights"))
        let snapshot = try Qwen35AffineResearchSnapshot.create(source: source, stagingParent: staging,
            files: ["weights": digest(data)], diskGate: { _, _ in })
        let gate = ExperimentalRequestArenaGate()
        do {
            let _: Void = try gate.withRequest(operation: { arena in
                try snapshot.retain(in: arena)
                #expect(throws: Qwen35AffineResearchSnapshot.SnapshotError.self) { try snapshot.dispose() }
                throw CancellationError()
            }, drain: { _ in throw Failure.gate }, invalidate: { _ in })
            Issue.record("failed drain accepted")
        } catch let error as ExperimentalRequestArenaCleanupError {
            #expect(error.primary is CancellationError)
            #expect(error.cleanup is Failure)
        }
        #expect(throws: Qwen35AffineResearchSnapshot.SnapshotError.self) { try snapshot.dispose() }
        try gate.recover(drain: { _ in
            let stored = try Data(contentsOf: snapshot.directory.appendingPathComponent("weights"))
            #expect(stored == data)
        }, invalidate: { _ in
            #expect(throws: Qwen35AffineResearchSnapshot.SnapshotError.self) { try snapshot.dispose() }
        })
        try snapshot.dispose()
        #expect(!FileManager.default.fileExists(atPath: snapshot.directory.path))
    }

    @Test("registration into a sealed arena does not leak a snapshot lease")
    func sealedArenaLease() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let source = root.appendingPathComponent("source")
        let staging = root.appendingPathComponent("staging")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let data = Data("weight".utf8)
        try data.write(to: source.appendingPathComponent("weights"))
        let snapshot = try Qwen35AffineResearchSnapshot.create(source: source, stagingParent: staging,
            files: ["weights": digest(data)], diskGate: { _, _ in })
        var sealed: ExperimentalRequestArena?
        let _: Void = try ExperimentalRequestArenaGate().withRequest(operation: { arena in
            sealed = arena
            return .finished(())
        }, drain: { _ in }, invalidate: { _ in })
        #expect(throws: ExperimentalRequestArenaError.sealed) { try snapshot.retain(in: try #require(sealed)) }
        try snapshot.dispose()
        #expect(throws: Qwen35AffineResearchSnapshot.SnapshotError.self) { try snapshot.retain(in: try #require(sealed)) }
    }

}
#endif
