// Copyright © 2026 Apple Inc.

import Foundation
#if canImport(CryptoKit)
import CryptoKit
#endif
#if canImport(Darwin)
import Darwin
#endif

/// A clone-only private checkpoint staging area for research. The caller owns
/// its lifetime and must dispose it only after all model use, GPU work, and
/// file-backed arrays have ended. It is not integrated with inference.
///
/// Source files and the private staging tree must be trusted and immutable
/// while accessed. Checks do not defend against same-user concurrent path
/// replacement, chmod, truncation or hostile mutation. Per-file hashes identify
/// the staged bytes, not the provenance of the supplied expected digests.
public final class Qwen35AffineResearchSnapshot {
    public struct SnapshotError: Error, CustomStringConvertible {
        public let description: String
        init(_ description: String) { self.description = description }
    }

    public enum DiskGateStage: Sendable, Equatable {
        case beforeClone, afterClone, beforeHash, duringHash, afterHash
    }

    public let directory: URL
    private let lock = NSLock()
    private var disposed = false
    private var arenaLeases = 0
    private init(directory: URL) { self.directory = directory }

    /// `files` maps every required relative path to its independently pinned
    /// lowercase SHA-256 digest. No files are discovered implicitly.
    /// The mandatory gate owns all disk thresholds and is called before and
    /// after each clone and each hash. Clones do not reserve physical space;
    /// later writes may consume additional space. The gate is also checked for
    /// each 1 MiB hash block. Cancellation observes the current Swift task at
    /// boundaries and during hashing; callers outside a task must enforce their
    /// own cancellation through diskGate. It cannot interrupt a clone syscall.
    public static func create(source: URL, stagingParent: URL, files: [String: String],
                              diskGate: (DiskGateStage, String) throws -> Void) throws -> Qwen35AffineResearchSnapshot {
        #if canImport(Darwin) && canImport(CryptoKit)
        return try createForTesting(source: source, stagingParent: stagingParent, files: files,
            diskGate: diskGate, clone: { source, target in
                guard clonefile(source, target, 0) == 0 else {
                    throw SnapshotError("APFS clone failed: errno \(errno)")
                }
            })
        #else
        throw SnapshotError("APFS clone snapshots require Darwin and CryptoKit")
        #endif
    }

    #if canImport(Darwin) && canImport(CryptoKit)
    // Internal injection allows testing clone failure without an unsafe or
    // filesystem-dependent fallback; production always uses clonefile.
    static func createForTesting(source: URL, stagingParent: URL, files: [String: String],
                                 diskGate: (DiskGateStage, String) throws -> Void,
                                 clone: (String, String) throws -> Void) throws -> Qwen35AffineResearchSnapshot {
        try Task<Never, Never>.checkCancellation()
        guard !files.isEmpty else { throw SnapshotError("empty snapshot manifest") }
        for (path, digest) in files {
            let parts = path.split(separator: "/", omittingEmptySubsequences: false)
            guard !path.hasPrefix("/"), !parts.isEmpty,
                parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }),
                !path.utf8.contains(0), digest.utf8.count == 64,
                digest.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) })
            else { throw SnapshotError("invalid snapshot path or digest") }
        }
        let root = source.resolvingSymlinksInPath().standardizedFileURL
        let parent = stagingParent.resolvingSymlinksInPath().standardizedFileURL
        let destination = parent.appendingPathComponent("qwen35-affine-\(UUID().uuidString)", isDirectory: true)
        guard mkdir(destination.path, 0o700) == 0 else { throw SnapshotError("cannot create private snapshot directory") }
        do {
            for path in files.keys.sorted() {
                try Task<Never, Never>.checkCancellation()
                var sourcePath = root
                for part in path.split(separator: "/") {
                    sourcePath.appendPathComponent(String(part))
                    var st = stat()
                    guard lstat(sourcePath.path, &st) == 0, (st.st_mode & S_IFMT) != S_IFLNK else {
                        throw SnapshotError("missing or symbolic source component: \(path)")
                    }
                }
                var st = stat()
                guard lstat(sourcePath.path, &st) == 0, (st.st_mode & S_IFMT) == S_IFREG else {
                    throw SnapshotError("source is not a regular file: \(path)")
                }
                let target = destination.appendingPathComponent(path)
                try FileManager.default.createDirectory(at: target.deletingLastPathComponent(),
                    withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
                try diskGate(.beforeClone, path)
                try Task<Never, Never>.checkCancellation()
                try clone(sourcePath.path, target.path)
                try Task<Never, Never>.checkCancellation()
                try diskGate(.afterClone, path)
                try Task<Never, Never>.checkCancellation()
                guard chmod(target.path, 0o400) == 0 else { throw SnapshotError("cannot protect snapshot file: \(path)") }
                try diskGate(.beforeHash, path)
                try Task<Never, Never>.checkCancellation()
                let actual: String = try {
                    let handle = try FileHandle(forReadingFrom: target)
                    defer { try? handle.close() }
                    var hash = SHA256()
                    while let block = try handle.read(upToCount: 1024 * 1024), !block.isEmpty {
                        try Task<Never, Never>.checkCancellation()
                        try diskGate(.duringHash, path)
                        try Task<Never, Never>.checkCancellation()
                        hash.update(data: block)
                    }
                    return hash.finalize().map { String(format: "%02x", $0) }.joined()
                }()
                try Task<Never, Never>.checkCancellation()
                try diskGate(.afterHash, path)
                try Task<Never, Never>.checkCancellation()
                guard actual == files[path] else { throw SnapshotError("SHA-256 mismatch: \(path)") }
            }
            return Qwen35AffineResearchSnapshot(directory: destination)
        } catch {
            let primary = error
            do { try FileManager.default.removeItem(at: destination) }
            catch { throw SnapshotError("snapshot failed: \(primary); cleanup failed: \(error)") }
            throw primary
        }
    }
    #endif

    /// Keep this snapshot available while a request arena owns mappings. The
    /// arena retains the lease through failed drains and invalidation. This
    /// blocks `dispose()` only: it does not prevent mutation or path replacement
    /// by another same-user process, or track aliases escaping the arena.
    func retain(in arena: ExperimentalRequestArena) throws {
        lock.lock()
        guard !disposed, arenaLeases < Int.max else {
            lock.unlock()
            throw SnapshotError("snapshot disposed or lease count exhausted")
        }
        arenaLeases += 1
        lock.unlock()
        // If registration fails, deinit rolls back the reservation. The lease
        // holds the snapshot strongly for the full arena/quarantine lifetime.
        let lease = ArenaLease(snapshot: self)
        try arena.retain(lease)
    }

    private final class ArenaLease {
        let snapshot: Qwen35AffineResearchSnapshot
        init(snapshot: Qwen35AffineResearchSnapshot) { self.snapshot = snapshot }
        deinit { snapshot.releaseArenaLease() }
    }

    private func releaseArenaLease() {
        lock.lock()
        defer { lock.unlock() }
        precondition(arenaLeases > 0)
        arenaLeases -= 1
    }

    /// Caller must first establish that no future use or mapping is possible.
    public func dispose() throws {
        lock.lock()
        defer { lock.unlock() }
        guard !disposed else { return }
        guard arenaLeases == 0 else { throw SnapshotError("snapshot retained by an active or quarantined arena") }
        try FileManager.default.removeItem(at: directory)
        disposed = true
    }
}
