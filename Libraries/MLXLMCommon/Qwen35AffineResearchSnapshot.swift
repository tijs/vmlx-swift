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

    public let directory: URL
    private let lock = NSLock()
    private var disposed = false
    private init(directory: URL) { self.directory = directory }

    /// `files` maps every required relative path to its independently pinned
    /// lowercase SHA-256 digest. No files are discovered implicitly.
    public static func create(source: URL, stagingParent: URL, files: [String: String]) throws -> Qwen35AffineResearchSnapshot {
        #if canImport(Darwin) && canImport(CryptoKit)
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
                guard clonefile(sourcePath.path, target.path, 0) == 0 else {
                    throw SnapshotError("APFS clone failed for \(path): errno \(errno)")
                }
                guard chmod(target.path, 0o400) == 0 else { throw SnapshotError("cannot protect snapshot file: \(path)") }
                let handle = try FileHandle(forReadingFrom: target)
                defer { try? handle.close() }
                var hash = SHA256()
                while let block = try handle.read(upToCount: 1024 * 1024), !block.isEmpty {
                    hash.update(data: block)
                }
                let actual = hash.finalize().map { String(format: "%02x", $0) }.joined()
                guard actual == files[path] else { throw SnapshotError("SHA-256 mismatch: \(path)") }
            }
            return Qwen35AffineResearchSnapshot(directory: destination)
        } catch {
            let primary = error
            do { try FileManager.default.removeItem(at: destination) }
            catch { throw SnapshotError("snapshot failed: \(primary); cleanup failed: \(error)") }
            throw primary
        }
        #else
        throw SnapshotError("APFS clone snapshots require Darwin and CryptoKit")
        #endif
    }

    /// Caller must first establish that no future use or mapping is possible.
    public func dispose() throws {
        lock.lock()
        defer { lock.unlock() }
        guard !disposed else { return }
        try FileManager.default.removeItem(at: directory)
        disposed = true
    }
}
