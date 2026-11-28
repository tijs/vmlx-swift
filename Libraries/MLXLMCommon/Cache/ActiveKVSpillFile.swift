#if canImport(Darwin)
import Darwin
import Foundation

/// Request-scoped backing for immutable, equally sized KV pages.
/// The caller owns staging and must keep buffers alive until Metal has finished
/// using them. A page is readable only after its complete write has succeeded.
final class ActiveKVSpillFile: @unchecked Sendable {
    enum Failure: LocalizedError, Equatable {
        case invalidGeometry
        case invalidBuffer
        case invalidPage
        case capacityExceeded
        case unexpectedEndOfFile
        case systemCall(String, Int32)

        var errorDescription: String? {
            switch self {
            case .invalidGeometry: return "Invalid live KV spill page layout or disk allowance"
            case .invalidBuffer: return "Live KV spill buffer does not match its page layout"
            case .invalidPage: return "Live KV spill page is outside committed history"
            case .capacityExceeded: return "Live KV spill exceeded the request's disk allowance"
            case .unexpectedEndOfFile: return "Live KV spill page ended before all bytes were read"
            case .systemCall(let operation, let code):
                let message = String(cString: strerror(code))
                if operation == "mkstemp" { return "Could not open live KV spill file: \(message)" }
                return "Live KV spill \(operation) failed: \(message)"
            }
        }
    }

    let pageBytes: Int
    let capacity: Int

    private let descriptor: Int32
    private let lock = NSLock()
    private var committedPages = 0

    var pageCount: Int { lock.withLock { committedPages } }

    /// The caller supplies a request's disk budget. The file grows on demand.
    init(directory: URL, pageBytes: Int, byteLimit: Int) throws {
        guard pageBytes > 0, byteLimit >= pageBytes else {
            throw Failure.invalidGeometry
        }
        let template = directory.appendingPathComponent(".vmlx-kv-XXXXXX").path
        var path = Array(template.utf8CString)
        let fd = mkstemp(&path)
        guard fd >= 0 else { throw Failure.systemCall("mkstemp", errno) }

        // The open descriptor keeps the unlinked file alive. Remove its path
        // before writing any request data.
        guard unlink(path) == 0 else {
            let failure = Failure.systemCall("unlink", errno)
            Darwin.close(fd)
            throw failure
        }
        guard fcntl(fd, F_SETFD, FD_CLOEXEC) == 0,
              fcntl(fd, F_NOCACHE, 1) == 0 else {
            let failure = Failure.systemCall("fcntl", errno)
            Darwin.close(fd)
            throw failure
        }
        descriptor = fd
        self.pageBytes = pageBytes
        capacity = byteLimit / pageBytes
    }

    deinit { Darwin.close(descriptor) }

    @discardableResult
    func append(_ bytes: UnsafeRawBufferPointer) throws -> Int {
        guard bytes.count == pageBytes, let source = bytes.baseAddress else {
            throw Failure.invalidBuffer
        }
        return try lock.withLock {
            guard committedPages < capacity else { throw Failure.capacityExceeded }
            let page = committedPages
            try transfer("pwrite", page: page, zeroProgress: .systemCall("pwrite", EIO)) { done, remaining, offset in
                pwrite(descriptor, source.advanced(by: done), remaining, offset)
            }
            committedPages += 1
            return page
        }
    }

    func read(page: Int, into bytes: UnsafeMutableRawBufferPointer) throws {
        guard bytes.count == pageBytes, let destination = bytes.baseAddress else {
            throw Failure.invalidBuffer
        }
        try lock.withLock {
            guard (0..<committedPages).contains(page) else { throw Failure.invalidPage }
            try transfer("pread", page: page, zeroProgress: .unexpectedEndOfFile) { done, remaining, offset in
                pread(descriptor, destination.advanced(by: done), remaining, offset)
            }
        }
    }

    /// Release whole pages discarded by a request rollback. The caller handles
    /// a partially accepted tail in its staging buffer.
    func truncate(toPageCount count: Int) throws {
        try lock.withLock {
            guard (0...committedPages).contains(count) else { throw Failure.invalidPage }
            guard ftruncate(descriptor, off_t(count * pageBytes)) == 0 else {
                throw Failure.systemCall("ftruncate", errno)
            }
            committedPages = count
        }
    }

    private func transfer(
        _ operation: String, page: Int, zeroProgress: Failure,
        perform: (Int, Int, off_t) -> Int
    ) throws {
        var done = 0
        while done < pageBytes {
            try Task.checkCancellation()
            let count = perform(done, pageBytes - done, off_t(page * pageBytes + done))
            if count < 0 {
                let error = errno
                if error == EINTR { continue }
                throw Failure.systemCall(operation, error)
            }
            guard count > 0 else { throw zeroProgress }
            done += count
        }
    }
}
#endif
