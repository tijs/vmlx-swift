// Copyright © 2026 Apple Inc.

import Foundation

public enum ExperimentalRequestArenaError: Error, Equatable {
    case busy
    case quarantined
    case sealed
}

/// Both the operation and cleanup failures are retained for diagnosis.
public struct ExperimentalRequestArenaCleanupError: Error {
    public let primary: (any Error)?
    public let cleanup: any Error
}

/// Consumers must explicitly classify early stops as cancelled so that their
/// partial request state is invalidated, even when returning a value.
public enum ExperimentalRequestArenaCompletion<Value> {
    case finished(Value)
    case cancelled(Value)
}

/// Strongly retains resources explicitly registered before GPU submission.
/// Registration is synchronized and sealed when the admitted operation returns. It does not discover resources or prevent external aliases.
public final class ExperimentalRequestArena {
    private let lock = NSLock()
    private var resources: [AnyObject] = []
    private var sealed = false
    fileprivate init() {}
    /// Register before submission. After the operation returns, registration
    /// is sealed; late or escaped registrations fail rather than race cleanup.
    public func retain(_ resource: AnyObject) throws {
        lock.lock()
        defer { lock.unlock() }
        guard !sealed else { throw ExperimentalRequestArenaError.sealed }
        resources.append(resource)
    }
    // Rejects stale aliases, but is not an in-flight lease: callers must join
    // all submitting workers before the owning operation returns or throws.
    func isOpen() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return !sealed
    }
    // Read registered owners during gate-controlled recovery.
    func registeredResource<T>(ofType type: T.Type) -> T? {
        lock.lock()
        defer { lock.unlock() }
        return resources.first { $0 is T } as? T
    }
    fileprivate func seal() {
        lock.lock()
        defer { lock.unlock() }
        sealed = true
    }
}

/// Research-only process-lifetime owner for explicitly registered resources.
/// It serializes its own users, but does not guard the existing serving APIs,
/// model aliases, or other users of the same device/stream. It cannot retain
/// unregistered MLX temporaries or prove that all GPU buffers were registered.
/// Callers must register resources before any work can reference them. The
/// operation must stop and join every worker that can register or submit work
/// before returning or throwing; sealing alone cannot stop escaped workers.
/// Callers must arrange exclusive ownership across every other path, and provide drains for all
/// streams used. This primitive is not a safe serving integration by itself.
///
/// On failed drain the arena is quarantined and invalidation is deferred: it
/// might otherwise release resources still in use by the device. On failed
/// invalidation it is also quarantined. The static owner keeps the quarantine
/// alive even if the request error is discarded. Recovery runs under the same
/// admission lock, must establish completion of every affected stream, and must
/// repair all request/model state. A failed recovery keeps the arena alive and
/// blocks further requests. A successful recovery releases it only after both
/// callbacks return. Neither cancellation nor errors can interrupt callbacks.
public final class ExperimentalRequestArenaGate: @unchecked Sendable {
    public static let shared = ExperimentalRequestArenaGate()
    private let lock = NSLock()
    private var quarantine: ExperimentalRequestArena?
    // Internal construction permits isolated tests; external research callers
    // use the process-lifetime shared owner.
    init() {}

    public func withRequest<T>(
        operation: (ExperimentalRequestArena) throws -> ExperimentalRequestArenaCompletion<T>,
        drain: (ExperimentalRequestArena) throws -> Void,
        invalidate: (ExperimentalRequestArena) throws -> Void
    ) throws -> T {
        guard lock.try() else { throw ExperimentalRequestArenaError.busy }
        defer { lock.unlock() }
        guard quarantine == nil else { throw ExperimentalRequestArenaError.quarantined }
        let arena = ExperimentalRequestArena()
        let result: Result<ExperimentalRequestArenaCompletion<T>, any Error>
        do { result = .success(try operation(arena)) }
        catch { result = .failure(error) }
        arena.seal()

        let primary: (any Error)?
        let needsInvalidation: Bool
        switch result {
        case .success(.finished): primary = nil; needsInvalidation = false
        case .success(.cancelled): primary = nil; needsInvalidation = true
        case .failure(let error): primary = error; needsInvalidation = true
        }
        do {
            try drain(arena)
        } catch {
            quarantine = arena
            throw ExperimentalRequestArenaCleanupError(primary: primary, cleanup: error)
        }
        if needsInvalidation {
            do { try invalidate(arena) }
            catch {
                quarantine = arena
                throw ExperimentalRequestArenaCleanupError(primary: primary, cleanup: error)
            }
        }
        switch try result.get() {
        case .finished(let value), .cancelled(let value): return value
        }
    }

    /// Recovery callbacks must cover the exact streams and state of the failed
    /// request. The arena remains retained if either callback throws.
    public func recover(
        drain: (ExperimentalRequestArena) throws -> Void,
        invalidate: (ExperimentalRequestArena) throws -> Void
    ) throws {
        guard lock.try() else { throw ExperimentalRequestArenaError.busy }
        defer { lock.unlock() }
        guard let arena = quarantine else { return }
        try drain(arena)
        try invalidate(arena)
        quarantine = nil
        withExtendedLifetime(arena) {}
    }
}
