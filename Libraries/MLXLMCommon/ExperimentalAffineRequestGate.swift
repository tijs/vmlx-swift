// Copyright © 2026 Apple Inc.

import Foundation

/// An experimental request cannot enter while another request owns the gate.
public enum ExperimentalAffineRequestGateError: Error, Sendable, Equatable {
    case busy
    /// A previous drain or invalidation failed; recovery is required.
    case poisoned
}

/// Cleanup failed, possibly after an independent operation failure. All errors
/// are retained so that the primary failure is not replaced by a drain error.
public struct ExperimentalAffineRequestCleanupError: Error {
    public let primary: (any Error)?
    public let drain: (any Error)?
    public let invalidate: (any Error)?
}

/// A process-wide, synchronous research admission gate. It is not wired into
/// model serving and does not impose a byte limit. The operation must evaluate
/// its returned output before returning; the mandatory drain closure must wait
/// for the exact streams used by that operation. On failure or cancellation,
/// invalidate must reset model-owned per-request state and discard every cache
/// alias. Neither callback can make externally retained aliases safe by itself.
///
/// The gate uses nonblocking admission to reject reentry without deadlock.
/// `shouldCancel` is needed when work is dispatched outside the calling Swift
/// task; cancellation cannot interrupt a synchronous operation or drain.
/// An already-cancelled or rejected call never owns the gate and invokes no
/// callbacks. Cancellation is observed at the pre-operation, post-operation,
/// and post-drain boundaries; it can race with a later successful return.
public final class ExperimentalAffineRequestGate: @unchecked Sendable {
    public static let shared = ExperimentalAffineRequestGate()
    private let lock = NSLock()
    // Accessed only while holding lock.
    private var poisoned = false

    private init() {}

    public func withRequest<T>(
        shouldCancel: () -> Bool = { false },
        operation: () throws -> T,
        drain: () throws -> Void,
        invalidate: () throws -> Void
    ) throws -> T {
        if shouldCancel() || Task<Never, Never>.isCancelled { throw CancellationError() }
        guard lock.try() else { throw ExperimentalAffineRequestGateError.busy }
        defer { lock.unlock() }
        guard !poisoned else { throw ExperimentalAffineRequestGateError.poisoned }

        var result: Result<T, any Error>
        do {
            if shouldCancel() || Task<Never, Never>.isCancelled { throw CancellationError() }
            let output = try operation()
            if shouldCancel() || Task<Never, Never>.isCancelled { throw CancellationError() }
            result = .success(output)
        } catch {
            result = .failure(error)
        }

        var drainFailure: (any Error)?
        do { try drain() } catch { drainFailure = error }
        if drainFailure != nil { poisoned = true }
        if case .success = result, shouldCancel() || Task<Never, Never>.isCancelled {
            result = .failure(CancellationError())
        }

        let primary: (any Error)?
        switch result {
        case .success: primary = nil
        case .failure(let error): primary = error
        }
        var invalidationFailure: (any Error)?
        if primary != nil || drainFailure != nil {
            do { try invalidate() } catch { invalidationFailure = error }
        }
        if invalidationFailure != nil { poisoned = true }
        if drainFailure != nil || invalidationFailure != nil {
            throw ExperimentalAffineRequestCleanupError(
                primary: primary, drain: drainFailure, invalidate: invalidationFailure)
        }
        return try result.get()
    }

    /// Recover after failed cleanup. The closure must establish that all
    /// streams from the failed request have completed and repair any failed
    /// invalidation. Throwing leaves the gate poisoned. Recovery is synchronous
    /// and holds exclusive admission for its entire duration.
    public func recover(verifyCleanup: () throws -> Void) throws {
        guard lock.try() else { throw ExperimentalAffineRequestGateError.busy }
        defer { lock.unlock() }
        guard poisoned else { return }
        try verifyCleanup()
        poisoned = false
    }
}
