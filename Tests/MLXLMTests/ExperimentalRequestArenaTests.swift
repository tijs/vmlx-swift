// Copyright © 2026 Apple Inc.

import Foundation
import Testing
@testable import MLXLMCommon

@Suite("experimental request arena", .serialized)
struct ExperimentalRequestArenaTests {
    enum Failure: Error, Equatable { case operation, drain, invalidate }
    final class Resource {
        let onDeinit: () -> Void
        init(_ onDeinit: @escaping () -> Void) { self.onDeinit = onDeinit }
        deinit { onDeinit() }
    }

    @Test("failed drain quarantines resources through failed and successful recovery")
    func failedDrain() throws {
        let gate = ExperimentalRequestArenaGate()
        var destroyed = false
        var invalidated = false
        do {
            let _: Int = try gate.withRequest(
                operation: { arena in
                    try arena.retain(Resource { destroyed = true })
                    throw Failure.operation
                }, drain: { _ in throw Failure.drain },
                invalidate: { _ in invalidated = true })
            Issue.record("request unexpectedly succeeded")
        } catch let error as ExperimentalRequestArenaCleanupError {
            #expect(error.primary as? Failure == .operation)
            #expect(error.cleanup as? Failure == .drain)
        }
        #expect(!destroyed)
        #expect(!invalidated)
        #expect(throws: ExperimentalRequestArenaError.quarantined) {
            try gate.withRequest(operation: { _ in .finished(1) }, drain: { _ in }, invalidate: { _ in })
        }
        #expect(throws: Failure.drain) {
            try gate.recover(drain: { _ in throw Failure.drain }, invalidate: { _ in })
        }
        #expect(!destroyed)
        #expect(throws: Failure.invalidate) {
            try gate.recover(drain: { _ in }, invalidate: { _ in throw Failure.invalidate })
        }
        #expect(!destroyed)
        try gate.recover(drain: { _ in #expect(!destroyed) },
            invalidate: { _ in #expect(!destroyed); invalidated = true })
        #expect(invalidated)
        #expect(destroyed)
        #expect(try gate.withRequest(operation: { _ in .finished(1) }, drain: { _ in }, invalidate: { _ in }) == 1)
    }

    @Test("failed invalidation quarantines even after a successful drain")
    func failedInvalidation() throws {
        let gate = ExperimentalRequestArenaGate()
        var destroyed = false
        do {
            let _: Int = try gate.withRequest(
                operation: { arena in try arena.retain(Resource { destroyed = true }); throw Failure.operation },
                drain: { _ in }, invalidate: { _ in throw Failure.invalidate })
            Issue.record("request unexpectedly succeeded")
        } catch let error as ExperimentalRequestArenaCleanupError {
            #expect(error.primary as? Failure == .operation)
            #expect(error.cleanup as? Failure == .invalidate)
        }
        #expect(!destroyed)
        try gate.recover(drain: { _ in }, invalidate: { _ in })
        #expect(destroyed)
    }

    @Test("request and recovery reject reentry while admitted")
    func reentry() throws {
        let gate = ExperimentalRequestArenaGate()
        _ = try gate.withRequest(operation: { _ in
            #expect(throws: ExperimentalRequestArenaError.busy) {
                try gate.withRequest(operation: { _ in .finished(1) }, drain: { _ in }, invalidate: { _ in })
            }
            #expect(throws: ExperimentalRequestArenaError.busy) {
                try gate.recover(drain: { _ in }, invalidate: { _ in })
            }
            return .finished(1)
        }, drain: { _ in }, invalidate: { _ in })
    }
    @Test("early stop invalidates before returning its partial value")
    func earlyStop() throws {
        var events: [String] = []
        let value = try ExperimentalRequestArenaGate().withRequest(
            operation: { _ in events.append("operation"); return .cancelled(4) },
            drain: { _ in events.append("drain") },
            invalidate: { _ in events.append("invalidate") })
        #expect(value == 4)
        #expect(events == ["operation", "drain", "invalidate"])
    }

    @Test("registration is sealed before cleanup and after escape")
    func sealed() throws {
        var escaped: ExperimentalRequestArena?
        _ = try ExperimentalRequestArenaGate().withRequest(
            operation: { arena in escaped = arena; return .finished(1) },
            drain: { arena in
                #expect(throws: ExperimentalRequestArenaError.sealed) {
                    try arena.retain(Resource {})
                }
            }, invalidate: { _ in })
        #expect(throws: ExperimentalRequestArenaError.sealed) {
            try escaped?.retain(Resource {})
        }
    }

    final class Coordination: @unchecked Sendable {
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let done = DispatchSemaphore(value: 0)
        private let lock = NSLock()
        private var workerError: (any Error)?
        func record(_ error: any Error) {
            lock.lock()
            defer { lock.unlock() }
            workerError = error
        }
        func error() -> (any Error)? {
            lock.lock()
            defer { lock.unlock() }
            return workerError
        }
    }

    @Test("another thread cannot enter or recover during a request")
    func contention() throws {
        let gate = ExperimentalRequestArenaGate()
        let state = Coordination()
        DispatchQueue.global().async {
            defer { state.done.signal() }
            do {
                _ = try gate.withRequest(operation: { _ in
                    state.entered.signal()
                    if state.release.wait(timeout: .now() + 5) == .timedOut {
                        throw Failure.operation
                    }
                    return .finished(1)
                }, drain: { _ in }, invalidate: { _ in })
            } catch { state.record(error) }
        }
        defer { state.release.signal() }
        guard state.entered.wait(timeout: .now() + 5) == .success else {
            Issue.record("worker did not enter request in time")
            return
        }
        #expect(throws: ExperimentalRequestArenaError.busy) {
            try gate.withRequest(operation: { _ in .finished(1) }, drain: { _ in }, invalidate: { _ in })
        }
        #expect(throws: ExperimentalRequestArenaError.busy) {
            try gate.recover(drain: { _ in }, invalidate: { _ in })
        }
        state.release.signal()
        state.done.wait()
    }

    @Test("recovery holds exclusive admission until resource release")
    func recoveryContention() throws {
        let gate = ExperimentalRequestArenaGate()
        let state = Coordination()
        final class Lifetime: @unchecked Sendable {
            private let lock = NSLock()
            private var released = false
            func markReleased() { lock.lock(); released = true; lock.unlock() }
            func isReleased() -> Bool { lock.lock(); defer { lock.unlock() }; return released }
        }
        let lifetime = Lifetime()
        do {
            let _: Int = try gate.withRequest(operation: { arena in
                try arena.retain(Resource { lifetime.markReleased() })
                return .finished(1)
            }, drain: { _ in throw Failure.drain }, invalidate: { _ in })
            Issue.record("failed drain was accepted")
        } catch is ExperimentalRequestArenaCleanupError {}
        #expect(!lifetime.isReleased())
        DispatchQueue.global().async {
            defer { state.done.signal() }
            do {
                try gate.recover(drain: { _ in
                    state.entered.signal()
                    if state.release.wait(timeout: .now() + 5) == .timedOut {
                        throw Failure.drain
                    }
                }, invalidate: { _ in })
            } catch { state.record(error) }
        }
        defer { state.release.signal() }
        guard state.entered.wait(timeout: .now() + 5) == .success else {
            Issue.record("recovery did not start in time")
            return
        }
        #expect(!lifetime.isReleased())
        #expect(throws: ExperimentalRequestArenaError.busy) {
            try gate.recover(drain: { _ in }, invalidate: { _ in })
        }
        #expect(throws: ExperimentalRequestArenaError.busy) {
            try gate.withRequest(operation: { _ in .finished(1) }, drain: { _ in }, invalidate: { _ in })
        }
        state.release.signal()
        guard state.done.wait(timeout: .now() + 5) == .success else {
            Issue.record("recovery did not finish in time")
            return
        }
        if let error = state.error() { Issue.record("recovery failed: \(error)") }
        #expect(lifetime.isReleased())
    }

}
