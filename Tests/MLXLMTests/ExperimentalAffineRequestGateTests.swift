// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("experimental affine request ownership", .serialized)
struct ExperimentalAffineRequestGateTests {
    enum Failure: Error, Equatable { case forward, drain, invalidate }

    final class ConcurrentState: @unchecked Sendable {
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let finished = DispatchSemaphore(value: 0)
        private let lock = NSLock()
        private var error: (any Error)?

        func record(_ error: any Error) {
            lock.lock()
            self.error = error
            lock.unlock()
        }

        func recordedError() -> (any Error)? {
            lock.lock()
            defer { lock.unlock() }
            return error
        }
    }

    @Test("successful request drains before return")
    func success() throws {
        var events: [String] = []
        let value = try ExperimentalAffineRequestGate.shared.withRequest(
            operation: { events.append("forward"); return 7 },
            drain: { events.append("drain") },
            invalidate: { events.append("invalidate") })
        #expect(value == 7)
        #expect(events == ["forward", "drain"])
    }

    @Test("failed request drains and invalidates while retaining its error")
    func failure() {
        var events: [String] = []
        do {
            let _: Int = try ExperimentalAffineRequestGate.shared.withRequest(
                operation: { events.append("forward"); throw Failure.forward },
                drain: { events.append("drain") },
                invalidate: { events.append("invalidate") })
            Issue.record("forward failure was accepted")
        } catch let error as Failure {
            #expect(error == .forward)
        } catch { Issue.record("unexpected error: \(error)") }
        #expect(events == ["forward", "drain", "invalidate"])
    }

    @Test("cleanup failure retains every error and gate is released")
    func cleanupFailures() throws {
        do {
            let _: Int = try ExperimentalAffineRequestGate.shared.withRequest(
                operation: { throw Failure.forward },
                drain: { throw Failure.drain },
                invalidate: { throw Failure.invalidate })
            Issue.record("cleanup failures were accepted")
        } catch let error as ExperimentalAffineRequestCleanupError {
            #expect(error.primary as? Failure == .forward)
            #expect(error.drain as? Failure == .drain)
            #expect(error.invalidate as? Failure == .invalidate)
        }
        #expect(throws: ExperimentalAffineRequestGateError.poisoned) {
            try ExperimentalAffineRequestGate.shared.withRequest(
                operation: { 1 }, drain: {}, invalidate: {})
        }
        #expect(throws: Failure.drain) {
            try ExperimentalAffineRequestGate.shared.recover { throw Failure.drain }
        }
        #expect(throws: ExperimentalAffineRequestGateError.poisoned) {
            try ExperimentalAffineRequestGate.shared.withRequest(
                operation: { 1 }, drain: {}, invalidate: {})
        }
        try ExperimentalAffineRequestGate.shared.recover(verifyCleanup: {})
        let next = try ExperimentalAffineRequestGate.shared.withRequest(
            operation: { 5 }, drain: {}, invalidate: {})
        #expect(next == 5)
    }

    @Test("drain failure after successful work still invalidates")
    func drainFailure() throws {
        var invalidations = 0
        do {
            _ = try ExperimentalAffineRequestGate.shared.withRequest(
                operation: { 5 }, drain: { throw Failure.drain },
                invalidate: { invalidations += 1 })
            Issue.record("drain failure was accepted")
        } catch let error as ExperimentalAffineRequestCleanupError {
            #expect(error.primary == nil)
            #expect(error.drain as? Failure == .drain)
            #expect(error.invalidate == nil)
        } catch { Issue.record("unexpected error: \(error)") }
        #expect(invalidations == 1)
        try ExperimentalAffineRequestGate.shared.recover(verifyCleanup: {})
    }

    @Test("cancellation while draining invalidates instead of returning output")
    func cancellationDuringDrain() {
        var cancelled = false
        var invalidations = 0
        #expect(throws: CancellationError.self) {
            try ExperimentalAffineRequestGate.shared.withRequest(
                shouldCancel: { cancelled }, operation: { 5 },
                drain: { cancelled = true }, invalidate: { invalidations += 1 })
        }
        #expect(invalidations == 1)
    }

    @Test("another thread fails promptly while gate is held")
    func contention() throws {
        let state = ConcurrentState()
        DispatchQueue.global().async {
            defer { state.finished.signal() }
            do {
                _ = try ExperimentalAffineRequestGate.shared.withRequest(
                    operation: {
                        state.entered.signal()
                        guard state.release.wait(timeout: .now() + 3) == .success else {
                            throw Failure.forward
                        }
                        return 1
                    }, drain: {}, invalidate: {})
            } catch { state.record(error) }
        }
        guard state.entered.wait(timeout: .now() + 3) == .success else {
            state.release.signal()
            Issue.record("first request did not enter")
            return
        }
        defer { state.release.signal() }
        do {
            _ = try ExperimentalAffineRequestGate.shared.withRequest(
                operation: { 2 }, drain: {}, invalidate: {})
            Issue.record("overlapping request was accepted")
        } catch let error as ExperimentalAffineRequestGateError {
            #expect(error == .busy)
        }
        // The owner is still holding the gate until it completes its work.
        #expect(state.finished.wait(timeout: .now() + 0.01) == .timedOut)
        state.release.signal()
        #expect(state.finished.wait(timeout: .now() + 3) == .success)
        #expect(state.recordedError() == nil)
        _ = try ExperimentalAffineRequestGate.shared.withRequest(
            operation: { 3 }, drain: {}, invalidate: {})
    }
}
