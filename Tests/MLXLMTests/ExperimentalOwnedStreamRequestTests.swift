// Copyright © 2026 Apple Inc.

import MLX
import Testing
@testable import MLXLMCommon

@Suite("experimental owned stream request")
struct ExperimentalOwnedStreamRequestTests {
    enum Failure: Error, Equatable { case stream }
    final class Owner {
        let released: () -> Void
        init(_ released: @escaping () -> Void) { self.released = released }
        deinit { released() }
    }

    @Test("both streams drain on failure and captured owner survives recovery")
    func partialDrain() throws {
        let gate = ExperimentalRequestArenaGate()
        var released = false
        var failed = true
        var calls: [Int] = []
        var invalidated = 0
        func attempt() throws {
            let owner = Owner { released = true }
            try ExperimentalOwnedStreamRequest.run(gate: gate, streams: [
                ExperimentalOwnedStream(drain: {
                    withExtendedLifetime(owner) {}
                    calls.append(1)
                    if failed { throw Failure.stream }
                }),
                ExperimentalOwnedStream(drain: { calls.append(2) }),
            ], operation: { _ in }, invalidate: { invalidated += 1 })
        }
        do {
            try attempt()
            Issue.record("drain failure was accepted")
        } catch let error as ExperimentalRequestArenaCleanupError {
            let drain = error.cleanup as? ExperimentalOwnedStreamsDrainError
            #expect(drain?.failures.count == 1)
            #expect(drain?.failures.first?.streamIndex == 0)
            #expect(drain?.failures.first?.error as? Failure == .stream)
        }
        #expect(calls == [1, 2])
        #expect(!released)
        #expect(invalidated == 0)
        failed = false
        try ExperimentalOwnedStreamRequest.recover(gate: gate)
        #expect(calls == [1, 2, 1, 2])
        #expect(invalidated == 1)
        #expect(released)
    }

    @Test("cancelled operation drains each stream and invalidates")
    func cancellation() {
        let gate = ExperimentalRequestArenaGate()
        var events: [String] = []
        do {
            try ExperimentalOwnedStreamRequest.run(gate: gate, streams: [
                ExperimentalOwnedStream(drain: { events.append("first") }),
                ExperimentalOwnedStream(drain: { events.append("second") }),
            ], operation: { _ in throw CancellationError() }, invalidate: { events.append("invalidate") })
            Issue.record("cancellation was accepted")
        } catch is CancellationError {} catch { Issue.record("unexpected error: \(error)") }
        #expect(events == ["first", "second", "invalidate"])
    }

    @Test("empty stream list fails before operation")
    func empty() {
        let gate = ExperimentalRequestArenaGate()
        var called = false
        do {
            try ExperimentalOwnedStreamRequest.run(gate: gate, streams: [],
                operation: { _ in called = true }, invalidate: {})
            Issue.record("empty stream list was accepted")
        } catch ExperimentalOwnedStreamRequestError.emptyStreams {} catch {
            Issue.record("unexpected error: \(error)")
        }
        #expect(!called)
    }
    @Test("operation registers a dynamic owner before submission")
    func dynamicOwner() throws {
        let gate = ExperimentalRequestArenaGate()
        var released = false
        var failed = true
        do {
            try ExperimentalOwnedStreamRequest.run(gate: gate,
                streams: [ExperimentalOwnedStream(drain: { if failed { throw Failure.stream } })],
                operation: { arena in try arena.retain(Owner { released = true }) },
                invalidate: {})
            Issue.record("failure was accepted")
        } catch is ExperimentalRequestArenaCleanupError {}
        #expect(!released)
        failed = false
        try ExperimentalOwnedStreamRequest.recover(gate: gate)
        #expect(released)
    }

    @Test("typed CPU stream drains through experimental synchronization")
    func cpuStream() throws {
        try MLXMetalTestLock.withLock {
            let gate = ExperimentalRequestArenaGate()
            var called = false
            try ExperimentalOwnedStreamRequest.run(gate: gate,
                streams: [ExperimentalOwnedStream(stream: .cpu)],
                operation: { _ in called = true }, invalidate: {})
            #expect(called)
        }
    }

    @Test("failed recovery invalidation keeps owners for retry")
    func recoveryInvalidationFailure() throws {
        let gate = ExperimentalRequestArenaGate()
        var released = false
        var invalidationFails = true
        func attempt() throws {
            let owner = Owner { released = true }
            try ExperimentalOwnedStreamRequest.run(gate: gate,
                streams: [ExperimentalOwnedStream(drain: { withExtendedLifetime(owner) {} })],
                operation: { _ in throw Failure.stream },
                invalidate: { if invalidationFails { throw Failure.stream } })
        }
        do { try attempt(); Issue.record("invalidation failure was accepted") }
        catch is ExperimentalRequestArenaCleanupError {}
        #expect(!released)
        #expect(throws: Failure.stream) { try ExperimentalOwnedStreamRequest.recover(gate: gate) }
        #expect(!released)
        invalidationFails = false
        try ExperimentalOwnedStreamRequest.recover(gate: gate)
        #expect(released)
    }

    @Test("task cancellation after operation still drains and invalidates")
    func taskCancellation() async {
        let result = await Task.detached { () -> (Bool, [String]) in
            let gate = ExperimentalRequestArenaGate()
            var events: [String] = []
            do {
                try ExperimentalOwnedStreamRequest.run(gate: gate,
                    streams: [ExperimentalOwnedStream(drain: { events.append("drain") })],
                    operation: { _ in withUnsafeCurrentTask { $0?.cancel() } },
                    invalidate: { events.append("invalidate") })
                return (false, events)
            } catch is CancellationError { return (true, events) }
            catch { return (false, events) }
        }.value
        #expect(result.0)
        #expect(result.1 == ["drain", "invalidate"])
    }

    @Test("multiple failed streams retain their admission indices")
    func failureIndices() throws {
        let gate = ExperimentalRequestArenaGate()
        var failed = true
        do {
            try ExperimentalOwnedStreamRequest.run(gate: gate, streams: [
                ExperimentalOwnedStream(drain: { if failed { throw Failure.stream } }),
                ExperimentalOwnedStream(drain: {}),
                ExperimentalOwnedStream(drain: { if failed { throw Failure.stream } }),
            ], operation: { _ in }, invalidate: {})
            Issue.record("failed drains were accepted")
        } catch let error as ExperimentalRequestArenaCleanupError {
            let failures = (error.cleanup as? ExperimentalOwnedStreamsDrainError)?.failures
            #expect(failures?.map(\.streamIndex) == [0, 2])
            #expect(failures?.allSatisfy { $0.error as? Failure == .stream } == true)
        }
        failed = false
        try ExperimentalOwnedStreamRequest.recover(gate: gate)
    }

}
