// Copyright © 2026 Apple Inc.

import Foundation
import Testing
@testable import MLXLMCommon

@Suite("experimental throwing token producer")
struct ExperimentalThrowingTokenTests {
    enum Failure: Error, Equatable { case prefill, decode }

    final class State: @unchecked Sendable {
        private let lock = NSLock()
        private var recorded: [ExperimentalTokenProducerOutcome] = []
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let consumerEntered = DispatchSemaphore(value: 0)
        func waitForEntry() -> Bool { entered.wait(timeout: .now() + 3) == .success }
        func waitForConsumer() -> Bool { consumerEntered.wait(timeout: .now() + 3) == .success }
        func record(_ outcome: ExperimentalTokenProducerOutcome) {
            lock.lock()
            recorded.append(outcome)
            lock.unlock()
        }
        func outcomes() -> [ExperimentalTokenProducerOutcome] {
            lock.lock()
            defer { lock.unlock() }
            return recorded
        }
    }

    struct Source: ExperimentalThrowingTokenSource {
        let failAfter: Int?
        var emitted = 0
        mutating func nextToken() throws -> Int? {
            if emitted == failAfter { throw Failure.decode }
            guard emitted < 2 else { return nil }
            emitted += 1
            return emitted
        }
    }

    struct BlockingSource: ExperimentalThrowingTokenSource {
        let state: State
        var shouldFail = false
        mutating func nextToken() throws -> Int? {
            state.entered.signal()
            state.release.wait()
            if shouldFail { throw Failure.decode }
            return 7
        }
    }

    struct YieldThenBlockSource: ExperimentalThrowingTokenSource {
        let state: State
        var first = true
        mutating func nextToken() throws -> Int? {
            if first {
                first = false
                return 7
            }
            state.entered.signal()
            state.release.wait()
            return 8
        }
    }

    @Test("successful source emits tokens then cleans up before stream finishes")
    func success() async throws {
        let state = State()
        let (stream, task) = experimentalThrowingTokenTask(
            makeSource: { Source(failAfter: nil) }, cleanup: { state.record($0) })
        var values: [Int] = []
        for try await value in stream { values.append(value) }
        #expect(values == [1, 2])
        #expect(state.outcomes() == [.finished])
        await task.value
    }

    @Test("source construction error reaches consumer without a completion")
    func prefillFailure() async {
        let state = State()
        let (stream, task) = experimentalThrowingTokenTask(
            makeSource: { throw Failure.prefill }, cleanup: { state.record($0) })
        do {
            for try await _ in stream { Issue.record("unexpected token") }
            Issue.record("expected prefill failure")
        } catch let failure as Failure {
            #expect(failure == .prefill)
        } catch { Issue.record("unexpected error: \(error)") }
        #expect(state.outcomes() == [.failed])
        await task.value
    }

    @Test("step error follows emitted tokens and skips successful cleanup")
    func decodeFailure() async {
        let state = State()
        let (stream, task) = experimentalThrowingTokenTask(
            makeSource: { Source(failAfter: 1) }, cleanup: { state.record($0) })
        var values: [Int] = []
        do {
            for try await value in stream { values.append(value) }
            Issue.record("expected decode failure")
        } catch let failure as Failure {
            #expect(failure == .decode)
        } catch { Issue.record("unexpected error: \(error)") }
        #expect(values == [1])
        #expect(state.outcomes() == [.failed])
        await task.value
    }

    @Test("cancellation waits for synchronous step and discards its token")
    func cancellation() async {
        let state = State()
        let (stream, task) = experimentalThrowingTokenTask(
            makeSource: { BlockingSource(state: state) }, cleanup: { state.record($0) })
        let began = await Task.detached { state.waitForEntry() }.value
        #expect(began)
        task.cancel()
        state.release.signal()
        await task.value
        var values: [Int] = []
        do {
            for try await value in stream { values.append(value) }
            Issue.record("expected cancellation")
        } catch is CancellationError {
        } catch { Issue.record("unexpected error: \(error)") }
        #expect(values.isEmpty)
        #expect(state.outcomes() == [.cancelled])
    }

    @Test("consumer cancellation cancels and drains the producer")
    func consumerCancellation() async {
        let state = State()
        let (stream, task) = experimentalThrowingTokenTask(
            makeSource: { BlockingSource(state: state) }, cleanup: { state.record($0) })
        let consumer = Task {
            state.consumerEntered.signal()
            var values: [Int] = []
            do { for try await value in stream { values.append(value) } }
            catch is CancellationError { }
            catch { Issue.record("unexpected error: \(error)") }
            return values
        }
        let began = await Task.detached { state.waitForEntry() && state.waitForConsumer() }.value
        #expect(began)
        consumer.cancel()
        // Let the consumer reach the stream cancellation handler before
        // unblocking a synchronous producer step.
        _ = await consumer.value
        state.release.signal()
        await task.value
        #expect(state.outcomes() == [.cancelled])
    }

    @Test("early break requires owner cancellation and drain")
    func earlyBreak() async {
        let state = State()
        let (stream, task) = experimentalThrowingTokenTask(
            makeSource: { YieldThenBlockSource(state: state) }, cleanup: { state.record($0) })
        var values: [Int] = []
        do {
            for try await value in stream { values.append(value); break }
        } catch { Issue.record("unexpected error: \(error)") }
        #expect(values == [7])
        let began = await Task.detached { state.waitForEntry() }.value
        #expect(began)
        #expect(state.outcomes().isEmpty)
        task.cancel()
        state.release.signal()
        await task.value
        #expect(state.outcomes() == [.cancelled])
    }

    @Test("step error wins over cancellation requested during the step")
    func failureDuringCancellation() async {
        let state = State()
        let (stream, task) = experimentalThrowingTokenTask(
            makeSource: { BlockingSource(state: state, shouldFail: true) },
            cleanup: { state.record($0) })
        let began = await Task.detached { state.waitForEntry() }.value
        #expect(began)
        task.cancel()
        state.release.signal()
        await task.value
        do {
            for try await _ in stream { Issue.record("unexpected token") }
            Issue.record("expected decode failure")
        } catch let failure as Failure {
            #expect(failure == .decode)
        } catch { Issue.record("unexpected error: \(error)") }
        #expect(state.outcomes() == [.failed])
    }
}
