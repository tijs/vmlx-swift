// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXLMCommon

@Suite("experimental throwing token producer", .serialized)
struct ExperimentalThrowingTokenTests {
    enum Failure: Error, Equatable { case prefill, decode }

    final class WeightlessModel: Module, LanguageModel {
        var kvHeads: [Int] { [] }
        func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            .logits(LMOutput(logits: MLXArray([Float(0), 1, 0]).reshaped([1, 1, 3])))
        }
        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            MLXArray([Float(0), 0, 1]).reshaped([1, 1, 3])
        }
    }

    final class WeightlessCacheModel: Module, LanguageModel {
        var kvHeads: [Int] { [1] }
        func newCache(parameters: GenerateParameters?) -> [KVCache] { [KVCacheSimple()] }
        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            let value = MLXArray.ones([1, 1, input.text.tokens.size, 4])
            _ = cache[0].update(keys: value, values: value)
            return .logits(LMOutput(logits: MLXArray([Float(0), 1, 0]).reshaped([1, 1, 3])))
        }
        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            MLXArray([Float(0), 0, 1]).reshaped([1, 1, 3])
        }
    }

    @Test("failed experimental forward cannot persist a mutated shared cache")
    func failedForwardDoesNotStoreCache() throws {
        try MLXMetalTestLock.withLock {
            let previousDevice = Device.defaultDevice()
            Device.setDefault(device: .cpu)
            defer { Device.setDefault(device: previousDevice) }
            func prepare() throws -> (TokenIterator, CacheCoordinator, KVCache) {
                let model = WeightlessCacheModel()
                let cache = KVCacheSimple()
                let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
                    usePagedCache: true, enableDiskCache: false, pagedBlockSize: 4,
                    maxCacheBlocks: 32, maxCacheBytes: 1_048_576, modelKey: "fallible-test"))
                var parameters = GenerateParameters(maxTokens: 2, temperature: 0, prefillStepSize: 128)
                parameters.enableCompiledDecode = false
                let input = LMInput(tokens: MLXArray((0 ..< 8).map { Int32($0 + 1) }).reshaped([1, 8]))
                let iterator = try TokenIterator(input: input, model: model, cache: [cache],
                    parameters: parameters, cacheCoordinator: coordinator)
                return (iterator, coordinator, cache)
            }
            let append: ([KVCache]?) -> Void = { cache in
                let value = MLXArray.ones([1, 1, 1, 4])
                _ = cache![0].update(keys: value, values: value)
            }
            var (success, successCoordinator, _) = try prepare()
            let token = try success.experimentalNextToken { _, cache, _ in
                append(cache)
                return LMOutput(logits: MLXArray([Float(0), 0, 1]).reshaped([1, 1, 3]))
            }
            #expect(try #require(successCoordinator.pagedCache).snapshotStats().residentBytes == 0)
            success.storeCacheAfterGeneration(generatedTokenIds: [try #require(token)],
                includeGeneratedBoundary: true)
            #expect(try #require(successCoordinator.pagedCache).snapshotStats().residentBytes > 0)

            var (failed, failedCoordinator, sharedCache) = try prepare()
            let before = sharedCache.offset
            #expect(throws: Failure.decode) {
                try failed.experimentalNextToken { _, cache, _ in
                    append(cache)
                    throw Failure.decode
                }
            }
            // The shared alias really changed: skipping cache persistence does
            // not undo the model's partial mutation; the owner must discard it.
            #expect(sharedCache.offset == before + 1)
            #expect(failed.next() == nil)
            failed.storeCacheAfterGeneration(generatedTokenIds: [1], includeGeneratedBoundary: true)
            #expect(try #require(failedCoordinator.pagedCache).snapshotStats().residentBytes == 0)
        }
    }

    final class CancelOnDecodeSampler: LogitSampler {
        private var samples = 0
        func sample(logits: MLXArray) -> MLXArray {
            samples += 1
            if samples == 2 { withUnsafeCurrentTask { $0?.cancel() } }
            return MLX.argMax(logits, axis: -1)
        }
    }

    @Test("real iterator discards a token when cancellation arrives during sampling")
    func iteratorCancellationDuringSampling() async throws {
        try await MLXMetalTestLock.withLock {
            let previousDevice = Device.defaultDevice()
            Device.setDefault(device: .cpu)
            defer { Device.setDefault(device: previousDevice) }
            let result = try await Task {
                let model = WeightlessModel()
                let input = LMInput(text: .init(tokens: MLXArray([Int32(1)])))
                let iterator = try TokenIterator(input: input, model: model, processor: nil,
                    sampler: CancelOnDecodeSampler(), maxTokens: 2)
                var adapter = ExperimentalThrowingTokenIterator(iterator: iterator) { text, cache, state in
                    model(text, cache: cache, state: state)
                }
                do {
                    _ = try adapter.nextToken()
                    return false
                } catch is CancellationError {
                    do {
                        _ = try adapter.nextToken()
                        return false
                    } catch ExperimentalDecodeError.invalidated {
                        return true
                    }
                }
            }.value
            #expect(result)
        }
    }

    @Test("real iterator preserves eager decode and propagates a failed forward")
    func iteratorBridge() throws {
        try MLXMetalTestLock.withLock {
            let previousDevice = Device.defaultDevice()
            Device.setDefault(device: .cpu)
            defer { Device.setDefault(device: previousDevice) }
            let model = WeightlessModel()
            let input = LMInput(text: .init(tokens: MLXArray([Int32(1)])))
            func makeIterator() throws -> TokenIterator {
                try TokenIterator(input: input, model: model, processor: nil,
                                  sampler: ArgMaxSampler(), maxTokens: 3)
            }
            var normal = try makeIterator()
            let expected = [normal.next(), normal.next(), normal.next(), normal.next()]
            var bridged = ExperimentalThrowingTokenIterator(iterator: try makeIterator()) { text, cache, state in
                model(text, cache: cache, state: state)
            }
            let actual = try [bridged.nextToken(), bridged.nextToken(), bridged.nextToken(), bridged.nextToken()]
            #expect(actual == expected)
            var failing = ExperimentalThrowingTokenIterator(iterator: try makeIterator()) { _, _, _ in
                throw Failure.decode
            }
            #expect(throws: Failure.decode) { try failing.nextToken() }
            #expect(throws: ExperimentalDecodeError.invalidated) { try failing.nextToken() }
        }
    }

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
