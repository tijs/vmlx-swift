// Copyright © 2026 Apple Inc.

import MLX
import MLXNN
import Testing
@testable import MLXLMCommon

@Suite("experimental arena prefill", .serialized)
struct ExperimentalArenaPrefillTests {
    enum Failure: Error, Equatable { case prefill, drain }

    final class TrackedCache: KVCacheSimple {
        let onDeinit: () -> Void
        init(onDeinit: @escaping () -> Void) { self.onDeinit = onDeinit; super.init() }
        deinit { onDeinit() }
    }
    final class Model: Module, LanguageModel {
        let onCacheDeinit: () -> Void
        let onDeinit: () -> Void
        init(onCacheDeinit: @escaping () -> Void, onDeinit: @escaping () -> Void) {
            self.onCacheDeinit = onCacheDeinit
            self.onDeinit = onDeinit
            super.init()
        }
        deinit { onDeinit() }
        var kvHeads: [Int] { [1] }
        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            [TrackedCache(onDeinit: onCacheDeinit)]
        }
        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            .tokens(input.text)
        }
        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray { inputs }
    }

    @Test("prefill failure retains model and cache until failed drain is recovered")
    func quarantine() throws {
        try MLXMetalTestLock.withLock {
        var modelReleased = false
        var cacheReleased = false
        var invalidated = false
        let gate = ExperimentalRequestArenaGate()
        do {
            let _: Int = try gate.withRequest(operation: { arena in
                let model = Model(onCacheDeinit: { cacheReleased = true },
                                  onDeinit: { modelReleased = true })
                _ = try ExperimentalThrowingTokenIterator.experimentalArenaTextPrefill(
                    input: LMInput(tokens: MLXArray([Int32(1)])), model: model, arena: arena,
                    processor: nil, sampler: ArgMaxSampler(),
                    prepare: { _, cache, _ in
                        let value = MLXArray.ones([1, 1, 1, 4])
                        _ = cache[0].update(keys: value, values: value)
                        #expect(cache[0].offset == 1)
                        throw Failure.prefill
                    },
                    forward: { _, _, _ in throw Failure.prefill })
                return .finished(1)
            }, drain: { _ in throw Failure.drain },
            invalidate: { _ in invalidated = true })
            Issue.record("prefill failure was accepted")
        } catch let error as ExperimentalRequestArenaCleanupError {
            #expect(error.primary as? Failure == .prefill)
            #expect(error.cleanup as? Failure == .drain)
        }
        #expect(!modelReleased)
        #expect(!cacheReleased)
        #expect(!invalidated)
        #expect(throws: Failure.drain) {
            try gate.recover(drain: { _ in throw Failure.drain }, invalidate: { _ in })
        }
        #expect(!modelReleased)
        #expect(!cacheReleased)
        try gate.recover(drain: { _ in }, invalidate: { _ in invalidated = true })
        #expect(invalidated)
        #expect(modelReleased)
        #expect(cacheReleased)
        }
    }
    final class Sentinel {
        let onDeinit: () -> Void
        init(_ onDeinit: @escaping () -> Void) { self.onDeinit = onDeinit }
        deinit { onDeinit() }
    }

    @Test("decode failure retains iterator owner until drain recovery")
    func decodeQuarantine() throws {
        try MLXMetalTestLock.withLock {
        var released = false
        let gate = ExperimentalRequestArenaGate()
        do {
            let _: Int = try gate.withRequest(operation: { arena in
                let sentinel = Sentinel { released = true }
                let model = Model(onCacheDeinit: {}, onDeinit: {})
                let iterator = try ExperimentalThrowingTokenIterator.experimentalArenaTextPrefill(
                    input: LMInput(tokens: MLXArray([Int32(1)])), model: model, arena: arena,
                    processor: nil, sampler: ArgMaxSampler(),
                    prepare: { _, _, _ in
                        .logits(LMOutput(logits: MLXArray([Float(0), 1]).reshaped([1, 1, 2])))
                    }, forward: { _, _, _ in
                        withExtendedLifetime(sentinel) {}
                        throw Failure.prefill
                    })
                _ = try iterator.nextToken()
                return .finished(1)
            }, drain: { _ in throw Failure.drain }, invalidate: { _ in })
            Issue.record("decode failure was accepted")
        } catch let error as ExperimentalRequestArenaCleanupError {
            #expect(error.primary as? Failure == .prefill)
            #expect(error.cleanup as? Failure == .drain)
        }
        #expect(!released)
        try gate.recover(drain: { _ in #expect(!released) }, invalidate: { _ in })
        #expect(released)
        }
    }

}
