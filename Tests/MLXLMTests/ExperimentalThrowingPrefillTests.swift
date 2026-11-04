// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXLMCommon

@Suite("experimental throwing text prefill", .serialized)
struct ExperimentalThrowingPrefillTests {
    enum Failure: Error, Equatable { case prefill, prime }

    final class Model: Module, LanguageModel {
        var kvHeads: [Int] { [1] }
        func newCache(parameters: GenerateParameters?) -> [KVCache] { [KVCacheSimple()] }
        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            .tokens(input.text)
        }
        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            MLXArray([Float(0), 0, 1]).reshaped([1, 1, 3])
        }
    }

    private func input() -> LMInput {
        LMInput(tokens: MLXArray([Int32(2), 1, 0]).reshaped([1, 3]))
    }

    @Test("fallible token prefill matches ordinary eager initialization")
    func tokenPrefillParity() throws {
        try MLXMetalTestLock.withLock {
            let oldDevice = Device.defaultDevice()
            Device.setDefault(device: .cpu)
            defer { Device.setDefault(device: oldDevice) }
            let model = Model()
            var ordinary = try TokenIterator(input: input(), model: model, processor: nil,
                sampler: ArgMaxSampler(), prefillStepSize: 2, maxTokens: 3)
            var discardCount = 0
            var throwing = try ExperimentalThrowingTokenIterator.experimentalTextPrefill(input: input(), model: model,
                processor: nil, sampler: ArgMaxSampler(), prefillStepSize: 2, maxTokens: 3,
                prepare: { input, _, size in
                    #expect(size == 2)
                    return .tokens(input.text)
                }, forward: { text, cache, _ in
                    LMOutput(logits: model(text.tokens, cache: cache))
                }, discardOnFailure: { discardCount += 1 })
            for _ in 0 ..< 3 {
                #expect(ordinary.next() == (try throwing.nextToken()))
            }
            #expect(try throwing.nextToken() == nil)
            #expect(discardCount == 0)
        }
    }

    @Test("logits prefill does not invoke the prime forward")
    func logitsPrefill() throws {
        try MLXMetalTestLock.withLock {
            let oldDevice = Device.defaultDevice()
            Device.setDefault(device: .cpu)
            defer { Device.setDefault(device: oldDevice) }
            let model = Model()
            var forwarded = false
            var iterator = try ExperimentalThrowingTokenIterator.experimentalTextPrefill(input: input(), model: model,
                processor: nil, sampler: ArgMaxSampler(), maxTokens: 1,
                prepare: { _, _, _ in
                    .logits(LMOutput(logits: MLXArray([Float(0), 1, 0]).reshaped([1, 1, 3])))
                }, forward: { text, cache, _ in
                    forwarded = true
                    return LMOutput(logits: model(text.tokens, cache: cache))
                }, discardOnFailure: { Issue.record("unexpected discard") })
            #expect(!forwarded)
            #expect(try iterator.nextToken() == 1)
            #expect(forwarded)
        }
    }

    @Test("prefill and prime errors drain and invoke required discard")
    func failuresDiscard() throws {
        try MLXMetalTestLock.withLock {
            let oldDevice = Device.defaultDevice()
            Device.setDefault(device: .cpu)
            defer { Device.setDefault(device: oldDevice) }
            let model = Model()
            for failAtPrime in [false, true] {
                var discarded = 0
                var observedCache: KVCache?
                do {
                    _ = try ExperimentalThrowingTokenIterator.experimentalTextPrefill(input: input(), model: model,
                        processor: nil, sampler: ArgMaxSampler(),
                        prepare: { input, cache, _ in
                            observedCache = cache[0]
                            let value = MLXArray.ones([1, 1, 1, 4])
                            _ = cache[0].update(keys: value, values: value)
                            if !failAtPrime { throw Failure.prefill }
                            return .tokens(input.text)
                        }, forward: { _, _, _ in
                            throw Failure.prime
                        }, discardOnFailure: { discarded += 1 })
                    Issue.record("expected experimental prefill failure")
                } catch let error as Failure {
                    #expect(error == (failAtPrime ? .prime : .prefill))
                }
                #expect(discarded == 1)
                // This test intentionally retains an alias to prove cleanup
                // is a caller obligation, not an implicit rollback.
                #expect(try #require(observedCache).offset == 1)
            }
        }
    }

    @Test("invalid prefill configuration is rejected before callbacks")
    func unsupportedInputs() throws {
        try MLXMetalTestLock.withLock {
            let model = Model()
            var calls = 0
            #expect(throws: ExperimentalDecodeError.unsupportedPrefill) {
                try ExperimentalThrowingTokenIterator.experimentalTextPrefill(input: input(), model: model,
                    processor: nil, sampler: ArgMaxSampler(), prefillStepSize: 0,
                    prepare: { _, _, _ in calls += 1; throw Failure.prefill },
                    forward: { _, _, _ in calls += 1; throw Failure.prime },
                    discardOnFailure: { calls += 1 })
            }
            #expect(calls == 0)
        }
    }

    @Test("task cancellation after prefill discards partially mutated cache")
    func cancellationAfterPrepare() async {
        let result = await Task.detached { () -> (Bool, Int, Int) in
            MLXMetalTestLock.withLock {
                let oldDevice = Device.defaultDevice()
                Device.setDefault(device: .cpu)
                defer { Device.setDefault(device: oldDevice) }
                let model = Model()
                var discarded = 0
                var observed: KVCache?
                do {
                    _ = try ExperimentalThrowingTokenIterator.experimentalTextPrefill(
                        input: LMInput(tokens: MLXArray([Int32(2), 1, 0]).reshaped([1, 3])),
                        model: model, processor: nil, sampler: ArgMaxSampler(),
                        prepare: { input, cache, _ in
                            observed = cache[0]
                            let value = MLXArray.ones([1, 1, 1, 4])
                            _ = cache[0].update(keys: value, values: value)
                            withUnsafeCurrentTask { $0?.cancel() }
                            return .tokens(input.text)
                        }, forward: { _, _, _ in
                            throw Failure.prime
                        }, discardOnFailure: { discarded += 1 })
                    return (false, discarded, observed?.offset ?? -1)
                } catch is CancellationError {
                    return (true, discarded, observed?.offset ?? -1)
                } catch {
                    return (false, discarded, observed?.offset ?? -1)
                }
            }
        }.value
        #expect(result.0)
        #expect(result.1 == 1)
        #expect(result.2 == 1)
    }
}
