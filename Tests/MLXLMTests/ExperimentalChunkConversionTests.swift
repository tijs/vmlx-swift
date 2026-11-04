// Copyright © 2026 Apple Inc.
import MLX
import Testing
@testable import MLXLMCommon

struct ExperimentalChunkConversionTests {
    enum Failure: Error { case evaluation, drain }

    @Test("owned conversion returns checked BF16 values in bounded chunks")
    func parity() throws {
        let gate = ExperimentalRequestArenaGate()
        let stream = MLX.Stream.cpu
        var values: [Float] = []
        try ExperimentalOwnedStreamRequest.run(gate: gate,
            streams: [ExperimentalOwnedStream(stream: stream)], operation: { arena in
                let input = ["a": MLXArray([Float(1), 2, 3, 4]), "b": MLXArray([Float(5), 6, 7, 8])]
                let converted = try ExperimentalChunkConversion.convert(input, arena: arena,
                    streams: [stream], conversionStream: stream, maxChunkBytes: 24, maxTotalBytes: 48)
                let first = try #require(converted["a"])
                let second = try #require(converted["b"])
                #expect(first.dtype == .bfloat16 && second.dtype == .bfloat16)
                values = first.asType(.float32).asArray(Float.self) + second.asType(.float32).asArray(Float.self)
            }, invalidate: {})
        #expect(values == [1, 2, 3, 4, 5, 6, 7, 8])
    }

    @Test("partial failure stops conversion and quarantines graphs until actual drain")
    func quarantine() throws {
        let gate = ExperimentalRequestArenaGate()
        let stream = MLX.Stream.cpu
        var chunks = 0
        weak var retainedOutput: MLXArray?
        do {
            let _: Void = try gate.withRequest(operation: { arena in
                let input = ["a": MLXArray([Float(1), 2, 3, 4]),
                             "b": MLXArray([Float(5), 6, 7, 8]),
                             "c": MLXArray([Float(9), 10, 11, 12])]
                _ = try ExperimentalChunkConversion.perform(input, arena: arena, stream: stream,
                    maxChunkBytes: 24, maxTotalBytes: 72, evaluate: { outputs in
                        chunks += 1
                        if chunks == 2 {
                            retainedOutput = outputs.first
                            MLX.asyncEval(outputs)
                            throw Failure.evaluation
                        }
                        try MLX.checkedEval(outputs)
                    })
                return .finished(())
            }, drain: { _ in throw Failure.drain }, invalidate: { _ in })
            Issue.record("partial failure was accepted")
        } catch let error as ExperimentalRequestArenaCleanupError {
            #expect(error.primary is Failure)
            #expect(error.cleanup is Failure)
        }
        #expect(chunks == 2)
        #expect(retainedOutput != nil)
        try gate.recover(drain: { _ in try stream.experimentalSynchronize() }, invalidate: { _ in })
        #expect(retainedOutput == nil)
    }

    @Test("budget rejection happens before any evaluation")
    func budget() throws {
        let gate = ExperimentalRequestArenaGate()
        var evaluations = 0
        #expect(throws: ExperimentalChunkConversion.ConversionError.self) {
            let _: Void = try gate.withRequest(operation: { arena in
                _ = try ExperimentalChunkConversion.perform(["a": MLXArray([Float(1), 2, 3, 4])],
                    arena: arena, stream: .cpu, maxChunkBytes: 23, maxTotalBytes: 24,
                    evaluate: { _ in evaluations += 1 })
                return .finished(())
            }, drain: { _ in }, invalidate: { _ in })
        }
        #expect(evaluations == 0)
    }
}
