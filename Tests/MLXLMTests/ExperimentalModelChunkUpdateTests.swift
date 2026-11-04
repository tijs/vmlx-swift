// Copyright © 2026 Apple Inc.
import Cmlx
@testable import MLX
import MLXNN
import Testing
@testable import MLXLMCommon

struct ExperimentalModelChunkUpdateTests {
    final class TinyModel: Module {
        @ParameterInfo var a: MLXArray
        @ParameterInfo var b: MLXArray
        @ParameterInfo var placeholder: MLXArray
        override init() {
            _a.wrappedValue = MLXArray.zeros([4])
            _b.wrappedValue = MLXArray.zeros([4])
            _placeholder.wrappedValue = MLX.add(MLXArray([Float(1), 2, 3, 4]), 1, stream: .cpu)
            super.init()
        }
    }
    enum Failure: Error { case evaluate, drain }
    private let descriptors: [ExperimentalModelChunkUpdate.Descriptor] = [
        .init(key: "a", shape: [4], dtype: .float32),
        .init(key: "b", shape: [4], dtype: .bfloat16),
    ]

    @Test("one reusable slot updates only named tensors with BF16 passthrough")
    func update() throws {
        let gate = ExperimentalRequestArenaGate()
        let stream = MLX.Stream.cpu
        var chunks = 0
        var copied: [Float] = []
        try ExperimentalOwnedStreamRequest.run(gate: gate, streams: [ExperimentalOwnedStream(stream: stream)],
            operation: { arena in
                let model = TinyModel()
                var before = true
                #expect(_mlx_array_is_available(&before, model.placeholder.ctx) == 0)
                #expect(!before)
                try ExperimentalModelChunkUpdate.update(model: model, descriptors: descriptors,
                    arena: arena, streams: [stream], conversionStream: stream, maxChunkBytes: 24,
                    loadChunk: { keys in
                        chunks += 1
                        var result: [String: MLXArray] = [:]
                        for key in keys {
                            result[key] = key == "a" ? MLXArray([Float(1), 2, 3, 4])
                                : MLXArray([Float(5), 6, 7, 8]).asType(.bfloat16, stream: .cpu)
                        }
                        return result
                    })
                #expect(model.a.dtype == .bfloat16 && model.b.dtype == .bfloat16)
                copied = model.a.asType(.float32).asArray(Float.self) + model.b.asType(.float32).asArray(Float.self)
                var after = true
                #expect(_mlx_array_is_available(&after, model.placeholder.ctx) == 0)
                #expect(!after)
            }, invalidate: {})
        #expect(chunks == 2)
        #expect(copied == [1, 2, 3, 4, 5, 6, 7, 8])
    }

    @Test("failed second chunk and failed drain retain private model and graph")
    func failure() throws {
        let gate = ExperimentalRequestArenaGate()
        let stream = MLX.Stream.cpu
        weak var retainedModel: TinyModel?
        weak var retainedOutput: MLXArray?
        var chunks = 0
        do {
            let _: Void = try gate.withRequest(operation: { arena in
                let model = TinyModel(); retainedModel = model
                try ExperimentalModelChunkUpdate.perform(model: model, descriptors: descriptors, arena: arena,
                    stream: stream, maxChunkBytes: 24, loadChunk: { keys in
                        Dictionary(uniqueKeysWithValues: keys.map { key in
                            (key, key == "a" ? MLXArray([Float(1), 2, 3, 4])
                                : MLXArray([Float(5), 6, 7, 8]).asType(.bfloat16, stream: .cpu))
                        })
                    }, evaluate: { outputs in
                        chunks += 1
                        if chunks == 2 {
                            retainedOutput = outputs.first
                            MLX.asyncEval(outputs)
                            throw Failure.evaluate
                        }
                        try MLX.checkedEval(outputs)
                    }, checkpoint: { try stream.experimentalSynchronize() })
                return .finished(())
            }, drain: { _ in throw Failure.drain }, invalidate: { _ in })
            Issue.record("failure was accepted")
        } catch let error as ExperimentalRequestArenaCleanupError {
            #expect(error.primary is Failure)
            #expect(error.cleanup is Failure)
        }
        #expect(chunks == 2)
        #expect(retainedModel != nil && retainedOutput != nil)
        try gate.recover(drain: { _ in try stream.experimentalSynchronize() }, invalidate: { _ in })
        #expect(retainedModel == nil && retainedOutput == nil)
    }

    @Test("budget and unsupported placeholders reject before loading")
    func preflight() throws {
        let gate = ExperimentalRequestArenaGate()
        var loads = 0
        #expect(throws: ExperimentalModelChunkUpdate.UpdateError.self) {
            let _: Void = try gate.withRequest(operation: { arena in
                try ExperimentalModelChunkUpdate.perform(model: TinyModel(), descriptors: descriptors,
                    arena: arena, stream: .cpu, maxChunkBytes: 23, loadChunk: { _ in loads += 1; return [:] },
                    evaluate: { _ in }, checkpoint: {})
                return .finished(())
            }, drain: { _ in }, invalidate: { _ in })
        }
        #expect(loads == 0)
    }

    @Test("cancellation after loading or evaluation stops before later boundaries")
    func cancellationBoundaries() throws {
        let gate = ExperimentalRequestArenaGate()
        var cancelled = false
        var evaluations = 0
        #expect(throws: CancellationError.self) {
            let _: Void = try gate.withRequest(operation: { arena in
                try ExperimentalModelChunkUpdate.perform(model: TinyModel(), descriptors: [descriptors[0]],
                    arena: arena, stream: .cpu, maxChunkBytes: 24, shouldCancel: { cancelled },
                    loadChunk: { _ in cancelled = true; return ["a": MLXArray([Float(1), 2, 3, 4])] },
                    evaluate: { _ in evaluations += 1 }, checkpoint: {})
                return .finished(())
            }, drain: { _ in }, invalidate: { _ in })
        }
        #expect(evaluations == 0)

        cancelled = false
        var checkpoints = 0
        #expect(throws: CancellationError.self) {
            let _: Void = try gate.withRequest(operation: { arena in
                try ExperimentalModelChunkUpdate.perform(model: TinyModel(), descriptors: [descriptors[0]],
                    arena: arena, stream: .cpu, maxChunkBytes: 24, shouldCancel: { cancelled },
                    loadChunk: { _ in ["a": MLXArray([Float(1), 2, 3, 4])] },
                    evaluate: { _ in cancelled = true }, checkpoint: { checkpoints += 1 })
                return .finished(())
            }, drain: { _ in }, invalidate: { _ in })
        }
        #expect(checkpoints == 0)

        cancelled = false
        let privateModel = TinyModel()
        #expect(throws: CancellationError.self) {
            let _: Void = try gate.withRequest(operation: { arena in
                try ExperimentalModelChunkUpdate.perform(model: privateModel, descriptors: [descriptors[0]],
                    arena: arena, stream: .cpu, maxChunkBytes: 24, shouldCancel: { cancelled },
                    loadChunk: { _ in ["a": MLXArray([Float(1), 2, 3, 4])] },
                    evaluate: { try MLX.checkedEval($0) }, checkpoint: {
                        try MLX.Stream.cpu.experimentalSynchronize(); cancelled = true
                    })
                return .finished(())
            }, drain: { _ in }, invalidate: { _ in })
        }
        #expect(privateModel.a.dtype == .float32)
    }

    @Test("cancellation between leaves quarantines partial model without further updates")
    func cancellationBetweenLeaves() throws {
        let gate = ExperimentalRequestArenaGate()
        let stream = MLX.Stream.cpu
        weak var retained: TinyModel?
        do {
            let _: Void = try gate.withRequest(operation: { arena in
                let model = TinyModel(); retained = model
                try ExperimentalModelChunkUpdate.perform(model: model, descriptors: descriptors,
                    arena: arena, stream: stream, maxChunkBytes: 32,
                    shouldCancel: { model.a.dtype == .bfloat16 }, loadChunk: { keys in
                        Dictionary(uniqueKeysWithValues: keys.map { key in
                            (key, key == "a" ? MLXArray([Float(1), 2, 3, 4])
                                : MLXArray([Float(5), 6, 7, 8]).asType(.bfloat16, stream: .cpu))
                        })
                    }, evaluate: { try MLX.checkedEval($0) }, checkpoint: { try stream.experimentalSynchronize() })
                return .finished(())
            }, drain: { _ in throw Failure.drain }, invalidate: { _ in })
            Issue.record("cancelled partial update was accepted")
        } catch let error as ExperimentalRequestArenaCleanupError {
            #expect(error.primary is CancellationError)
        }
        #expect(retained?.a.dtype == .bfloat16)
        #expect(retained?.b.dtype == .float32)
        try gate.recover(drain: { _ in try stream.experimentalSynchronize() }, invalidate: { _ in })
        #expect(retained == nil)
    }
}
