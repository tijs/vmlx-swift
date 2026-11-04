// Copyright © 2026 Apple Inc.
import MLX
import MLXNN
import Testing
@testable import MLXLMCommon

struct ExperimentalAffineLeafInstallTests {
    final class TinyModel: Module {
        @ModuleInfo var linear: Linear
        @ModuleInfo var embedding: Embedding
        override init() {
            _linear.wrappedValue = Linear(64, 8, bias: false)
            _embedding.wrappedValue = Embedding(embeddingCount: 8, dimensions: 64)
            super.init()
        }
    }
    final class Router: Module {
        @ModuleInfo var gate: Linear
        @ModuleInfo(key: "shared_expert_gate") var sharedGate: Linear
        override init() {
            _gate.wrappedValue = Linear(64, 8, bias: false)
            _sharedGate.wrappedValue = Linear(64, 1, bias: false)
            super.init()
        }
    }
    final class RouterModel: Module {
        @ModuleInfo var mlp: Router
        override init() { _mlp.wrappedValue = Router(); super.init() }
    }
    final class MalformedModel: Module {
        @ModuleInfo var linear: Linear
        @ModuleInfo var embedding: Embedding
        override init() {
            _linear.wrappedValue = Linear(weight: MLXArray(Float(1)))
            _embedding.wrappedValue = Embedding(weight: MLXArray(Float(1)))
            super.init()
        }
    }
    enum Failure: Error { case evaluation, drain }
    private func triplet(seed: Int) -> ExperimentalAffineLeafInstall.Triplet {
        let dense = MLXArray((0..<512).map { Float(($0 * 7 + seed) % 37 - 18) / 32 }, [8, 64]).asType(.bfloat16, stream: .cpu)
        let q = MLX.quantized(dense, groupSize: 64, bits: 4, mode: .affine, stream: .cpu)
        return .init(weight: q.wq, scales: q.scales.asType(.bfloat16, stream: .cpu), biases: q.biases!.asType(.bfloat16, stream: .cpu))
    }
    @Test("owned q4 linear and embedding installs match native quantized modules")
    func parity() throws {
        try MLXMetalTestLock.withLock {
            let gate = ExperimentalRequestArenaGate()
            let stream = MLX.Stream.cpu
            let streams = [stream, StreamOrDevice.default.stream]
            let linearSource = triplet(seed: 1)
            let embeddingSource = triplet(seed: 2)
            var linearEqual = false, embeddingEqual = false
            try ExperimentalOwnedStreamRequest.run(gate: gate, streams: streams.map { ExperimentalOwnedStream(stream: $0) }, operation: { arena in
                let model = TinyModel()
                try ExperimentalAffineLeafInstall.install(model: model, path: "linear", arena: arena,
                    streams: streams, maxBytes: 512, load: { linearSource })
                try ExperimentalAffineLeafInstall.install(model: model, path: "embedding", arena: arena,
                    streams: streams, maxBytes: 512, load: { embeddingSource })
                let nativeLinear = QuantizedLinear(weight: linearSource.weight, bias: nil,
                    scales: linearSource.scales, biases: linearSource.biases, groupSize: 64, bits: 4)
                let nativeEmbedding = QuantizedEmbedding(weight: embeddingSource.weight,
                    scales: embeddingSource.scales, biases: embeddingSource.biases, groupSize: 64, bits: 4)
                let input = MLXArray([Float](repeating: 0.25, count: 64), [1, 64]).asType(.bfloat16)
                let indices = MLXArray([Int32(0), 3])
                linearEqual = MLX.arrayEqual(model.linear(input), nativeLinear(input)).item(Bool.self)
                embeddingEqual = MLX.arrayEqual(model.embedding(indices), nativeEmbedding(indices)).item(Bool.self)
            }, invalidate: {})
            #expect(linearEqual && embeddingEqual)
        }
    }

    @Test("submitted triplet and private model survive evaluation and drain failures")
    func quarantine() throws {
        let gate = ExperimentalRequestArenaGate()
        let stream = MLX.Stream.cpu
        weak var modelRef: TinyModel?
        weak var weightRef: MLXArray?
        do {
            let _: Void = try gate.withRequest(operation: { arena in
                let model = TinyModel(); modelRef = model
                try ExperimentalAffineLeafInstall.perform(model: model, path: "linear", arena: arena,
                    streams: [stream], maxBytes: 512, load: {
                        let source = triplet(seed: 7); weightRef = source.weight; return source
                    }, evaluate: { arrays in MLX.asyncEval(arrays); throw Failure.evaluation },
                    checkpoint: { try stream.experimentalSynchronize() })
                return .finished(())
            }, drain: { _ in throw Failure.drain }, invalidate: { _ in })
            Issue.record("failed install accepted")
        } catch let error as ExperimentalRequestArenaCleanupError {
            #expect(error.primary is Failure && error.cleanup is Failure)
        }
        #expect(modelRef != nil && weightRef != nil)
        try gate.recover(drain: { _ in try stream.experimentalSynchronize() }, invalidate: { _ in })
        #expect(modelRef == nil && weightRef == nil)
    }

    @Test("cancellation and unsupported triplets fail before installation")
    func rejects() throws {
        let gate = ExperimentalRequestArenaGate()
        let stream = MLX.Stream.cpu
        let model = TinyModel()
        var cancel = false
        #expect(throws: CancellationError.self) {
            let _: Void = try gate.withRequest(operation: { arena in
                try ExperimentalAffineLeafInstall.perform(model: model, path: "linear", arena: arena,
                    streams: [stream], maxBytes: 512, shouldCancel: { cancel }, load: {
                        cancel = true; return triplet(seed: 1)
                    }, evaluate: { _ in Issue.record("cancelled triplet evaluated") }, checkpoint: {})
                return .finished(())
            }, drain: { _ in }, invalidate: { _ in })
        }
        #expect(type(of: model.linear) == Linear.self)
        #expect(throws: ExperimentalAffineLeafInstall.InstallError.self) {
            let _: Void = try gate.withRequest(operation: { arena in
                try ExperimentalAffineLeafInstall.perform(model: model, path: "linear", arena: arena,
                    streams: [stream], maxBytes: 512, load: {
                        let source = triplet(seed: 2)
                        return .init(weight: source.weight.asType(.float32, stream: .cpu),
                            scales: source.scales, biases: source.biases)
                    }, evaluate: { _ in Issue.record("malformed triplet evaluated") }, checkpoint: {})
                return .finished(())
            }, drain: { _ in }, invalidate: { _ in })
        }
        #expect(throws: ExperimentalAffineLeafInstall.InstallError.self) {
            let _: Void = try gate.withRequest(operation: { arena in
                try ExperimentalAffineLeafInstall.perform(model: model, path: "linear", arena: arena,
                    streams: [stream], maxBytes: 287, load: { triplet(seed: 1) },
                    evaluate: { _ in Issue.record("oversized triplet evaluated") }, checkpoint: {})
                return .finished(())
            }, drain: { _ in }, invalidate: { _ in })
        }
    }

    @Test("router gates and malformed pristine ranks reject before loading")
    func routerAndRank() throws {
        let gate = ExperimentalRequestArenaGate()
        let stream = MLX.Stream.cpu
        var loads = 0
        let router = RouterModel()
        let malformed = MalformedModel()
        let cases: [(Module, String)] = [(router, "mlp.gate"), (router, "mlp.shared_expert_gate"),
                                         (malformed, "linear"), (malformed, "embedding")]
        for (model, path) in cases {
            #expect(throws: ExperimentalAffineLeafInstall.InstallError.self) {
                let _: Void = try gate.withRequest(operation: { arena in
                    try ExperimentalAffineLeafInstall.perform(model: model, path: path, arena: arena,
                        streams: [stream], maxBytes: 512, load: { loads += 1; return triplet(seed: 1) },
                        evaluate: { _ in Issue.record("invalid leaf evaluated") }, checkpoint: {})
                    return .finished(())
                }, drain: { _ in }, invalidate: { _ in })
            }
        }
        #expect(loads == 0)
    }
}
