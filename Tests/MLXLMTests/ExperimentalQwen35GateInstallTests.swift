// Copyright © 2026 Apple Inc.
import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXLMCommon
@testable import MLXVLM

struct ExperimentalQwen35GateInstallTests {
    enum Failure: Error { case eval, drain }
    private func model() throws -> Qwen35 {
        let json = """
        {"model_type":"qwen3_5_moe","text_config":{"hidden_size":64,"num_hidden_layers":1,
         "num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"vocab_size":128,
         "linear_num_value_heads":2,"linear_num_key_heads":2,"linear_key_head_dim":16,
         "linear_value_head_dim":16,"full_attention_interval":1,"num_experts":2,
         "num_experts_per_tok":1,"moe_intermediate_size":64,"shared_expert_intermediate_size":64}}
        """
        return try Qwen35(JSONDecoder().decode(Qwen35Configuration.self, from: Data(json.utf8)), requesting: [.text])
    }
    private func triplet(rows: Int) -> ExperimentalQwen35GateInstall.Triplet {
        let data = (0..<(rows * 64)).map { Float(($0 * 7) % 47 - 29) / 32 }
        let dense = MLXArray(data, [rows, 64]).asType(.bfloat16, stream: .cpu)
        let q = MLX.quantized(dense, groupSize: 64, bits: 8, mode: .affine, stream: .cpu)
        return .init(weight: q.wq, scales: q.scales.asType(.bfloat16, stream: .cpu),
            biases: q.biases!.asType(.bfloat16, stream: .cpu))
    }
    @Test("q8 dense gate install matches current generic loader BF16 results")
    func loaderParity() throws {
        try MLXMetalTestLock.withLock {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let pristine = try model()
            var weights = Dictionary(uniqueKeysWithValues: pristine.parameters().flattened())
            let sources = [triplet(rows: 2), triplet(rows: 1)]
            let paths = ["language_model.model.layers.0.mlp.gate", "language_model.model.layers.0.mlp.shared_expert_gate"]
            for (path, source) in zip(paths, sources) {
                weights[path + ".weight"] = source.weight
                weights[path + ".scales"] = source.scales
                weights[path + ".biases"] = source.biases
            }
            try MLX.save(arrays: weights, metadata: ["format": "mlx"], url: root.appendingPathComponent("model.safetensors"))
            try JSONEncoder().encode(pristine.config).write(to: root.appendingPathComponent("config.json"))
            let configJSON = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("config.json"))) as? [String: Any])
            let textJSON = try #require(configJSON["text_config"] as? [String: Any])
            #expect(textJSON["hidden_size"] as? Int == 64)
            let persisted = try MLX.loadArrays(url: root.appendingPathComponent("model.safetensors"))
            var checkpointSources: [ExperimentalQwen35GateInstall.Triplet] = []
            for path in paths {
                let weight = try #require(persisted[path + ".weight"])
                let scales = try #require(persisted[path + ".scales"])
                let biases = try #require(persisted[path + ".biases"])
                let inference = JangLoader.inferBitWidthAndGroupSize(packedDim: weight.shape.last ?? 0,
                    numGroups: scales.shape.last ?? 0, knownGroupSize: 64, bitWidthsUsed: [4], expectedInDim: 64)
                #expect(inference.bits == 8 && inference.groupSize == 64)
                #expect(MLX.max(MLX.abs(biases.asType(.float32))).item(Float.self) > 0)
                checkpointSources.append(.init(weight: weight, scales: scales, biases: biases))
            }
            let native = try model()
            try loadWeights(modelDirectory: root, model: native, quantization: .init(groupSize: 64, bits: 4))
            let nativeParams = Dictionary(uniqueKeysWithValues: native.parameters().flattened())
            let streams = [MLX.Stream.cpu, StreamOrDevice.default.stream]
            let conversionStream = StreamOrDevice.default.stream
            let gate = ExperimentalRequestArenaGate()
            try ExperimentalOwnedStreamRequest.run(gate: gate, streams: streams.map { ExperimentalOwnedStream(stream: $0) }, operation: { arena in
                let research = try model()
                for (path, source) in zip(paths, checkpointSources) {
                    try ExperimentalQwen35GateInstall.install(model: research, path: path, layerCount: 1,
                        numExperts: 2, hiddenSize: 64, bits: 8, groupSize: 64, arena: arena,
                        streams: streams, conversionStream: conversionStream, maxBytes: 2048, load: { source })
                }
                let params = Dictionary(uniqueKeysWithValues: research.parameters().flattened())
                for path in paths {
                    let actual = try #require(params[path + ".weight"])
                    let expected = try #require(nativeParams[path + ".weight"])
                    #expect(actual.dtype == .bfloat16 && expected.dtype == .bfloat16)
                    let difference = MLX.max(MLX.abs(actual.asType(.float32) - expected.asType(.float32))).item(Float.self)
                    print("QWEN35_Q8_GATE_LOADER_PARITY path=\(path) hidden=64 inferred=8/64 max_abs=\(difference)")
                    #expect(MLX.arrayEqual(actual, expected).item(Bool.self))
                }
            }, invalidate: {})
        }
    }
    @Test("injected evaluation and drain failures retain graph and model")
    func quarantine() throws {
        let gate = ExperimentalRequestArenaGate()
        let stream = MLX.Stream.cpu
        weak var retainedModel: Qwen35?
        weak var retainedGraph: MLXArray?
        do {
            let _: Void = try gate.withRequest(operation: { arena in
                let m = try model(); retainedModel = m
                try ExperimentalQwen35GateInstall.perform(model: m,
                    path: "language_model.model.layers.0.mlp.gate", layerCount: 1, numExperts: 2,
                    hiddenSize: 64, bits: 8, groupSize: 64, arena: arena, streams: [stream],
                    conversionStream: stream, maxBytes: 2048, load: { triplet(rows: 2) },
                    evaluate: { graphs in retainedGraph = graphs.last; MLX.asyncEval(graphs); throw Failure.eval },
                    checkpoint: { try stream.experimentalSynchronize() })
                return .finished(())
            }, drain: { _ in throw Failure.drain }, invalidate: { _ in })
            Issue.record("failed gate update accepted")
        } catch let error as ExperimentalRequestArenaCleanupError {
            #expect(error.primary is Failure && error.cleanup is Failure)
        }
        #expect(retainedModel != nil && retainedGraph != nil)
        try gate.recover(drain: { _ in try stream.experimentalSynchronize() }, invalidate: { _ in })
        #expect(retainedModel == nil && retainedGraph == nil)
    }
    @Test("invalid policy, geometry and source dtype reject before evaluation")
    func rejects() throws {
        let gate = ExperimentalRequestArenaGate()
        let m = try model()
        let stream = MLX.Stream.cpu
        var evaluations = 0
        for (path, bits, experts, badDtype) in [
            ("language_model.model.layers.0.mlp.gate", 4, 2, false),
            ("language_model.model.layers.0.mlp.shared_expert.gate_proj", 8, 2, false),
            ("language_model.model.layers.0.mlp.gate", 8, 3, false),
            ("language_model.model.layers.0.mlp.gate", 8, 2, true),
        ] {
            #expect(throws: ExperimentalQwen35GateInstall.InstallError.self) {
                let _: Void = try gate.withRequest(operation: { arena in
                    try ExperimentalQwen35GateInstall.perform(model: m, path: path, layerCount: 1,
                        numExperts: experts, hiddenSize: 64, bits: bits, groupSize: 64,
                        arena: arena, streams: [stream], conversionStream: stream, maxBytes: 2048,
                        load: {
                            let src = triplet(rows: 2)
                            return badDtype ? .init(weight: src.weight, scales: src.scales.asType(.float32, stream: .cpu), biases: src.biases) : src
                        }, evaluate: { _ in evaluations += 1 }, checkpoint: {})
                    return .finished(())
                }, drain: { _ in }, invalidate: { _ in })
            }
        }
        #expect(evaluations == 0)
    }

    @Test("cancellation after update keeps private model quarantined")
    func cancellationAfterUpdate() throws {
        let gate = ExperimentalRequestArenaGate()
        let stream = MLX.Stream.cpu
        let key = "language_model.model.layers.0.mlp.gate.weight"
        weak var retained: Qwen35?
        do {
            let _: Void = try gate.withRequest(operation: { arena in
                let m = try model(); retained = m
                let weight = try #require(Dictionary(uniqueKeysWithValues: m.parameters().flattened())[key])
                #expect(weight.dtype == .float32)
                try ExperimentalQwen35GateInstall.perform(model: m,
                    path: "language_model.model.layers.0.mlp.gate", layerCount: 1, numExperts: 2,
                    hiddenSize: 64, bits: 8, groupSize: 64, arena: arena, streams: [stream],
                    conversionStream: stream, maxBytes: 2048,
                    shouldCancel: { weight.dtype == .bfloat16 }, load: { triplet(rows: 2) },
                    evaluate: { try MLX.checkedEval($0) }, checkpoint: { try stream.experimentalSynchronize() })
                return .finished(())
            }, drain: { _ in throw Failure.drain }, invalidate: { _ in })
            Issue.record("post-update cancellation was accepted")
        } catch let error as ExperimentalRequestArenaCleanupError {
            #expect(error.primary is CancellationError)
        }
        let updatedDType = retained.flatMap { Dictionary(uniqueKeysWithValues: $0.parameters().flattened())[key]?.dtype }
        #expect(updatedDType == .bfloat16)
        try gate.recover(drain: { _ in try stream.experimentalSynchronize() }, invalidate: { _ in })
        #expect(retained == nil)
    }

    @Test("failed second gate leaves first installed and retains model until recovery")
    func partialTwoGates() throws {
        let gate = ExperimentalRequestArenaGate()
        let stream = MLX.Stream.cpu
        let first = "language_model.model.layers.0.mlp.gate"
        let second = "language_model.model.layers.0.mlp.shared_expert_gate"
        weak var retained: Qwen35?
        weak var retainedGraph: MLXArray?
        do {
            let _: Void = try gate.withRequest(operation: { arena in
                let m = try model(); retained = m
                try ExperimentalQwen35GateInstall.perform(model: m, path: first, layerCount: 1, numExperts: 2,
                    hiddenSize: 64, bits: 8, groupSize: 64, arena: arena, streams: [stream],
                    conversionStream: stream, maxBytes: 2048, load: { triplet(rows: 2) },
                    evaluate: { try MLX.checkedEval($0) }, checkpoint: { try stream.experimentalSynchronize() })
                try ExperimentalQwen35GateInstall.perform(model: m, path: second, layerCount: 1, numExperts: 2,
                    hiddenSize: 64, bits: 8, groupSize: 64, arena: arena, streams: [stream],
                    conversionStream: stream, maxBytes: 2048, load: { triplet(rows: 1) },
                    evaluate: { graphs in retainedGraph = graphs.last; MLX.asyncEval(graphs); throw Failure.eval },
                    checkpoint: { try stream.experimentalSynchronize() })
                return .finished(())
            }, drain: { _ in throw Failure.drain }, invalidate: { _ in })
            Issue.record("partial two-gate failure was accepted")
        } catch let error as ExperimentalRequestArenaCleanupError {
            #expect(error.primary is Failure && error.cleanup is Failure)
        }
        #expect(retainedGraph != nil)
        if let m = retained {
            let parameters = Dictionary(uniqueKeysWithValues: m.parameters().flattened())
            #expect(parameters[first + ".weight"]?.dtype == .bfloat16)
            #expect(parameters[second + ".weight"]?.dtype == .float32)
        } else { Issue.record("private partial model was not retained") }
        try gate.recover(drain: { _ in try stream.experimentalSynchronize() }, invalidate: { _ in })
        #expect(retained == nil && retainedGraph == nil)
    }
}
