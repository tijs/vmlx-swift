import Foundation
import MLX
import MLXLMCommon
import XCTest
@testable import MLXLLM

/// Explicit supervised numerical dump, never part of ordinary unit execution.
final class InklingFullWeightProbeTests: XCTestCase {
    func testPinnedFullWeightLogitDump() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["VMLX_INKLING_FULL_PROBE"] == "1" else {
            throw XCTSkip("Full checkpoint requires an exclusive supervised session")
        }
        let modelPath = try XCTUnwrap(env["VMLX_INKLING_PROBE_MODEL"])
        let tokenPath = try XCTUnwrap(env["VMLX_INKLING_PROBE_TOKENS"])
        let outputPath = try XCTUnwrap(env["VMLX_INKLING_PROBE_OUTPUT"])
        _ = try XCTUnwrap(env["BENCH_ADMITTED_CONFIG_SHA256"])
        let output = URL(fileURLWithPath: outputPath)
        guard !FileManager.default.fileExists(atPath: output.path) else {
            throw NSError(domain: "InklingProbe", code: 1, userInfo: [NSLocalizedDescriptionKey: "Choose a fresh output directory"])
        }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        var record: [String: Any] = ["status": "loading", "scope": "Native numerical dump only; reference comparison is separate"]
        func checkpoint() throws {
            try JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys])
                .write(to: output.appendingPathComponent("record.json"), options: .atomic)
        }
        try checkpoint()
        try Device.withDefaultDevice(.gpu) {
            let directory = URL(fileURLWithPath: modelPath)
            let config = try JSONDecoder().decode(Inkling2LoopConfiguration.self,
                from: Data(contentsOf: directory.appendingPathComponent("config.json")))
            let ids = try JSONDecoder().decode([Int].self, from: Data(contentsOf: URL(fileURLWithPath: tokenPath)))
            guard !ids.isEmpty, ids.count + 8 <= 4096,
                ids.allSatisfy({ 0 <= $0 && $0 < config.text.unpaddedVocab }) else {
                throw NSError(domain: "InklingProbe", code: 2)
            }
            Memory.cacheLimit = 536_870_912
            Memory.memoryLimit = 24_051_816_857
            let workingSet = GPU.maxRecommendedWorkingSetBytes() ?? Int.max
            let target = try XCTUnwrap(modelResidencyWiredTarget(modelBytes: 19_526_043_704,
                workingSet: workingSet, physicalMemory: Int(ProcessInfo.processInfo.physicalMemory)))
            guard setModelResidencyWiredLimit(target) else {
                throw NSError(domain: "InklingProbe", code: 3)
            }
            let model = Inkling2LoopModel(config)
            try loadWeights(modelDirectory: directory, model: model)
            eval(model)
            let tokens = MLXArray(ids.map(Int32.init), [1, ids.count])
            let cache = model.newCache(parameters: nil)
            var logits = model(tokens, cache: cache)[0..., -1, 0...]
            eval(logits)
            if env["VMLX_INKLING_TRACE_LAYERS"] == "1" {
                // Test-only transparent replay of the native schedule. Validate
                // its final result against the ordinary public forward above.
                let text = model.language.model, t = config.text
                let traceCache = model.newCache(parameters: nil)
                var x = text.embedNorm(text.embed(tokens)), blocks: [MLXArray] = []
                var preLoop: (MLXArray, [MLXArray])?, previous: ([MLXArray], MLXArray)?
                var captures: [String: MLXArray] = [:]
                for (visit, physical) in t.schedule.enumerated() {
                    if physical == t.loop.coreStart {
                        if let saved = preLoop {
                            previous = (Array(blocks.dropFirst(saved.1.count + 1)), x)
                            x = saved.0; blocks = saved.1
                        } else { preLoop = (x, blocks) }
                    }
                    let messages: MLXArray?
                    if let previous, physical < t.loop.coreEnd {
                        messages = text.messageInput(previous.0, previous.1, layer: physical - t.loop.coreStart)
                    } else { messages = nil }
                    captures[String(format: "visit-%02d", visit)] = x
                    x = try text.layers[physical](x, blocks: &blocks,
                        cache: traceCache[visit] as? Inkling2VisitCache, messages: messages)
                }
                let prehead = text.norm(text.finalInput(blocks, x)[.ellipsis, 0, 0...])
                captures["prehead-f32"] = prehead
                let rounded = prehead.asType(text.embedNorm.weight.dtype)
                captures["visit-48"] = rounded
                let head = model.language.head.weight.asType(.float32).T
                let cap = config.text.logitCap
                func project(_ hidden: MLXArray) -> MLXArray {
                    let raw = matmul(hidden.asType(.float32), head)
                    return (tanh(raw / cap) * cap)[0..., -1, 0..<model.vocabularySize]
                }
                let replay = project(rounded)
                eval(replay)
                record["trace_replay_max_abs_error"] = max(abs(replay - logits)).item(Float.self)
                XCTAssertLessThan(max(abs(replay - logits)).item(Float.self), 1e-5)
                captures["logits-f32-prehead"] = project(prehead)
                try save(arrays: captures, url: output.appendingPathComponent("layer-captures.safetensors"))
            }
            let single = model.newCache(parameters: nil)
            var chunked: MLXArray?
            for id in ids {
                chunked = model(MLXArray([Int32(id)], [1,1]), cache: single)[0..., -1, 0...]
                eval(chunked!)
            }
            try save(arrays: ["logits": chunked!], url: output.appendingPathComponent("single-token-prefill.safetensors"))
            record["prefill_max_abs_error"] = max(abs(logits - chunked!)).item(Float.self)
            record["single_token_prefill_greedy"] = argMax(chunked!, axis: -1).item(Int.self)
            record["prefill_offsets"] = cache.map(\.offset)
            record["single_token_offsets"] = single.map(\.offset)
            guard cache.count == 48, single.count == 48,
                (cache + single).allSatisfy({ $0.offset == ids.count && $0.state.count == 6 }) else {
                throw NSError(domain: "InklingProbe", code: 4)
            }
            var greedy: [Int] = []
            record["status"] = "running"
            for step in 0..<8 {
                eval(logits)
                guard all(isFinite(logits)).item(Bool.self) else {
                    throw NSError(domain: "InklingProbe", code: 5)
                }
                let next = argMax(logits, axis: -1).item(Int.self)
                greedy.append(next)
                try save(arrays: ["logits": logits], url: output.appendingPathComponent(String(format: "logits-%02d.safetensors", step)))
                record["greedy_ids"] = greedy
                record["cache_offsets"] = cache.map(\.offset)
                try checkpoint()
                if step < 7 {
                    logits = model(MLXArray([Int32(next)], [1,1]), cache: cache)[0..., -1, 0...]
                }
            }
            record["status"] = "completed"
            try checkpoint()
        }
    }
}
