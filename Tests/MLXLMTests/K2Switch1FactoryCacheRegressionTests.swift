// Synthetic loader/cache proof only. No claim of trained-model coherence or full-bundle support.
import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class K2Switch1FactoryCacheRegressionTests: XCTestCase {
    // Deliberately tiny append-only tokenizer. Native K2/Jinja semantics are tested
    // separately; this fixture isolates real factory admission and cache ownership.
    private struct FixtureTokenizer: GenerationPromptControllableTokenizer {
        var bosToken: String? { "<bos>" }
        var eosToken: String? { "<eos>" }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            (addSpecialTokens ? [0] : []) + text.utf8.map { 16 + Int($0) % 48 }
        }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map(String.init).joined(separator: " ")
        }
        func convertTokenToId(_ token: String) -> Int? {
            token == "<bos>" ? 0 : token == "<eos>" ? 1 : nil
        }
        func convertIdToToken(_ id: Int) -> String? {
            id == 0 ? "<bos>" : id == 1 ? "<eos>" : nil
        }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            try applyChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext,
                addGenerationPrompt: true)
        }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?, addGenerationPrompt: Bool
        ) throws -> [Int] {
            var ids: [Int] = []
            for message in messages {
                ids.append((message["role"] as? String) == "assistant" ? 3 : 2)
                if let reasoning = message["reasoning_content"] as? String {
                    ids += encode(text: reasoning, addSpecialTokens: false) + [6]
                }
                ids +=
                    encode(text: message["content"] as? String ?? "", addSpecialTokens: false) + [5]
            }
            if addGenerationPrompt { ids.append(3) }
            return ids
        }
    }
    private struct FixtureTokenizerLoader: TokenizerLoader {
        func load(from directory: URL) async throws -> any Tokenizer { FixtureTokenizer() }
    }

    private static func json(_ object: Any, to url: URL) throws {
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]).write(to: url)
    }
    private static func fixture(_ directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try json(
            [
                "eos_token_id": [1], "bos_token_id": 0, "do_sample": true,
                "temperature": 1.0, "top_p": 0.95,
            ], to: directory.appendingPathComponent("generation_config.json"))
        var config: [String: Any] = [
            "model_type": "k2_horizon", "hidden_size": 64,
            "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 64,
            "layernorm_num_groups": 4, "rms_norm_eps": 1e-6, "hidden_act": "silu",
            "tie_word_embeddings": false, "mlp_layout": "dense",
            "rope_parameters": ["rope_theta": 10_000_000, "rope_type": "default"],
        ]
        let cfg = try JSONDecoder().decode(
            K2HorizonConfiguration.self, from: JSONSerialization.data(withJSONObject: config))
        let shell = try K2HorizonModel(cfg)
        var ordinary: [String: MLXArray] = [:]
        var plan: [String: Any] = ["bits": 4, "group_size": 32, "mode": "affine"]
        var index: [String: String] = [:]
        for (name, parameter) in shell.parameters().flattened() where !name.contains(".mlp.") {
            let norm = name.contains("layernorm") || name == "model.norm.weight"
            let salt = name.utf8.reduce(0) { $0 + Int($1) }
            var values = [Float]()
            values.reserveCapacity(parameter.size)
            for i in 0 ..< parameter.size {
                let value = Float((i * 7 + salt) % 31 - 15) / 256
                values.append(norm ? 1 + value : value)
            }
            ordinary[name] = MLXArray(values, parameter.shape).asType(.bfloat16)
            if name.hasSuffix(".weight") { plan[String(name.dropLast(7))] = false }
            index[name] = "ordinary.safetensors"
        }
        try MLX.save(
            arrays: ordinary, url: directory.appendingPathComponent("ordinary.safetensors"))
        // Six actual mapped one-expert banks, one complete gate/up/down set per layer.
        var header: [String: Any] = [:]
        var payload = Data()
        for layer in 0 ..< 2 {
            for role in ["gate_proj", "up_proj", "down_proj"] {
                let input = role == "down_proj" ? 128 : 64
                let output = role == "down_proj" ? 64 : 128
                let path = "model.layers.\(layer).mlp.switch_mlp.\(role)"
                var words = [UInt32](repeating: 0, count: input * output / 8)
                for i in 0 ..< (input * output) {
                    words[i / 8] |= UInt32((i * 7 + i / input + layer) % 16) << ((i % 8) * 4)
                }
                let packed = words.withUnsafeBytes { Data($0) }
                let scales = [Float16](repeating: 0.125, count: output).withUnsafeBytes { Data($0) }
                for (suffix, bytes, dtype, shape) in [
                    ("tq2_packed", packed, "U32", [1, output, input / 8]),
                    ("tq2_scales", scales, "F16", [1, output]),
                ] {
                    let start = payload.count
                    payload.append(bytes)
                    header[path + "." + suffix] = [
                        "dtype": dtype, "shape": shape, "data_offsets": [start, payload.count],
                    ]
                    index[path + "." + suffix] = "banks.safetensors"
                }
                plan[path] = ["mode": "jangtq2", "bits": 4, "rotation": "hadamard32"]
            }
        }
        var encoded = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        let padded = ((encoded.count + 8 + 4095) / 4096) * 4096 - 8
        encoded.append(Data(repeating: 32, count: padded - encoded.count))
        var length = UInt64(encoded.count).littleEndian
        var bankFile = withUnsafeBytes(of: &length) { Data($0) }
        bankFile.append(encoded)
        bankFile.append(payload)
        try bankFile.write(to: directory.appendingPathComponent("banks.safetensors"))
        let levels = (0 ..< 16).map { (Double($0) - 7.5) * 0.25 }
        config["mlp_layout"] = "switch1"
        config["quantization"] = plan
        config["quantization_config"] = plan
        config["jangtq"] = [
            "version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
            "codebook_family": "odd-cubic", "rotation": "hadamard32",
            "codebooks": ["4": ["alpha": 0.25, "beta": 0, "levels": levels]],
        ]
        try json(config, to: directory.appendingPathComponent("config.json"))
        try json(
            ["weight_map": index],
            to: directory.appendingPathComponent("model.safetensors.index.json"))
    }

    func testFactorySwitch1TwoTurnColdLiveAndDiskCache() async throws {
        try await MLXMetalTestLock.withLock {
            try await Self.checkFactorySwitch1TwoTurnColdLiveAndDiskCache()
        }
    }

    private static func checkFactorySwitch1TwoTurnColdLiveAndDiskCache() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "k2-switch1-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        try Device.withDefaultDevice(.cpu) { try fixture(directory) }
        let context = try await LLMModelFactory.shared.load(
            from: directory, using: FixtureTokenizerLoader())
        let model = try XCTUnwrap(context.model as? K2HorizonModel)
        XCTAssertEqual(model.configuration.mlpLayout, "switch1")
        XCTAssertEqual(
            model.excludedSafetensorsKeys.count, 12,
            "six opaque mapped packed/scale pairs must be admitted")
        let first: [Chat.Message] = [.user("Remember invoice QZ-70419, total $136.99.")]
        let second =
            first + [
                Chat.Message(
                    role: .assistant, content: "Invoice QZ-70419 totals $136.99.",
                    reasoningContent: "I will retain the supplied invoice in this conversation."),
                .user("What is the invoice total?"),
            ]
        let a = try await context.processor.prepare(input: UserInput(chat: first))
        let b = try await context.processor.prepare(input: UserInput(chat: second))
        let firstIDs = a.text.tokens.reshaped(-1).asArray(Int.self)
        let ids = b.text.tokens.reshaped(-1).asArray(Int.self)
        let prefix = zip(firstIDs, ids).prefix(while: { $0.0 == $0.1 }).count
        XCTAssertEqual(
            prefix, firstIDs.count, "fixture tokenizer must preserve the first assistant opening")
        XCTAssertGreaterThan(ids.count, prefix)
        XCTAssertLessThan(firstIDs.count, 64, "first turn exercises routed decode")
        XCTAssertGreaterThanOrEqual(ids.count - prefix, 64, "second turn exercises routed prefill")
        // BF16 on the normal GPU path; no environment or precision-default mutation.
        // Cold replay uses identical prefill boundaries so approximation/dispatch changes
        // cannot masquerade as a serialization error. Single-shot vs split precision is
        // covered separately by the precise numerical diagnostic.
        try Device.withDefaultDevice(.gpu) {
            let cache = model.newCache(parameters: nil)
            let firstLogits = model(MLXArray(firstIDs).reshaped(1, firstIDs.count), cache: cache)
            eval(firstLogits)
            let path = directory.appendingPathComponent("boundary.safetensors")
            try savePromptCache(url: path, cache: cache, metadata: ["boundary": String(prefix)])
            let (restored, metadata) = try loadPromptCache(url: path)
            XCTAssertEqual(metadata["boundary"], String(prefix))
            XCTAssertEqual(restored.map(\.metaState), cache.map(\.metaState))
            let suffix = MLXArray(Array(ids.dropFirst(prefix))).reshaped(1, ids.count - prefix)
            let live = model(suffix, cache: cache)
            let disk = model(suffix, cache: restored)
            let coldCache = model.newCache(parameters: nil)
            eval(model(MLXArray(firstIDs).reshaped(1, firstIDs.count), cache: coldCache))
            let cold = model(suffix, cache: coldCache)
            eval(live, disk, cold)
            XCTAssertEqual(abs(live - disk).max().item(Float.self), 0)
            XCTAssertEqual(abs(live - cold).max().item(Float.self), 0)
            XCTAssertEqual(live.dtype, .bfloat16)
            XCTAssertTrue(restored.flatMap(\.state).allSatisfy { $0.dtype == .bfloat16 })
            XCTAssertTrue(coldCache.allSatisfy { $0.offset == ids.count })
            XCTAssertTrue(cache.allSatisfy { $0.offset == ids.count })
            XCTAssertTrue(restored.allSatisfy { $0.offset == ids.count })
            XCTAssertTrue(live.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
        }
    }
}
