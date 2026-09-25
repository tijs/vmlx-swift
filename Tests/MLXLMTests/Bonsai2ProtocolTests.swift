// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import CoreImage
import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Testing
@preconcurrency import VMLXTokenizers

private enum Bonsai2ProtocolFixture {
    static let bundleNames = ["Bonsai-2-27B-Ternary-JANG", "Bonsai-2-27B-1.75bit-JANG"]
    static var root: URL? {
        ProcessInfo.processInfo.environment["BONSAI2_PROTOCOL_BUNDLE_ROOT"].map {
            URL(fileURLWithPath: $0, isDirectory: true)
        }
    }

    static let tools: [ToolSpec] = {
        let string: ToolSpec = ["type": "string"]
        let labels: ToolSpec = ["type": "array", "items": string]
        let metadata: ToolSpec = ["type": "object", "properties": ["code": string]]
        let properties: ToolSpec = [
            "path": string, "content": string,
            "enabled": ["type": "boolean"], "count": ["type": "integer"],
            "labels": labels, "metadata": metadata,
        ]
        let writeParameters: ToolSpec = [
            "type": "object", "properties": properties,
            "required": ["path", "content"], "additionalProperties": false,
        ]
        let readParameters: ToolSpec = [
            "type": "object", "properties": ["path": string],
            "required": ["path"], "additionalProperties": false,
        ]
        let write: ToolSpec = [
            "name": "write_note", "description": "Write a note without interpreting its content.",
            "parameters": writeParameters,
        ]
        let read: ToolSpec = [
            "name": "read_note", "description": "Read a saved note.", "parameters": readParameters,
        ]
        return [["type": "function", "function": write], ["type": "function", "function": read]]
    }()

    static func call(_ name: String, _ arguments: [String: JSONValue], order: [String]) -> ToolCall
    {
        ToolCall(function: .init(name: name, arguments: arguments, argumentOrder: order))
    }

    static func history(images: [UserInput.Image] = []) -> [Chat.Message] {
        [
            .system("Keep the user-provided note unchanged."),
            .user("Save the code 007, then read it back.", images: images),
            .init(
                role: .assistant, content: "", reasoningContent: "Verify the round trip.",
                toolCalls: [
                    call(
                        "write_note", ["path": .string("note.md"), "content": .string("007")],
                        order: ["path", "content"]),
                    call("read_note", ["path": .string("note.md")], order: ["path"]),
                ]),
            .tool("SAVED_NOTE_OK"), .tool("READ_NOTE_CODE=007"),
            .user("What code did you read?"),
        ]
    }

    static func canonicalJSON(_ value: Any) throws -> Data {
        try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
    }

    static func renderedTools(_ rendered: String) throws -> [JSONValue] {
        let start = try #require(rendered.range(of: "<tools>"))
        let end = try #require(
            rendered.range(of: "</tools>", range: start.upperBound ..< rendered.endIndex))
        return try rendered[start.upperBound ..< end.lowerBound].split(separator: "\n").map {
            try JSONDecoder().decode(JSONValue.self, from: Data($0.utf8))
        }
    }

    static func stream(_ chunks: [String], promptTail: String) throws -> [Generation] {
        var reasoning = try #require(
            ReasoningParser.forPrompt(stampName: "qwen3", promptTail: promptTail))
        let format = try #require(ToolCallFormat.fromCapabilityName("qwen3_coder"))
        let tools = ToolCallProcessor(format: format, tools: Self.tools)
        var events: [Generation] = []
        var lastChannel: GenerationTextChannel = .content
        func append(_ segments: [ReasoningSegment]) {
            for segment in segments {
                switch segment {
                case .reasoning(let value):
                    lastChannel = .reasoning
                    events += routeGenerationText(value, channel: .reasoning, through: tools)
                case .content(let value):
                    lastChannel = .content
                    events += routeGenerationText(value, channel: .content, through: tools)
                }
            }
        }
        for chunk in chunks { append(reasoning.feed(chunk)) }
        append(reasoning.flush())
        events += flushGenerationText(channel: lastChannel, through: tools)
        #expect(tools.toolCallProtocolFailure == nil)
        return events
    }

    static func fragments(_ text: String, width: Int) -> [String] {
        let characters = Array(text)
        return stride(from: 0, to: characters.count, by: width).map {
            String(characters[$0 ..< min($0 + width, characters.count)])
        }
    }
}

/// Opt-in, offline tests against the actual installed tokenizer/config artifacts.
/// No model weights are opened. A missing environment variable skips this suite;
/// a supplied root with either missing bundle is an error, not a vacuous pass.
@Suite(
    "Actual Bonsai2 tokenizer and media processor; no model weights", .serialized,
    .enabled(if: Bonsai2ProtocolFixture.root != nil))
struct ActualBonsai2ProtocolTests {
    @Test(
        "native default and Off/low/medium/xhigh preserve schemas and tool history",
        arguments: Bonsai2ProtocolFixture.bundleNames)
    func tokenizerContract(bundle: String) async throws {
        let root = try #require(Bonsai2ProtocolFixture.root)
        let directory = root.appendingPathComponent(bundle)
        let config = try JangLoader.loadConfig(at: directory)
        #expect(config.capabilities?.reasoningParser == "qwen3")
        #expect(config.capabilities?.toolParser == "qwen3_coder")
        let capability = ReasoningCapability.forModel(at: directory)
        #expect(capability.source == .declared)
        #expect(capability.levels == [0, 1, 2, 3])
        #expect(capability.efforts == ["low", "medium", "xhigh"])
        #expect(capability.defaultLevel == 3)
        let template = try String(
            contentsOf: directory.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        #expect(!ChatTemplateRepair.needsRepair(template))
        #expect(ChatTemplateRepair.repaired(template) == template)
        let tokenizer = try await #huggingFaceTokenizerLoader().load(
            from: JangLoader.resolveChatTemplateSidecarSubstitution(for: directory))
        let controllable = try #require(tokenizer as? any GenerationPromptControllableTokenizer)
        #expect(Qwen3XMLToolTemplate.matchesTemplate(template))
        let defaults = try JSONDecoder().decode(
            GenerationConfigFile.self,
            from: Data(contentsOf: directory.appendingPathComponent("generation_config.json")))
        let parameters = GenerateParameters(generationConfig: defaults)
        #expect(parameters.temperature == 1)
        #expect(parameters.topP == 0.95)
        #expect(parameters.topK == 20)
        #expect(parameters.repetitionPenalty == 1)
        #expect(tokenizer.convertTokenToId("<|im_end|>") == 248046)
        #expect(tokenizer.convertTokenToId("<|endoftext|>") == 248044)

        let messages = DefaultMessageGenerator().generate(
            messages: Bonsai2ProtocolFixture.history())
        let expectedTools = try Bonsai2ProtocolFixture.tools.map {
            try JSONDecoder().decode(JSONValue.self, from: Bonsai2ProtocolFixture.canonicalJSON($0))
        }
        var renderedByLevel: [Int: String] = [:]
        for level in [-1, 0, 1, 2, 3] {
            let context = level < 0 ? nil : capability.applying(level: level)
            let ids = try tokenizer.applyChatTemplate(
                messages: messages,
                tools: Bonsai2ProtocolFixture.tools, additionalContext: context)
            let rendered = tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
            renderedByLevel[level] = rendered
            let observedTools = try Bonsai2ProtocolFixture.renderedTools(rendered)
            #expect(observedTools == expectedTools)
            let canonicalIDs = try controllable.applyChatTemplate(
                messages: messages, tools: Bonsai2ProtocolFixture.tools,
                additionalContext: context, addGenerationPrompt: false)
            let canonical = tokenizer.decode(tokenIds: canonicalIDs, skipSpecialTokens: false)
            let canonicalTools = try Bonsai2ProtocolFixture.renderedTools(canonical)
            #expect(canonicalTools == expectedTools)
            #expect(canonicalIDs.count < ids.count)
            #expect(ids.starts(with: canonicalIDs))
            let explicitIDs = try controllable.applyChatTemplate(
                messages: messages, tools: Bonsai2ProtocolFixture.tools,
                additionalContext: context, addGenerationPrompt: true)
            #expect(explicitIDs == ids)
            #expect(rendered.contains("SAVED_NOTE_OK"))
            #expect(rendered.contains("READ_NOTE_CODE=007"))
            #expect(rendered.contains("Verify the round trip."))
            #expect(rendered.contains("<function=write_note>"))
            #expect(rendered.contains("<function=read_note>"))
            #expect(rendered.contains("<parameter=content>\n007\n</parameter>"))
            #expect(!rendered.contains("_vmlx_tool_argument_orders"))
            let off = level == 0
            #expect(
                rendered.hasSuffix(
                    off
                        ? "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                        : "<|im_start|>assistant\n<think>\n"))
            let parser = try #require(
                ReasoningParser.forPrompt(
                    stampName: config.capabilities?.reasoningParser, promptTail: rendered))
            #expect(parser.isInsideReasoning == !off)
            if level == 1 { #expect(rendered.contains("Reasoning effort is set to low.")) }
            if level == 2 || off { #expect(!rendered.contains("Reasoning effort is set to")) }
            if level == 3 || level == -1 {
                #expect(rendered.contains("Reasoning effort is set to xhigh."))
            }
            print(
                "BONSAI2_PROTOCOL bundle=\(bundle) level=\(level) prompt_tokens=\(ids.count) weights_loaded=false"
            )
        }
        #expect(renderedByLevel[-1] == renderedByLevel[3])
        #expect(renderedByLevel[0] != renderedByLevel[2])
        let disabledPreserve: [String: any Sendable] = ["preserve_thinking": false]
        let stripped = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: Bonsai2ProtocolFixture.tools, additionalContext: disabledPreserve)
        #expect(
            !tokenizer.decode(tokenIds: stripped, skipSpecialTokens: false).contains(
                "Verify the round trip."))
        #expect(throws: (any Error).self) {
            _ = try tokenizer.applyChatTemplate(
                messages: messages, tools: Bonsai2ProtocolFixture.tools,
                additionalContext: ["enable_thinking": true, "reasoning_effort": "unsupported"])
        }
        for addGenerationPrompt in [false, true] {
            #expect(throws: (any Error).self) {
                _ = try controllable.applyChatTemplate(
                    messages: messages, tools: Bonsai2ProtocolFixture.tools,
                    additionalContext: [
                        "enable_thinking": true, "reasoning_effort": "unsupported",
                    ],
                    addGenerationPrompt: addGenerationPrompt)
            }
            #expect(throws: (any Error).self) {
                _ = try controllable.applyChatTemplate(
                    messages: [], tools: nil, additionalContext: nil,
                    addGenerationPrompt: addGenerationPrompt)
            }
            let invalidMedia: [String: any Sendable] = [
                "role": "system", "content": [["type": "image"]],
            ]
            #expect(throws: (any Error).self) {
                _ = try controllable.applyChatTemplate(
                    messages: [invalidMedia, ["role": "user", "content": "What is this?"]],
                    tools: nil, additionalContext: nil,
                    addGenerationPrompt: addGenerationPrompt)
            }
        }
    }

    @Test(
        "real two-image preprocessing after tool history retains pixels, tokens, schemas and cache isolation",
        arguments: Bonsai2ProtocolFixture.bundleNames)
    func mediaProcessor(bundle: String) async throws {
        let root = try #require(Bonsai2ProtocolFixture.root)
        let directory = root.appendingPathComponent(bundle)
        let tokenizer = try await #huggingFaceTokenizerLoader().load(
            from: JangLoader.resolveChatTemplateSidecarSubstitution(for: directory))
        let config = try JSONDecoder().decode(
            Qwen3VLProcessorConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("preprocessor_config.json")))
        let processor = Qwen3VLProcessor(config, tokenizer: tokenizer)
        let red = CIImage(color: CIColor(red: 1, green: 0, blue: 0)).cropped(
            to: CGRect(x: 0, y: 0, width: 256, height: 256))
        let blue = CIImage(color: CIColor(red: 0, green: 0, blue: 1)).cropped(to: red.extent)
        let capability = ReasoningCapability.forModel(at: directory)
        let off = try #require(capability.applying(level: 0))
        let low = try #require(capability.applying(level: 1))
        func prepare(_ second: CIImage, context: [String: any Sendable]) async throws -> LMInput {
            var history = Bonsai2ProtocolFixture.history(images: [.ciImage(red)])
            history.append(
                .user("Compare the new image with the earlier one.", images: [.ciImage(second)]))
            return try await processor.prepare(
                input: UserInput(
                    chat: history,
                    tools: Bonsai2ProtocolFixture.tools, additionalContext: context))
        }
        let input = try await prepare(blue, context: off)
        let same = try await prepare(blue, context: off)
        let changedImage = try await prepare(red, context: off)
        let changedEffort = try await prepare(blue, context: low)
        let image = try #require(input.image)
        let frames = try #require(image.frames)
        #expect(frames.count == 2)
        #expect(frames.allSatisfy { $0.t == 1 && $0.h == 16 && $0.w == 16 })
        #expect(image.pixels.shape == [512, 1536])
        #expect(input.video == nil && input.audio == nil)
        let ids = try #require(input.text.tokenIds)
        let imageToken = try #require(tokenizer.convertTokenToId("<|image_pad|>"))
        #expect(ids.filter { $0 == imageToken }.count == 128)
        #expect(input.mediaTokenIds?.contains(imageToken) == true)
        let schemas = try #require(input.toolSchemas)
        let observedJSON = try Bonsai2ProtocolFixture.canonicalJSON(schemas)
        let expectedJSON = try Bonsai2ProtocolFixture.canonicalJSON(Bonsai2ProtocolFixture.tools)
        #expect(observedJSON == expectedJSON)
        let rendered = tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
        #expect(rendered.contains("SAVED_NOTE_OK") && rendered.contains("READ_NOTE_CODE=007"))
        #expect(
            rendered.contains("<function=write_note>") && rendered.contains("<function=read_note>"))
        #expect(rendered.contains("Verify the round trip."))
        MLX.eval(image.pixels)
        let finite = image.pixels.asArray(Float.self).allSatisfy { $0.isFinite }
        #expect(finite)
        let mediaSalt = try #require(computeMediaSalt(for: input))
        #expect(mediaSalt == computeMediaSalt(for: same))
        #expect(mediaSalt != computeMediaSalt(for: changedImage))
        #expect(computeCacheSalt(for: input) != computeCacheSalt(for: changedEffort))
        #expect(input.cacheScopeSalt == "reasoning=off")
        #expect(changedEffort.cacheScopeSalt == "reasoning=on|effort=low")
        print(
            "BONSAI2_PROCESSOR bundle=\(bundle) prompt_tokens=\(ids.count) image_slots=128 frames=2 pixels=\(image.pixels.shape) cache_boundaries=\(input.cachePrefixTokenCounts) weights_loaded=false vlm_forward=false"
        )
    }
}

@Suite("Bonsai2 native fragmented tool/reasoning protocol")
struct Bonsai2ToolStreamTests {
    @Test func nativeTemplateDetectionDoesNotChangeOtherDialects() {
        let native = "<|im_start|><|im_end|><tool_call><function=f><parameter=x></tool_call>"
        #expect(Qwen3XMLToolTemplate.matchesTemplate(native))
        #expect(!Qwen3XMLToolTemplate.matchesTemplate(nil))
        #expect(
            !Qwen3XMLToolTemplate.matchesTemplate(
                "<|im_start|><tool_call>{\"name\":\"f\"}</tool_call>"))
        #expect(
            !Qwen3XMLToolTemplate.matchesTemplate(
                "<function name=\"f\"><param name=\"x\"><![CDATA[x]]></param></function>"))
        #expect(
            !Qwen3XMLToolTemplate.matchesTemplate(
                native.replacingOccurrences(of: "<|im_end|>", with: "")))
    }

    @Test func mediaGeneratorPreservesAllCanonicalMetadata() throws {
        let generator = Qwen3VLMessageGenerator()
        var messages = Bonsai2ProtocolFixture.history()
        messages.append(.tool("correlated result", toolCallId: "call-007"))
        for message in messages {
            var media = generator.generate(message: message)
            var canonical = defaultMessageDict(for: message)
            let content = try #require(media.removeValue(forKey: "content") as? [[String: String]])
            #expect(content.last == ["type": "text", "text": message.content])
            canonical.removeValue(forKey: "content")
            let mediaJSON = try Bonsai2ProtocolFixture.canonicalJSON(media)
            let canonicalJSON = try Bonsai2ProtocolFixture.canonicalJSON(canonical)
            #expect(mediaJSON == canonicalJSON)
        }
    }

    @Test("two calls retain ordering, types and exact string values", arguments: [1, 2, 7, 4096])
    func twoCalls(width: Int) throws {
        let payload = """
            Checking the note.</think><tool_call><function=write_note>
            <parameter=path>note.md</parameter><parameter=content>007</parameter>
            <parameter=enabled>true</parameter><parameter=count>2</parameter>
            <parameter=labels>["a","b"]</parameter><parameter=metadata>{"code":"007"}</parameter>
            </function></tool_call><tool_call><function=read_note><parameter=path>note.md</parameter></function></tool_call>
            """
        let events = try Bonsai2ProtocolFixture.stream(
            Bonsai2ProtocolFixture.fragments(payload, width: width),
            promptTail: "<|im_start|>assistant\n<think>\n")
        let calls = events.compactMap(\.toolCall)
        #expect(calls.map(\.function.name) == ["write_note", "read_note"])
        let first = try #require(calls.first)
        #expect(
            first.function.arguments == [
                "path": .string("note.md"), "content": .string("007"),
                "enabled": .bool(true), "count": .int(2),
                "labels": .array([.string("a"), .string("b")]),
                "metadata": .object(["code": .string("007")]),
            ])
        #expect(events.compactMap(\.reasoning).joined() == "Checking the note.")
        #expect(
            events.compactMap(\.chunk).joined().trimmingCharacters(in: .whitespacesAndNewlines)
                .isEmpty)
        let persisted = try JSONDecoder().decode([ToolCall].self, from: JSONEncoder().encode(calls))
        #expect(persisted.map(\.function.arguments) == calls.map(\.function.arguments))
        #expect(persisted.map(\.function.argumentOrder) == calls.map(\.function.argumentOrder))
    }

    @Test(
        "literal reasoning tags inside a committed native XML value are payload",
        arguments: [1, 7, 4096])
    func literalReasoningTags(width: Int) throws {
        let content = "before <think>literal-not-reasoning</think> after"
        let payload =
            "<tool_call><function=write_note><parameter=path>tags.md</parameter><parameter=content>\(content)</parameter></function></tool_call>"
        let events = try Bonsai2ProtocolFixture.stream(
            Bonsai2ProtocolFixture.fragments(payload, width: width),
            promptTail: "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        let calls = events.compactMap(\.toolCall)
        #expect(calls.count == 1)
        let call = try #require(calls.first)
        #expect(call.function.arguments["content"] == .string(content))
        #expect(events.compactMap(\.reasoning).joined().isEmpty)
    }

    @Test(
        "opaque payload ends at native closer; normal reasoning resumes",
        arguments: [1, 2, 7, 4096])
    func reasoningAfterTool(width: Int) throws {
        let payload =
            "<tool_call><function=write_note><parameter=path>tags.md</parameter><parameter=content><think>literal</think></parameter></function></tool_call><think>Now check.</think>Done."
        let events = try Bonsai2ProtocolFixture.stream(
            Bonsai2ProtocolFixture.fragments(payload, width: width), promptTail: "</think>\n")
        #expect(
            events.compactMap(\.toolCall).first?.function.arguments["content"]
                == .string("<think>literal</think>"))
        #expect(events.compactMap(\.reasoning).joined() == "Now check.")
        #expect(events.compactMap(\.chunk).joined() == "Done.")
    }

    @Test func incompletePayloadStreamsWithoutInventingACloser() throws {
        var parser = try #require(
            ReasoningParser.forPrompt(stampName: "qwen3", promptTail: "</think>"))
        let payload =
            "<tool_call><function=write_note><parameter=content>"
            + String(repeating: "<think>literal</think>", count: 300) + "<thi"
        let early = parser.feed(payload)
        func content(_ segments: [ReasoningSegment]) -> String {
            segments.compactMap { if case .content(let text) = $0 { text } else { nil } }.joined()
        }
        #expect(content(early).count >= payload.count - 11)
        let final = parser.flush()
        #expect(content(early + final) == payload)
        #expect(!parser.isInsideReasoning)
        let processor = ToolCallProcessor(format: .xmlFunction, tools: Bonsai2ProtocolFixture.tools)
        _ = processor.processChunk(content(early + final))
        _ = processor.processEOS()
        #expect(processor.toolCalls.isEmpty)
        #expect(
            parser.feed("<think>New.</think>Done.") + parser.flush() == [
                .reasoning("New."), .content("Done."),
            ])
    }

    @Test func reasoningSideExamplesDoNotActivateOpaqueContentMode() throws {
        var parser = try #require(
            ReasoningParser.forPrompt(stampName: "qwen3", promptTail: "<think>"))
        let segments =
            parser.feed("Example: <tool_call>illustration</tool_call>.</think>Answer.")
            + parser.flush()
        #expect(
            segments == [
                .reasoning("Example: <tool_call>illustration</tool_call>."), .content("Answer."),
            ])
        #expect(
            ReasoningParser.fromCapabilityName("minicpm5")?.preservesQwenToolCallPayloads == false)
        #expect(
            ReasoningParser.fromCapabilityName("deepseek_r1")?.preservesQwenToolCallPayloads
                == false)
        #expect(ReasoningParser.fromCapabilityName("qwen3")?.preservesQwenToolCallPayloads == true)
    }
}
