import Foundation
import XCTest
@testable import MLXLMCommon

final class InklingProtocolTests: XCTestCase {
    func testNativeInvocationProgressArrivesBeforeClosingAndIncludesFinalBytes() throws {
        var splitter = try XCTUnwrap(ReasoningParser.fromCapabilityName("inkling2"))
        let tools: [[String: any Sendable]] = [["type": "function", "function": ["name": "write_script"] as [String: any Sendable]]]
        let processor = ToolCallProcessor(format: .inkling2, tools: tools)
        var progress = "", calls = [ToolCall](), visible = ""
        func consume(_ text: String) {
            for segment in splitter.feed(text) {
                switch segment {
                case .content(let content):
                    for event in routeGenerationText(content, channel: .content, through: processor) {
                        switch event {
                        case .toolCallProgress(let delta): progress += delta
                        case .toolCall(let call): calls.append(call)
                        case .chunk(let value): visible += value
                        default: break
                        }
                    }
                case .reasoning: XCTFail("tool bytes became reasoning")
                }
            }
        }
        let prefix = "<|message_model|>write_script<|content_invoke_tool_json|>"
        let body = "{\"name\":\"write_script\",\"args\":{\"source\":\"literal <|end_message|> and \\\"quotes\\\"\"}}"
        consume(prefix)
        for character in body { consume(String(character)) }
        XCTAssertFalse(progress.isEmpty, "long invocation must not wait for its end marker")
        XCTAssertTrue(calls.isEmpty)
        consume("<|end_")
        XCTAssertTrue(calls.isEmpty)
        consume("message|>")
        XCTAssertEqual(progress, prefix + body + "<|end_message|>")
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.function.arguments["source"], .string("literal <|end_message|> and \"quotes\""))
        XCTAssertEqual(visible, "")
    }
    func testInvocationProgressPreservesSplitCombiningAndJoinedUnicode() throws {
        var splitter = try XCTUnwrap(ReasoningParser.fromCapabilityName("inkling2"))
        let processor = ToolCallProcessor(format: .inkling2, tools: nil)
        let prefix = "<|message_model|>lookup<|content_invoke_tool_json|>"
        let body = #"{"name":"lookup","args":{"text":"e\#u{301} 👩‍💻"}}"#
        let raw = prefix + body + "<|end_message|>"
        var progress = ""
        for scalar in raw.unicodeScalars {
            for part in splitter.feed(String(scalar)) {
                if case .content(let text) = part {
                    for event in routeGenerationText(text, channel: .content, through: processor) {
                        if case .toolCallProgress(let delta) = event { progress += delta }
                    }
                }
            }
        }
        XCTAssertEqual(progress, raw)
    }

    func testSplitChannelsAndNativeInvocationAtEveryBoundary() throws {
        let raw = "<|message_model|><|content_thinking|>check<|end_message|><|message_model|><|content_text|>answer<|end_message|><|message_model|>lookup<|content_invoke_tool_json|>{\"name\":\"lookup\",\"args\":{\"text\":\"literal <|end_message|> stays\"}}<|end_message|><|content_model_end_sampling|>"
        for n in 1...raw.count {
            var splitter = try XCTUnwrap(ReasoningParser.fromCapabilityName("inkling2"))
            var chunks: [ReasoningSegment] = []
            var start = raw.startIndex
            while start < raw.endIndex {
                let end = raw.index(start, offsetBy: n, limitedBy: raw.endIndex) ?? raw.endIndex
                chunks += splitter.feed(String(raw[start..<end])); start = end
            }
            chunks += splitter.flush()
            let tools: [[String: any Sendable]] = [["type": "function", "function": ["name": "lookup"] as [String: any Sendable]]]
            let processor = ToolCallProcessor(format: .inkling2, tools: tools)
            var answer = "", thought = ""
            for chunk in chunks {
                switch chunk {
                case .reasoning(let text): thought += text
                case .content(let text): answer += processor.processChunk(text) ?? ""
                }
            }
            answer += processor.processEOS() ?? ""
            XCTAssertEqual(thought, "check"); XCTAssertEqual(answer, "answer")
            XCTAssertEqual(processor.toolCalls.count, 1)
            XCTAssertEqual(processor.toolCalls.first?.function.name, "lookup")
            XCTAssertEqual(processor.toolCalls.first?.function.arguments["text"], .string("literal <|end_message|> stays"))
        }
    }
    func testMalformedArgumentsAndUnregisteredToolsAreNotExecutable() {
        let parser = InklingToolCallParser()
        XCTAssertTrue(parser.parse(content: "{\"name\":\"x\",\"args\":\"{}\"}", tools: nil) == nil)
        XCTAssertTrue(parser.parse(content: "{\"name\":\"x\",\"args\":{}}", tools: []) == nil)
        XCTAssertTrue(!ToolCallFormat.inkling2.parsesToolCallsFromReasoningChannel)
        XCTAssertTrue(ToolCallFormat.infer(from: "inkling2_loop") == .inkling2)
    }
    func testToolLookingExampleInsideReasoningIsNeverExecutable() throws {
        var splitter = try XCTUnwrap(ReasoningParser.fromCapabilityName("inkling2"))
        let example = "<|content_invoke_tool_json|>{\"name\":\"delete\",\"args\":{}}"
        let raw = "<|message_model|><|content_thinking|>Example: " + example
            + "<|end_message|><|message_model|><|content_text|>I need confirmation.<|end_message|>"
        let parts = splitter.feed(raw) + splitter.flush()
        var visible = "", thought = ""
        for part in parts {
            switch part { case .content(let s): visible += s; case .reasoning(let s): thought += s }
        }
        XCTAssertEqual(visible, "I need confirmation.")
        XCTAssertEqual(thought, "Example: " + example)
    }

    func testInvocationRequiresNativeHeaderAndMatchingRecipient() {
        let parser = InklingToolCallParser()
        let payload = "<|content_invoke_tool_json|>{\"name\":\"lookup\",\"args\":{}}<|end_message|>"
        XCTAssertNil(parser.parse(content: payload, tools: nil))
        XCTAssertNil(parser.parse(content: "<|message_model|>delete" + payload, tools: nil))
        XCTAssertNotNil(parser.parse(content: "<|message_model|>lookup" + payload, tools: nil))
    }

}
