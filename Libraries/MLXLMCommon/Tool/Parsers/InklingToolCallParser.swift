// Native Inkling transport. No bare-JSON fallback and no executable reasoning.
import Foundation

public struct InklingToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<|message_model|>"
    public let endTag: String? = "<|end_message|>"
    public var usesCustomEndBoundary: Bool { true }
    public init() {}

    public func completeToolCallEnd(in content: String) -> String.Index? {
        guard let start = content.range(of: startTag!) else { return nil }
        let afterHeader = content[start.upperBound...]
        guard let invocation = afterHeader.range(of: "<|content_invoke_tool_json|>") else { return nil }
        let body = String(content[invocation.upperBound...])
        guard let end = InklingMessageStream.toolEnd(in: body) else { return nil }
        return content.index(invocation.upperBound, offsetBy: body.distance(from: body.startIndex, to: end.upperBound))
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        parse(content: toolCallBuffer, tools: tools).map { [$0] } ?? []
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        struct Invocation: Decodable { let name: String; let args: [String: JSONValue] }
        guard let start = content.range(of: startTag!),
              let invocation = content.range(of: "<|content_invoke_tool_json|>", range: start.upperBound..<content.endIndex) else { return nil }
        let recipient = String(content[start.upperBound..<invocation.lowerBound])
        var body = String(content[invocation.upperBound...])
        if let end = InklingMessageStream.toolEnd(in: body) { body = String(body[..<end.lowerBound]) }
        guard let data = body.data(using: .utf8),
              let value = try? JSONDecoder().decode(Invocation.self, from: data),
              !value.name.isEmpty, recipient == value.name else { return nil }
        if let tools, !tools.contains(where: {
            let function = $0["function"] as? [String: any Sendable] ?? $0
            return function["name"] as? String == value.name
        }) { return nil }
        return ToolCall(function: .init(name: value.name, arguments: value.args))
    }
}
