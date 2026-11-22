import Foundation

/// Split authored message headers from payload without interpreting quoted tool arguments.
/// Each instance belongs to one generation; no protocol state crosses requests.
struct InklingMessageStream: Sendable {
    private enum Rail { case content, reasoning, header, tool }
    private var rail: Rail = .content
    private var buffer = ""
    private var recipient = ""
    private var toolPrefix = ""
    private var emittedToolBytes = 0
    var isReasoning: Bool { rail == .reasoning }
    private let invoke = "<|content_invoke_tool_json|>"
    private let end = "<|end_message|>"
    private let markers = ["<|message_model|>", "<|content_thinking|>", "<|content_text|>",
                           "<|content_invoke_tool_json|>", "<|end_message|>", "<|content_model_end_sampling|>"]

    /// End-message markers inside a JSON string are literal argument bytes.
    static func toolEnd(in text: String) -> Range<String.Index>? {
        var quoted = false, escaped = false
        var i = text.startIndex
        while i < text.endIndex {
            let ch = text.unicodeScalars[i]
            if escaped { escaped = false }
            else if quoted && ch == "\\" { escaped = true }
            else if ch == "\"" { quoted.toggle() }
            else if !quoted && text[i...].hasPrefix("<|end_message|>") {
                return i..<text.index(i, offsetBy: "<|end_message|>".count)
            }
            i = text.unicodeScalars.index(after: i)
        }
        return nil
    }

    mutating func feed(_ chunk: String, final: Bool = false) -> [ReasoningSegment] {
        buffer += chunk
        var out: [ReasoningSegment] = []
        while !buffer.isEmpty {
            if rail == .tool {
                let bytes = Array(buffer.utf8)
                guard let boundary = Self.toolEnd(in: buffer) else {
                    // Forward committed invocation bytes as they arrive. Keep
                    // the complete body for quote-aware boundary detection,
                    // but hold a possible split end-marker out of the stream.
                    let retained = final ? 0 : (1...min(bytes.count, end.utf8.count)).reversed().first(where: {
                        Array(end.utf8).starts(with: bytes.suffix($0))
                    }) ?? 0
                    let available = bytes.count - retained
                    if available > emittedToolBytes || !toolPrefix.isEmpty {
                        out.append(.content(toolPrefix + String(decoding: bytes.dropFirst(emittedToolBytes).prefix(available - emittedToolBytes), as: UTF8.self)))
                        toolPrefix = ""
                        emittedToolBytes = available
                    }
                    if final { buffer = ""; emittedToolBytes = 0 }
                    break
                }
                let boundaryByte = buffer.utf8.distance(from: buffer.utf8.startIndex, to: boundary.upperBound)
                out.append(.content(toolPrefix + String(decoding: bytes[emittedToolBytes..<boundaryByte], as: UTF8.self)))
                toolPrefix = ""; emittedToolBytes = 0
                buffer = String(buffer[boundary.upperBound...]); rail = .content
                continue
            }
            let matches = markers.compactMap { tag -> (String, Range<String.Index>)? in
                buffer.range(of: tag).map { (tag, $0) }
            }
            if let (tag, range) = matches.min(by: { $0.1.lowerBound < $1.1.lowerBound }) {
                let payload = String(buffer[..<range.lowerBound])
                if rail == .header { recipient += payload }
                else if !payload.isEmpty { out.append(rail == .reasoning ? .reasoning(payload) : .content(payload)) }
                buffer = String(buffer[range.upperBound...])
                // Channel declarations belong to a message header. A quoted
                // invocation marker in deliberation never opens executable
                // transport or moves the rest of the thought to content.
                if rail == .reasoning && tag != end {
                    out.append(.reasoning(tag))
                    continue
                }
                switch tag {
                case "<|message_model|>": rail = .header; recipient = ""
                case "<|content_thinking|>": rail = .reasoning
                case "<|content_invoke_tool_json|>":
                    if rail == .header {
                        toolPrefix = "<|message_model|>" + recipient + invoke
                        emittedToolBytes = 0
                        rail = .tool
                    } else { out.append(.content(tag)) }
                default: rail = .content
                }
                continue
            }
            // Retain only a possible control-token prefix across streaming boundaries.
            let retained = final ? 0 : (1...min(buffer.count, 40)).reversed().first(where: { n in
                markers.contains(where: { $0.hasPrefix(String(buffer.suffix(n))) })
            }) ?? 0
            let split = buffer.index(buffer.endIndex, offsetBy: -retained)
            let payload = String(buffer[..<split])
            if rail == .header { recipient += payload }
            else if !payload.isEmpty { out.append(rail == .reasoning ? .reasoning(payload) : .content(payload)) }
            buffer = String(buffer[split...])
            break
        }
        return out
    }
}
