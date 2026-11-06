import Foundation

/// The native bundle has one supported reasoning mode, and requires a string
/// thinking field on every historical assistant turn. Never replace real history.
public enum K2HorizonTemplateContract {
    public struct FixedReasoningDeclaration: Sendable, Equatable {
        public let effort: String
    }
    public enum ContractError: Error, LocalizedError {
        case unsupportedReasoningMode(expectedEffort: String)
        public var errorDescription: String? {
            switch self {
            case .unsupportedReasoningMode(let effort):
                return "This bundle supports only its declared native \(effort) reasoning mode."
            }
        }
    }
    public static func matches(_ modelType: String?) -> Bool {
        modelType?.lowercased().replacingOccurrences(of: "-", with: "_") == "k2_horizon"
    }

    /// A family name alone cannot constrain future or unstamped bundles. Only
    /// the explicit one-mode serving declaration authorizes this validation.
    public static func fixedReasoningDeclaration(
        metadata: Data?, modelType: String?
    ) -> FixedReasoningDeclaration? {
        guard matches(modelType), let metadata,
            let root = try? JSONSerialization.jsonObject(with: metadata) as? [String: Any],
            let capabilities = root["capabilities"] as? [String: Any],
            capabilities["supports_reasoning_toggle"] as? Bool == false,
            let efforts = capabilities["reasoning_efforts"] as? [String], efforts.isEmpty,
            let reasoning = (root["reasoning"] as? [String: Any])
                ?? ((root["chat"] as? [String: Any])?["reasoning"] as? [String: Any]),
            reasoning["supported"] as? Bool == true,
            ((reasoning["supported_reasoning_efforts"] as? [String]) ?? []).isEmpty,
            reasoning["template_flag"] as? String == "reasoning_effort",
            let modes = reasoning["modes"] as? [String], modes.count == 1,
            reasoning["default_mode"] as? String == modes[0],
            let modeKwargs = reasoning["mode_kwargs"] as? [String: Any],
            let kwargs = modeKwargs[modes[0]] as? [String: Any],
            let effort = kwargs["reasoning_effort"] as? String,
            !effort.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            reasoning["reasoning_effort_transport"] == nil
                || reasoning["reasoning_effort_transport"] as? String == "chat_template_kwarg"
        else { return nil }
        return FixedReasoningDeclaration(effort: effort)
    }

    public static func validateContext(
        _ context: [String: any Sendable]?, modelType: String?,
        declaration: FixedReasoningDeclaration?
    ) throws {
        guard matches(modelType), let declaration, let context else { return }
        if let effort = context["reasoning_effort"], (effort as? String) != declaration.effort {
            throw ContractError.unsupportedReasoningMode(expectedEffort: declaration.effort)
        }
        if let thinking = context["enable_thinking"], (thinking as? Bool) != true {
            throw ContractError.unsupportedReasoningMode(expectedEffort: declaration.effort)
        }
    }
    public static func prepare(messages: [Message], modelType: String?) -> [Message] {
        guard matches(modelType) else { return messages }
        let fields = ["think", "think_fast", "think_faster", "reasoning_content", "reasoning"]
        return messages.map { message in
            guard message["role"] as? String == "assistant",
                !fields.contains(where: { message[$0] != nil })
            else { return message }
            var copy = message
            copy["reasoning_content"] = ""
            return copy
        }
    }
}
