import Foundation

/// Qwen's configured ChatML/XML template owns its schema and validation contract.
/// A template error is not permission to switch it to another ChatML dialect.
public enum Qwen3XMLToolTemplate {
    public static func matchesTemplate(_ template: String?) -> Bool {
        guard let template else { return false }
        return template.contains("<|im_start|>")
            && template.contains("<|im_end|>")
            && template.contains("<tool_call>")
            && template.contains("</tool_call>")
            && template.contains("<function=")
            && template.contains("<parameter=")
    }
}
