import Foundation

/// Validate the actual prepared generation contract, rather than guessing from
/// model names or silently changing a model's reasoning/template settings.
public func validateStructuredOutputRequest(
    input: LMInput, parameters: GenerateParameters, context: ModelContext
) throws {
    guard let schema = parameters.jsonSchema else { return }
    if !parameters.extraStopStrings.isEmpty {
        throw GenerationFailure(
            stage: .preparation,
            cause:
                "JSON schema decoding cannot be combined with text stop strings; completion is determined by the schema and EOS"
        )
    }
    if context.model is any BlockDiffusionModel {
        throw GenerationFailure(
            stage: .preparation,
            cause: "JSON schema decoding is not supported for block-diffusion models")
    }
    if input.toolSchemas?.isEmpty == false {
        throw GenerationFailure(
            stage: .preparation,
            cause:
                "JSON schema applies to a final response; tool-call envelopes are not yet supported"
        )
    }
    let tail = _decodePromptTail(input: input, tokenizer: context.tokenizer, tokens: 64)
    if ReasoningParser.forPrompt(
        stampName: context.configuration.reasoningParserName, promptTail: tail
    )?.isInsideReasoning == true {
        throw GenerationFailure(
            stage: .preparation,
            cause:
                "JSON schema decoding does not yet support an active reasoning envelope; select the model's native non-reasoning mode explicitly"
        )
    }
    _ = try JSONSchemaLogitProcessor(
        schema: schema, tokenizer: context.tokenizer,
        stopTokenIDs: buildStopTokenIds(
            modelConfiguration: context.configuration, tokenizer: context.tokenizer),
        base: parameters.processor())
}
