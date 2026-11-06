import Foundation
import MLXHuggingFace
import MLXLMCommon
import Testing
import VMLXTokenizers

@Suite("Hugging Face loader grammar metadata")
struct HuggingFaceGrammarMetadataTests {
    private func fixture(cleanup: Bool) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let tokenizer =
            #"{"model":{"type":"BPE","vocab":{"a":0,"Ã":2,"©":3},"merges":[]},"decoder":{"type":"ByteLevel"},"added_tokens":[{"id":5,"content":"é","special":false},{"id":8,"content":"<eos>","special":true}]}"#
        let config =
            #"{"tokenizer_class":"PreTrainedTokenizerFast","clean_up_tokenization_spaces":"#
            + String(cleanup)
            + #", "eos_token":"<eos>","chat_template":"{{ messages[0]['content'] }}{% if add_generation_prompt %}a{% endif %}"}"#
        try Data(tokenizer.utf8).write(to: directory.appendingPathComponent("tokenizer.json"))
        try Data(config.utf8).write(to: directory.appendingPathComponent("tokenizer_config.json"))
        return directory
    }

    @Test func loaderSuppliesExactMetadataAndPreservesAdaptorBehavior() async throws {
        let directory = try fixture(cleanup: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let upstream = try await AutoTokenizer.from(modelFolder: directory)
        let standalone = #adaptHuggingFaceTokenizer(upstream)
        #expect(standalone.grammarTokenVocabulary == nil)
        let loader = #huggingFaceTokenizerLoader()
        let loaded = try await loader.load(from: directory)
        let metadata = try #require(loaded.grammarTokenVocabulary)
        #expect(metadata.vocabulary.count == 9)
        #expect(metadata.vocabulary[2] == "Ã")
        #expect(metadata.vocabulary[3] == "©")
        #expect(metadata.vocabulary[5] == "Ã©")
        #expect(metadata.specialTokenIDs == Set([1, 4, 6, 7, 8]))
        #expect(loaded.incrementalByteLevelDecoder != nil)
        #expect(
            loaded.encode(text: "a", addSpecialTokens: false)
                == standalone.encode(text: "a", addSpecialTokens: false))
        #expect(
            loaded.decode(tokenIds: [2, 3, 5], skipSpecialTokens: false)
                == standalone.decode(tokenIds: [2, 3, 5], skipSpecialTokens: false))
        let base = try #require(standalone as? any GenerationPromptControllableTokenizer)
        let controlled = try #require(loaded as? any GenerationPromptControllableTokenizer)
        for addPrompt in [false, true] {
            let messages: [[String: any Sendable]] = [["role": "user", "content": "a"]]
            #expect(
                try controlled.applyChatTemplate(
                    messages: messages, tools: nil, additionalContext: nil,
                    addGenerationPrompt: addPrompt)
                    == base.applyChatTemplate(
                        messages: messages, tools: nil, additionalContext: nil,
                        addGenerationPrompt: addPrompt))
        }
    }

    @Test func cleanupEnabledRemainsUnsupportedWithoutChangingOrdinaryTokenization() async throws {
        let directory = try fixture(cleanup: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let loader = #huggingFaceTokenizerLoader()
        let loaded = try await loader.load(from: directory)
        #expect(loaded.grammarTokenVocabulary == nil)
        #expect(loaded.encode(text: "a", addSpecialTokens: false) == [0])
    }
}
