import Foundation
import Testing

@testable import MLXLMCommon

@Suite("Exact ByteLevel grammar metadata")
struct GrammarTokenVocabularyJSONTests {
    private let config = Data(#"{"clean_up_tokenization_spaces":false,"eos_token":"<eos>"}"#.utf8)

    @Test func retainsIDGapsAddedLiteralsAndSpecialIDs() throws {
        let data = Data(
            #"{"model":{"type":"BPE","vocab":{"a":0,"Ã":2,"©":3}},"decoder":{"type":"ByteLevel"},"added_tokens":[{"id":5,"content":"é","special":false},{"id":8,"content":"<eos>","special":true}]}"#
                .utf8)
        let result = try #require(
            GrammarTokenVocabulary.fromTokenizerJSON(data, tokenizerConfig: config))
        #expect(result.vocabulary.count == 9)
        #expect(result.vocabulary[2] == "Ã")  // first byte of é, not a replacement character
        #expect(result.vocabulary[3] == "©")  // second byte
        #expect(result.vocabulary[5] == "Ã©")  // literal added token bypasses ByteLevel decoder
        #expect(result.specialTokenIDs == Set([1, 4, 6, 7, 8]))
    }

    @Test func rejectsUnsupportedDecodersAndCleanup() {
        for decoder in [
            #"{"type":"Metaspace"}"#, #"{"type":"ByteFallback"}"#,
            #"{"type":"Sequence","decoders":[{"type":"ByteLevel"}]}"#,
        ] {
            let data = Data(
                (#"{"model":{"type":"BPE","vocab":{"a":0}},"decoder":"# + decoder + "}").utf8)
            #expect(GrammarTokenVocabulary.fromTokenizerJSON(data, tokenizerConfig: config) == nil)
        }
        let data = Data(
            #"{"model":{"type":"BPE","vocab":{"a":0}},"decoder":{"type":"ByteLevel"}}"#.utf8)
        for configuration in [#"{}"#, #"{"clean_up_tokenization_spaces":true}"#] {
            #expect(
                GrammarTokenVocabulary.fromTokenizerJSON(
                    data, tokenizerConfig: Data(configuration.utf8)) == nil)
        }
    }

    @Test func rejectsAmbiguousIDsOrInvalidByteAlphabet() {
        for vocab in [#"{"a":0,"b":0}"#, #"{"a":-1}"#, #"{"😀":0}"#] {
            let data = Data(
                (#"{"model":{"type":"BPE","vocab":"# + vocab + #"},"decoder":{"type":"ByteLevel"}}"#)
                    .utf8)
            #expect(GrammarTokenVocabulary.fromTokenizerJSON(data, tokenizerConfig: config) == nil)
        }
    }

    @Test func rejectsExcessivelySparseVocabularyBeforeAllocation() {
        for id in [1 << 22, 1_000_000_000_000] {
            let data = Data(
                (#"{"model":{"type":"BPE","vocab":{"a":0,"b":"# + String(id)
                    + #"}},"decoder":{"type":"ByteLevel"}}"#).utf8)
            #expect(GrammarTokenVocabulary.fromTokenizerJSON(data, tokenizerConfig: config) == nil)
        }
    }

    @Test func configSpecialTokenDoesNotBecomeOutputText() throws {
        let data = Data(
            #"{"model":{"type":"BPE","vocab":{"a":0,"<eos>":1}},"decoder":{"type":"ByteLevel"}}"#
                .utf8)
        let result = try #require(
            GrammarTokenVocabulary.fromTokenizerJSON(data, tokenizerConfig: config))
        #expect(result.specialTokenIDs == Set([1]))
    }
}
