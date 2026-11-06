import Foundation
import MLXHuggingFace
import VMLXHub
@testable import VMLXTokenizers
@testable import MLXLMCommon
import XCTest

final class IncrementalByteLevelDetokenizerTests: XCTestCase {
    final class DerivedTokenizer: PreTrainedTokenizer, @unchecked Sendable {}

    private func fixture(cleanup: Bool? = false, decoder: String = "ByteLevel",
                         derived: Bool = false) throws -> PreTrainedTokenizer {
        var bytes = Array(33...126) + Array(161...172) + Array(174...255)
        var scalars = bytes
        var extra = 0
        for byte in 0...255 where !bytes.contains(byte) {
            bytes.append(byte)
            scalars.append(256 + extra)
            extra += 1
        }
        var vocab: [String: Int] = [:]
        for (byte, scalar) in zip(bytes, scalars) { vocab[String(Unicode.Scalar(scalar)!)] = byte }
        var config: [NSString: Any] = ["tokenizer_class": "GPT2Tokenizer"]
        if let cleanup { config["clean_up_tokenization_spaces"] = cleanup }
        let decoderConfig: [String: Any] = decoder == "Sequence"
            ? ["type": "Sequence", "decoders": [["type": "ByteLevel"]]]
            : ["type": decoder]
        let data: [NSString: Any] = [
            "model": ["type": "BPE", "vocab": vocab, "merges": []] as [String: Any],
            "decoder": decoderConfig,
            "added_tokens": [
                ["id": 256, "content": "<ifm|think>", "special": true] as [String: Any],
                ["id": 257, "content": "<literal-added>", "special": false] as [String: Any],
            ],
        ]
        if derived { return try DerivedTokenizer(tokenizerConfig: Config(config), tokenizerData: Config(data)) }
        return try PreTrainedTokenizer(tokenizerConfig: Config(config), tokenizerData: Config(data))
    }

    func testCapabilityRequiresPlainByteLevelWithoutCleanupAndExactClass() throws {
        let accepted = try fixture()
        XCTAssertNotNil(accepted.incrementalByteLevelDecoder)
        let bridge = #adaptHuggingFaceTokenizer(accepted)
        XCTAssertNotNil(bridge.incrementalByteLevelDecoder)
        for rejected in [try fixture(cleanup: true), try fixture(cleanup: nil),
                         try fixture(decoder: "Sequence"), try fixture(decoder: "Fuse"),
                         try fixture(derived: true)] {
            XCTAssertNil(rejected.incrementalByteLevelDecoder)
            let bridged = #adaptHuggingFaceTokenizer(rejected)
            XCTAssertNil(bridged.incrementalByteLevelDecoder)
        }
    }

    private func stream(_ ids: [Int], tokenizer: PreTrainedTokenizer) -> [String] {
        let bridge = #adaptHuggingFaceTokenizer(tokenizer)
        var detokenizer = NaiveStreamingDetokenizer(tokenizer: bridge)
        var chunks: [String] = []
        for id in ids {
            detokenizer.append(token: id)
            if let chunk = detokenizer.next() { chunks.append(chunk) }
        }
        if let chunk = detokenizer.flush() { chunks.append(chunk) }
        XCTAssertNil(detokenizer.flush())
        return chunks
    }

    func testByteRunsAddedTokensUnknownIDsAndMalformedUTF8MatchUpstream() throws {
        let tokenizer = try fixture()
        let prefix = Array("A sufficiently long stable prefix before bytes. ".utf8).map(Int.init)
        let cases = [[0xE2, 0x82, 256, 0xAC], [0xF0, 0x9F, 257, 0x91, 0xA9],
                     [256, 257, Int.max, 256], [0xE0, 0x80, 0x80], [0xED, 0xA0, 0x80],
                     [0xF4, 0x90, 0x80, 0x80], [0xE2, 0x41, 0x82], [0xEF, 0xBF, 0xBD]]
        for middle in cases + (0...255).map({ [$0] }) {
            let ids = prefix + middle + [0x41]
            XCTAssertEqual(stream(ids, tokenizer: tokenizer).joined(), tokenizer.decode(tokens: ids, skipSpecialTokens: false))
        }
    }

    func testUnicodeHoldbackNewlinesAndProtocolLiteralsRemainExact() throws {
        let tokenizer = try fixture()
        let texts = [
            String(repeating: "🇺🇸 👩🏽‍💻 👨‍👩‍👧‍👦 e\u{301} ✈️ \r\n", count: 12),
            "12345678901234567890\nabcdefghijklmnopqrstuvwxYZ",
            "<ifm|think>Reason</ifm|think><ifm|tool_call>{\"name\":\"read\",\"arguments\":{\"account\":\"00123\",\"include_tax\":\"false\",\"status\":\"null\"}}</ifm|tool_call>",
        ]
        for text in texts {
            let chunks = stream(Array(text.utf8).map(Int.init), tokenizer: tokenizer)
            XCTAssertEqual(chunks.joined(), text)
            XCTAssertEqual(chunks.joined().count, chunks.reduce(0) { $0 + $1.count })
        }
    }

    func testFlushDefersIncompleteBytesAndSupportsSubsequentAppend() throws {
        let tokenizer = try fixture()
        let bridge = #adaptHuggingFaceTokenizer(tokenizer)
        var detokenizer = NaiveStreamingDetokenizer(tokenizer: bridge)
        detokenizer.append(token: 0xE2)
        detokenizer.append(token: 0x82)
        XCTAssertNil(detokenizer.next())
        XCTAssertNil(detokenizer.flush())
        XCTAssertNil(detokenizer.flush())
        detokenizer.append(token: 0xAC)
        XCTAssertEqual(detokenizer.flush(), "€")
        XCTAssertNil(detokenizer.flush())
        for byte in "again 🇺🇸".utf8 { detokenizer.append(token: Int(byte)) }
        XCTAssertEqual(detokenizer.flush(), "again 🇺🇸")
        XCTAssertNil(detokenizer.flush())
    }
}
