import Foundation
import XCTest
import MLXLMCommon
@testable import MLXVLM

final class QwenVLMediaBoundaryTests: XCTestCase {
    private struct Bytes: MLXLMCommon.Tokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            String(decoding: tokenIds.map(UInt8.init), as: UTF8.self)
        }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] { [] }
    }

    func testImageAndVideoBoundariesSurviveExpansion() throws {
        let tokenizer = Bytes()
        let image = "<|vision_start|><|image_pad|><|vision_end|>"
        let video = "<|vision_start|><|video_pad|><|vision_end|>"
        let history = "system user " + image + " then " + video + " end"
        let raw = tokenizer.encode(text: history + " assistant", addSpecialTokens: false)
        let initial = CanonicalChatCacheBoundaries(all: [7, history.utf8.count], stable: [7])
        let images = try QwenVL.expandPaddingTokens(
            in: raw, frames: [THW(1, 4, 4)], paddingToken: "<|image_pad|>",
            mergeSize: 2, tokenizer: tokenizer, boundaries: initial)
        let expanded = try QwenVL.expandPaddingTokens(
            in: images.tokens, frames: [THW(2, 4, 4)], paddingToken: "<|video_pad|>",
            mergeSize: 2, tokenizer: tokenizer, boundaries: images.boundaries)
        let expectedHistory = history
            .replacingOccurrences(of: "<|image_pad|>", with: String(repeating: "<|image_pad|>", count: 4))
            .replacingOccurrences(of: "<|video_pad|>", with: String(repeating: "<|video_pad|>", count: 8))
        XCTAssertEqual(expanded.boundaries.all, [7, expectedHistory.utf8.count])
        XCTAssertEqual(expanded.boundaries.stable, [7])
        XCTAssertEqual(Array(expanded.tokens.prefix(expectedHistory.utf8.count)),
                       tokenizer.encode(text: expectedHistory, addSpecialTokens: false))
        XCTAssertEqual(tokenizer.decode(tokenIds: expanded.tokens, skipSpecialTokens: false), expectedHistory + " assistant")
    }

    func testBoundaryInsideMediaIsRejected() throws {
        let tokenizer = Bytes()
        let placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
        let raw = tokenizer.encode(text: "a" + placeholder + "z", addSpecialTokens: false)
        let end = 1 + placeholder.utf8.count
        let expanded = try QwenVL.expandPaddingTokens(
            in: raw, frames: [THW(1, 4, 4)], paddingToken: "<|image_pad|>",
            mergeSize: 2, tokenizer: tokenizer,
            boundaries: .init(all: [0, 1, 2, end, raw.count], stable: [2]))
        XCTAssertEqual(expanded.boundaries.all, [1, expanded.tokens.count - 1])
        XCTAssertEqual(expanded.boundaries.stable, [])
    }
}
