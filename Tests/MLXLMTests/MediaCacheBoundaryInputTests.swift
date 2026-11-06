import MLX
import XCTest

@testable import MLXLMCommon

final class MediaCacheBoundaryInputTests: XCTestCase {
    private let ids = [1, 2, 90, 90, 3, 91, 91, 4]

    private func input(declared: [Int]? = [90, 91], prunedVideo: Bool = false,
                       mask: MLXArray? = nil, audio: Bool = false) -> LMInput {
        LMInput(text: .init(tokens: MLXArray(ids.map(Int32.init)).reshaped(1, ids.count), mask: mask),
                image: .init(pixels: MLXArray.zeros([4, 12]), frames: [THW(1, 2, 2)]),
                video: .init(pixels: MLXArray.ones([4, 12]), frames: [THW(1, 2, 2)],
                             embeddingTokenCount: prunedVideo ? 2 : nil),
                audio: audio ? .init(waveform: MLXArray.zeros([1, 128])) : nil,
                mediaTokenIds: declared, cacheScopeSalt: "reasoning=on",
                cacheRestorePolicy: .freshRequiredToolSelection,
                toolSchemas: [["type": "function", "function": ["name": "file_read"]]])
    }

    func testPrefixBeforeMediaDropsTowerPayloadsPreservingRequestContract() throws {
        let head = try XCTUnwrap(input(audio: true).inputForCacheBoundary(
            tokens: [1, 2], fullPromptTokenIds: ids))
        XCTAssertFalse(head.hasMediaContent)
        XCTAssertEqual(head.text.tokens.shape, [1, 2])
        XCTAssertEqual(head.text.tokenIds, [1, 2])
        XCTAssertEqual(head.cacheScopeSalt, "reasoning=on")
        XCTAssertEqual(head.cacheRestorePolicy, .freshRequiredToolSelection)
        XCTAssertNotNil(head.toolSchemas)
        XCTAssertEqual(head.mediaTokenIds, [90, 91])
    }

    func testCompleteMediaKeepsOrderedImageVideoAndAudioPayloads() throws {
        let original = input(audio: true)
        let head = try XCTUnwrap(original.inputForCacheBoundary(
            tokens: Array(ids.prefix(7)), fullPromptTokenIds: ids))
        XCTAssertTrue(head.image?.pixels === original.image?.pixels)
        XCTAssertTrue(head.video?.pixels === original.video?.pixels)
        XCTAssertTrue(head.audio?.waveform === original.audio?.waveform)
    }

    func testInsideImageVideoOrBetweenDistinctMediaFailsClosed() {
        for length in [3, 4, 5, 6] {
            XCTAssertNil(input().inputForCacheBoundary(
                tokens: Array(ids.prefix(length)), fullPromptTokenIds: ids))
        }
    }

    func testUnknownPlaceholderContractAndPrunedVideoFailClosed() {
        for original in [input(declared: nil), input(declared: []), input(prunedVideo: true)] {
            XCTAssertNil(original.inputForCacheBoundary(tokens: ids, fullPromptTokenIds: ids))
        }
        XCTAssertNil(input().inputForCacheBoundary(tokens: [9, 2], fullPromptTokenIds: ids))
        XCTAssertNil(input().inputForCacheBoundary(tokens: [], fullPromptTokenIds: ids))
    }

    func testTextOnlyCanonicalTailRetainsTokensAndSettings() throws {
        let original = LMInput(tokens: MLXArray([Int32(1), 2, 3]))
            .withCacheRestorePolicy(.freshRequiredToolSelection)
        let tail = try XCTUnwrap(original.inputForCacheBoundary(tokens: [3], fullPromptTokenIds: [1, 2, 3]))
        XCTAssertEqual(tail.text.tokenIds, [3])
        XCTAssertFalse(tail.hasMediaContent)
        XCTAssertEqual(tail.cacheRestorePolicy, .freshRequiredToolSelection)
    }

    func testTokenAlignedMaskSlicesAndOpaqueAttentionMaskDeclines() throws {
        let head = try XCTUnwrap(input(mask: MLXArray.ones([1, ids.count])).inputForCacheBoundary(
            tokens: [1, 2], fullPromptTokenIds: ids))
        XCTAssertEqual(head.text.mask?.shape, [1, 2])
        XCTAssertNil(input(mask: MLXArray.ones([1, 1, ids.count, ids.count]))
            .inputForCacheBoundary(tokens: [1, 2], fullPromptTokenIds: ids))
    }
}
