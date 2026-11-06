import Foundation
import MLXLMCommon
import XCTest

private struct FailureTransportTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

final class GenerationFailureTransportTests: XCTestCase {
    func testDeferredFailureKeepsOriginatingDiagnostic() async throws {
        try await MLXMetalTestLock.withLock {
            let original = NSError(
                domain: "fixture", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "fixture projection shape mismatch"])
            let (stream, task) = generateTaskDeferred(
                promptTokenCount: 8, modelConfiguration: ModelConfiguration(id: "fixture/model"),
                tokenizer: FailureTransportTokenizer(), promptTokenIds: [1],
                makeIterator: { throw original })
            var infos: [GenerateCompletionInfo] = []
            for await event in stream {
                guard case .info(let info) = event else {
                    return XCTFail("Unexpected model output")
                }
                infos.append(info)
            }
            await task.value
            XCTAssertEqual(infos.count, 1)
            XCTAssertEqual(
                infos.first?.generationFailure,
                GenerationFailure(stage: .preparation, cause: original.localizedDescription))
            XCTAssertEqual(infos.first?.generationTokenCount, 0)
        }
    }

    func testExplicitCancellationDoesNotBecomeGenerationFailure() async throws {
        try await MLXMetalTestLock.withLock {
            let (stream, task) = generateTaskDeferred(
                promptTokenCount: 8, modelConfiguration: ModelConfiguration(id: "fixture/model"),
                tokenizer: FailureTransportTokenizer(), promptTokenIds: [1],
                makeIterator: { throw CancellationError() })
            var infos: [GenerateCompletionInfo] = []
            for await event in stream {
                guard case .info(let info) = event else {
                    return XCTFail("Unexpected model output")
                }
                infos.append(info)
            }
            await task.value
            XCTAssertEqual(infos.count, 1)
            XCTAssertNil(infos.first?.generationFailure)
            guard case .cancelled = infos.first?.stopReason else {
                return XCTFail("Expected cancellation")
            }
        }
    }
}
