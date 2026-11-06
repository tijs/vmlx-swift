import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// Synthetic vocabulary tests for the actual grammar processor. These exercise
/// MLX masking, not model quality/performance. Root runs them with other MLX tests.
final class JSONSchemaLogitProcessorTests: XCTestCase {
    private let schema = #"{"type":"boolean"}"#

    private func make(base: (any LogitProcessor)? = nil, special: Set<Int> = [0, 4, 5]) throws
        -> JSONSchemaLogitProcessor
    {
        try JSONSchemaLogitProcessor(
            schema: schema, tokenizer: SchemaProcessorTokenizer(special: special),
            stopTokenIDs: [0, 5], base: base)
    }

    func testMaskPreservesAllowedScoresAndRejectsPaddingAndControlIDs() throws {
        try MLXMetalTestLock.withLock {
            let processor = try make()
            let logits = MLXArray([Float(100), 3, 7, -2, 1000, 200, 999]).reshaped(1, 7)
            let masked = processor.process(logits: logits).asArray(Float.self)
            XCTAssertNil(processor.constraintFailure)
            XCTAssertEqual(masked[1], 3)
            XCTAssertEqual(masked[2], 7)
            for id in [0, 4, 5, 6] { XCTAssertEqual(masked[id], -.infinity, "ID \(id)") }
            XCTAssertEqual(
                ArgMaxSampler().sample(logits: MLXArray(masked).reshaped(1, 7)).item(Int.self), 2)
        }
    }

    func testExplicitSpecialIDsAreExcludedEvenWhenTheirTextMatchesGrammar() throws {
        try MLXMetalTestLock.withLock {
            let processor = try make(special: [0, 1, 4, 5])  // "true" is a control ID in this synthetic tokenizer.
            let masked = processor.process(logits: MLXArray.zeros([1, 6])).asArray(Float.self)
            XCTAssertNil(processor.constraintFailure)
            XCTAssertEqual(masked[1], -.infinity)
            XCTAssertEqual(masked[2], 0)
        }
    }

    func testIndependentCopySeparatesMatcherAndBaseMutableHistory() throws {
        try MLXMetalTestLock.withLock {
            var original = try make(base: SchemaCountingProcessor())
            var fork = original.independentCopy()
            fork.didSample(token: MLXArray(Int32(1)))
            fork.didSample(token: MLXArray(Int32(0)))
            XCTAssertTrue(fork.constraintIsComplete)
            XCTAssertFalse(original.constraintIsComplete)
            let unchanged = original.process(logits: MLXArray.zeros([1, 6])).asArray(Float.self)
            XCTAssertEqual(
                unchanged[1], 0, "Rejected speculative history must not mutate base processor")
            XCTAssertEqual(unchanged[2], 0)
            original.didSample(token: MLXArray(Int32(2)))
            XCTAssertNil(original.constraintFailure)
            XCTAssertFalse(
                original.constraintIsComplete, "Closing JSON bytes alone are not accepted EOS")
            original.didSample(token: MLXArray(Int32(5)))
            XCTAssertTrue(original.constraintIsComplete)
            XCTAssertNil(fork.constraintFailure)
        }
    }

    func testPromptDoesNotEnterGrammarAndRequestsStayIndependent() throws {
        try MLXMetalTestLock.withLock {
            var first = try make()
            let second = try make()
            first.prompt(MLXArray([Int32(4), 1, 0]))  // arbitrary chat/history, not generated JSON
            first.didSample(token: MLXArray(Int32(1)))
            XCTAssertNil(first.constraintFailure)
            let otherMask = second.process(logits: MLXArray.zeros([1, 6])).asArray(Float.self)
            XCTAssertEqual(otherMask[1], 0)
            XCTAssertEqual(otherMask[2], 0)
            XCTAssertFalse(second.constraintIsComplete)
        }
    }

    func testEOSIsAdmittedOnlyAfterJSONAndLookaheadStaysTerminal() throws {
        try MLXMetalTestLock.withLock {
            var processor = try make()
            let initial = processor.process(logits: MLXArray.zeros([1, 6])).asArray(Float.self)
            XCTAssertEqual(initial[0], -.infinity)
            processor.didSample(token: MLXArray(Int32(1)))
            let completeJSON = processor.process(logits: MLXArray.zeros([1, 6])).asArray(Float.self)
            XCTAssertEqual(completeJSON[0], 0)
            XCTAssertEqual(completeJSON[5], 0)
            XCTAssertFalse(processor.constraintIsComplete)
            processor.didSample(token: MLXArray(Int32(0)))
            XCTAssertTrue(processor.isTerminated)
            let lookahead = processor.process(logits: MLXArray.zeros([1, 6])).asArray(Float.self)
            for id in [1, 2, 3, 4] { XCTAssertEqual(lookahead[id], -.infinity) }
            processor.didSample(token: MLXArray(Int32(0)))
            XCTAssertTrue(processor.constraintIsComplete)
            XCTAssertNil(processor.constraintFailure)
        }
    }

    func testExhaustedIntersectionAndInvalidTokenReportStickyFailure() throws {
        try MLXMetalTestLock.withLock {
            let exhausted = try make(base: SchemaSuppressAllProcessor())
            _ = exhausted.process(logits: MLXArray.zeros([1, 6]))
            XCTAssertNotNil(
                exhausted.constraintFailure,
                "Caller must stop instead of sampling an empty distribution")
            XCTAssertFalse(exhausted.constraintIsComplete)
            var invalid = try make()
            invalid.didSample(token: MLXArray(Int32(4)))
            XCTAssertNotNil(invalid.constraintFailure)
            invalid.didSample(token: MLXArray(Int32(1)))
            XCTAssertNotNil(invalid.constraintFailure)
            XCTAssertFalse(invalid.constraintIsComplete)
            XCTAssertNotNil(invalid.independentCopy().constraintFailure)
        }
    }

    func testAllowedDistributionRetainsRequestedTemperature() throws {
        try MLXMetalTestLock.withLock {
            let processor = try make()
            let logits = processor.process(
                logits: MLXArray([Float(100), 2, 4, -Float.infinity, 100, 100]).reshaped(1, 6))
            XCTAssertNil(processor.constraintFailure)
            let controller = SpeculativeSamplingController(
                parameters: GenerateParameters(temperature: 0.7, topP: 1, topK: 0))
            let probabilities = controller.probabilities(logits: logits).asArray(Float.self)
            XCTAssertEqual(
                probabilities[2] / probabilities[1], exp(Float(2) / 0.7), accuracy: 0.0001)
            for id in [0, 3, 4, 5] { XCTAssertEqual(probabilities[id], 0) }
        }
    }

    func testMissingVocabularyFailsBeforeGeneration() {
        XCTAssertThrowsError(
            try JSONSchemaLogitProcessor(
                schema: schema,
                tokenizer: SchemaProcessorTokenizer(special: [], grammarSupported: false),
                stopTokenIDs: [0]))
    }
}

private struct SchemaProcessorTokenizer: MLXLMCommon.Tokenizer {
    let special: Set<Int>
    let grammarSupported: Bool
    init(special: Set<Int>, grammarSupported: Bool = true) {
        self.special = special
        self.grammarSupported = grammarSupported
    }
    private let pieces = ["<eos>", "true", "false", " ", "<tool>", "<end>"]
    var grammarTokenVocabulary: GrammarTokenVocabulary? {
        guard grammarSupported else { return nil }
        return .init(vocabulary: pieces, vocabularyType: .raw, specialTokenIDs: special)
    }
    var bosToken: String? { nil }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { pieces[$0] }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { pieces.firstIndex(of: token) }
    func convertIdToToken(_ id: Int) -> String? { pieces.indices.contains(id) ? pieces[id] : nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

private struct SchemaCountingProcessor: LogitProcessor {
    private final class Counter { var count = 0 }
    private var counter = Counter()
    func independentCopy() -> Self {
        var copy = self
        copy.counter = Counter()
        copy.counter.count = counter.count
        return copy
    }
    mutating func prompt(_ prompt: MLXArray) {}
    func process(logits: MLXArray) -> MLXArray { logits - Float(counter.count) }
    mutating func didSample(token: MLXArray) { counter.count += 1 }
}

private struct SchemaSuppressAllProcessor: LogitProcessor {
    mutating func prompt(_ prompt: MLXArray) {}
    func process(logits: MLXArray) -> MLXArray {
        MLXArray.full(logits.shape, values: MLXArray(-Float.infinity))
    }
    mutating func didSample(token: MLXArray) {}
}
