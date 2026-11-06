import Foundation
import XCTest

@testable import MLXLMCommon

/// CPU-only compiler/matcher tests. No model, MLX array or Metal execution.
final class JSONSchemaGrammarTests: XCTestCase {
    private let pieces = [
        "<eos>", "{", "}", "\"", "answer", ":", "true", "false", " ", "[", "]", ",", "1", "2",
        "null", "extra", "<end>",
    ]
    private let schema =
        ##"{"type":"object","properties":{"answer":{"type":"boolean"}},"required":["answer"],"additionalProperties":false}"##

    private func tokenizer() throws -> JSONSchemaGrammarTokenizer {
        try JSONSchemaGrammarTokenizer(
            vocabulary: pieces, vocabularyType: .raw, stopTokenIDs: [0, 16])
    }
    private func allows(_ mask: JSONSchemaTokenMask, _ token: Int) -> Bool {
        (mask.words[token / 32] & (UInt32(1) << UInt32(token % 32))) != 0
    }
    private func feed(_ grammar: JSONSchemaGrammar, _ tokens: [Int]) throws {
        for token in tokens {
            XCTAssertTrue(
                allows(try grammar.nextTokenMask(), token), "Expected token \(token) admitted")
            try grammar.accept(tokenID: token)
        }
    }

    func testPropertyNamesPreserveJSONEscapingAndUnicode() throws {
        for name in [
            "quote\"key", "back\\slash", "literal\\n", "line\nfeed", "tab\tkey", "nul\0key",
            "café😀",
        ] {
            let schemaObject: [String: Any] = [
                "type": "object", "properties": [name: ["type": "boolean"]],
                "required": [name], "additionalProperties": false,
            ]
            let schema = String(
                decoding: try JSONSerialization.data(withJSONObject: schemaObject), as: UTF8.self)
            let output = String(
                decoding: try JSONSerialization.data(withJSONObject: [name: true]), as: UTF8.self)
            let wrong = String(
                decoding: try JSONSerialization.data(withJSONObject: ["different": true]),
                as: UTF8.self)
            let tokenizer = try JSONSchemaGrammarTokenizer(
                vocabulary: ["<eos>", output, wrong], vocabularyType: .raw, stopTokenIDs: [0])
            let grammar = try JSONSchemaGrammar(tokenizer: tokenizer, schema: schema)
            let initial = try grammar.nextTokenMask()
            XCTAssertTrue(allows(initial, 1), "Valid escaped property \(name.debugDescription)")
            XCTAssertFalse(allows(initial, 2), "Wrong property must remain forbidden")
            try grammar.accept(tokenID: 1)
            try grammar.accept(tokenID: 0)
            XCTAssertTrue(try grammar.isTerminated())
        }
    }

    func testNumericConstantsRejectLossySourceSpellingsRecursively() throws {
        for value in [
            "0.1", "9007199254740990.1", "9007199254740992", "18446744073709551615", "1e0",
            "[1,0.5]", "{\"x\":0.5}",
        ] {
            for keyword in ["const", "enum"] {
                let body = keyword == "enum" ? "[\(value)]" : value
                XCTAssertThrowsError(
                    try JSONSchemaGrammar.validateSupportedSchema("{\"\(keyword)\":\(body)}")
                ) { error in
                    guard case JSONSchemaGrammarError.unsupportedSchema = error else {
                        return XCTFail("Expected typed unsupported number, got \(error)")
                    }
                }
            }
        }
        for value in ["-9007199254740991", "9007199254740991", "[1,-2]", "{\"x\":3}"] {
            try JSONSchemaGrammar.validateSupportedSchema("{\"const\":\(value)}")
        }
        try JSONSchemaGrammar.validateSupportedSchema(##"{"type":"number","default":0.1}"##)
        try JSONSchemaGrammar.validateSupportedSchema(
            ##"{"type":"object","properties":{"const":{"type":"number"}},"additionalProperties":false}"##
        )
    }

    func testArrayBoundsRequirePlainIntegerSourceSpellings() throws {
        for keyword in ["minItems", "maxItems"] {
            for value in ["1.0000000000000001", "1.0", "1e0"] {
                XCTAssertThrowsError(
                    try JSONSchemaGrammar.validateSupportedSchema(
                        "{\"type\":\"array\",\"\(keyword)\":\(value)}")
                ) { error in
                    guard case JSONSchemaGrammarError.unsupportedSchema = error else {
                        return XCTFail("Expected typed unsupported bound, got \(error)")
                    }
                }
            }
            try JSONSchemaGrammar.validateSupportedSchema(
                "{\"type\":\"array\",\"\(keyword)\":1}")
        }
    }

    func testDeterministicStructuralFormattingPreservesStringWhitespace() throws {
        let schema =
            ##"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}"##
        let vocabulary = [
            "<eos>", "{", "\"answer\"", ":", "\" a  b\\n\\t \"", "}", " ", "\n", "\t",
        ]
        let tokenizer = try JSONSchemaGrammarTokenizer(
            vocabulary: vocabulary, vocabularyType: .raw, stopTokenIDs: [0])
        let grammar = try JSONSchemaGrammar(tokenizer: tokenizer, schema: schema)
        for token in [1, 2, 3, 4, 5] {
            let mask = try grammar.nextTokenMask()
            for space in [6, 7, 8] { XCTAssertFalse(allows(mask, space)) }
            XCTAssertTrue(allows(mask, token))
            try grammar.accept(tokenID: token)
        }
        let completed = try grammar.nextTokenMask()
        for space in [6, 7, 8] { XCTAssertFalse(allows(completed, space)) }
        XCTAssertTrue(allows(completed, 0))
        try grammar.accept(tokenID: 0)
        XCTAssertTrue(try grammar.isTerminated())
    }

    func testDeterministicFormattingPreservesEnumAndConstValues() throws {
        for keyword in ["enum", "const"] {
            let value: [String: Any] = [" spaced key ": " a  b\n\t "]
            let schemaValue: Any = keyword == "enum" ? [value] : value
            let schemaData = try JSONSerialization.data(withJSONObject: [keyword: schemaValue])
            let outputData = try JSONSerialization.data(withJSONObject: value)
            let output = String(decoding: outputData, as: UTF8.self)
            let tokenizer = try JSONSchemaGrammarTokenizer(
                vocabulary: ["<eos>", output, "\n"], vocabularyType: .raw, stopTokenIDs: [0])
            let grammar = try JSONSchemaGrammar(
                tokenizer: tokenizer,
                schema: String(decoding: schemaData, as: UTF8.self))
            XCTAssertTrue(allows(try grammar.nextTokenMask(), 1))
            try grammar.accept(tokenID: 1)
            XCTAssertFalse(allows(try grammar.nextTokenMask(), 2))
            try grammar.accept(tokenID: 0)
            XCTAssertTrue(try grammar.isTerminated())
        }
    }

    func testObjectMaskAndExplicitStopIDs() throws {
        let grammar = try JSONSchemaGrammar(tokenizer: tokenizer(), schema: schema)
        let initial = try grammar.nextTokenMask()
        XCTAssertEqual(initial.vocabularySize, pieces.count)
        XCTAssertFalse(allows(initial, 0))
        XCTAssertFalse(allows(initial, 16))
        XCTAssertThrowsError(try grammar.accept(tokenID: 7))
        try feed(grammar, [1, 3, 4, 3, 5, 6, 2])
        XCTAssertFalse(try grammar.isTerminated())
        let finished = try grammar.nextTokenMask()
        XCTAssertTrue(allows(finished, 0))
        XCTAssertTrue(allows(finished, 16))
        try grammar.accept(tokenID: 16)
        XCTAssertTrue(try grammar.isTerminated())
    }

    func testIndependentCopyReplaysWithoutMutatingOriginal() throws {
        let grammar = try JSONSchemaGrammar(tokenizer: tokenizer(), schema: schema)
        try feed(grammar, [1, 3, 4, 3, 5])
        let originalMask = try grammar.nextTokenMask()
        let copy = try grammar.independentCopy()
        try feed(copy, [7, 2, 0])
        XCTAssertTrue(try copy.isTerminated())
        XCTAssertFalse(try grammar.isTerminated())
        XCTAssertEqual(try grammar.nextTokenMask().words, originalMask.words)
        try feed(grammar, [6, 2, 16])
        XCTAssertTrue(try grammar.isTerminated())
        XCTAssertTrue(try grammar.independentCopy().isTerminated())
    }

    func testNamedPropertiesRequireExplicitClosedObject() throws {
        let schemas = [
            ##"{"type":"object","properties":{"answer":{"type":"boolean"}},"required":["answer"]}"##,
            ##"{"type":"object","properties":{"answer":{"type":"boolean"}},"additionalProperties":true}"##,
            ##"{"type":"object","properties":{"answer":{"type":"boolean"}},"additionalProperties":{"type":"null"}}"##,
        ]
        for schema in schemas {
            XCTAssertThrowsError(try JSONSchemaGrammar.validateSupportedSchema(schema)) { error in
                guard case JSONSchemaGrammarError.unsupportedSchema(let path, let keyword) = error
                else {
                    return XCTFail("Expected typed unsupported schema, got \(error)")
                }
                XCTAssertEqual(path, "#")
                XCTAssertEqual(keyword, "named properties require additionalProperties:false")
            }
        }
        // With the explicitly closed schema the additional-property branch is
        // absent: a second occurrence cannot bypass the named value's type.
        let grammar = try JSONSchemaGrammar(tokenizer: tokenizer(), schema: schema)
        try feed(grammar, [1, 3, 4, 3, 5, 6])
        XCTAssertFalse(allows(try grammar.nextTokenMask(), 11))
        XCTAssertThrowsError(try grammar.accept(tokenID: 11))
        try feed(grammar, [2, 0])
        XCTAssertTrue(try grammar.isTerminated())
    }

    func testOmittedAdditionalPropertiesUsesJSONSchemaDefault() throws {
        for schema in [##"{"type":"object"}"##, ##"{"type":"object","properties":{}}"##] {
            let grammar = try JSONSchemaGrammar(tokenizer: tokenizer(), schema: schema)
            try feed(grammar, [1, 3, 15, 3, 5, 6, 2, 0])
            XCTAssertTrue(try grammar.isTerminated())
        }
    }

    func testSupportedNestedSchemasAndReferences() throws {
        for schema in [
            ##"{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":2}"##,
            ##"{"type":["string","null"],"enum":["x",null]}"##,
            ##"{"anyOf":[{"type":"boolean"},{"type":"null"}]}"##,
            ##"{"$defs":{"flag":{"type":"boolean"}},"$ref":"#/$defs/flag"}"##,
            ##"{"type":"object","properties":{"not":{"type":"string"}},"required":["not"],"additionalProperties":false}"##,
        ] {
            try JSONSchemaGrammar.validateSupportedSchema(schema)
            _ = try JSONSchemaGrammar(tokenizer: tokenizer(), schema: schema)
        }
    }

    func testUnsupportedKeywordsAndCombinationsFailClosed() {
        let cases = [
            ##"{"oneOf":[{"type":"integer"},{"type":"number"}]}"##,
            ##"{"allOf":[{"type":"string"},{"minLength":3}]}"##,
            ##"{"type":"integer","multipleOf":2}"##,
            ##"{"type":"array","uniqueItems":true}"##,
            ##"{"type":"array","contains":{"type":"integer"}}"##,
            ##"{"type":"object","dependentRequired":{"a":["b"]}}"##,
            ##"{"type":"string","pattern":"a+","minLength":2}"##,
            ##"{"type":"string","format":"email"}"##,
            ##"{"type":"object","properties":{"answer":{"type":"integer","minimum":0}}}"##,
            ##"{"if":{"type":"string"},"then":{"const":"x"}}"##,
            ##"{"$ref":"https://example.invalid/schema"}"##,
            ##"{"$ref":"#/$defs/a~1b","$defs":{"a/b":{"type":"boolean"}}}"##,
            ##"{"enum":["x"],"minLength":2}"##,
            ##"{"anyOf":[{"type":"string"}],"type":"integer"}"##,
        ]
        for schema in cases {
            XCTAssertThrowsError(try JSONSchemaGrammar.validateSupportedSchema(schema), schema)
        }
    }

    func testStringLengthConstraintsAreTypedUnsupported() {
        for keyword in ["minLength", "maxLength"] {
            let schema = "{\"type\":\"string\",\"\(keyword)\":1}"
            XCTAssertThrowsError(try JSONSchemaGrammar.validateSupportedSchema(schema)) { error in
                guard case JSONSchemaGrammarError.unsupportedSchema(let path, let rejected) = error
                else {
                    return XCTFail("Expected typed unsupported schema error, got \(error)")
                }
                XCTAssertEqual(path, "#")
                XCTAssertEqual(rejected, keyword)
            }
        }
    }

    func testDialectAndReferenceTargetsFailClosed() throws {
        let rejected = [
            ##"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"string"}"##,
            ##"{"$schema":"unknown","type":"string"}"##,
            ##"{"$ref":"#"}"##,
            ##"{"$defs":{"a":{"$ref":"#/$defs/b"},"b":{"$ref":"#/$defs/a"}},"$ref":"#/$defs/a"}"##,
            ##"{"type":"object","properties":{"child":{"$ref":"#"}}}"##,
            ##"{"default":{"type":"string"},"$ref":"#/default"}"##,
            ##"{"examples":[{"type":"string"}],"$ref":"#/examples/0"}"##,
            ##"{"$defs":{"a":{"$ref":"#/$defs/missing"}},"$ref":"#/$defs/a"}"##,
        ]
        for schema in rejected {
            XCTAssertThrowsError(try JSONSchemaGrammar.validateSupportedSchema(schema)) { error in
                guard case JSONSchemaGrammarError.unsupportedSchema = error else {
                    return XCTFail("Expected unsupported schema, got \(error)")
                }
            }
        }
        try JSONSchemaGrammar.validateSupportedSchema(
            ##"{"$defs":{"a":{"$ref":"#/$defs/b"},"b":{"type":"boolean"}},"$ref":"#/$defs/a"}"##)
    }

    func testMalformedSchemasAndTokenizerFailClosed() {
        for schema in [
            "{", "[]", "42", "false", ##"{"type":42}"##,
            ##"{"type":"mystery"}"##, ##"{"type":"array","minItems":true}"##,
            ##"{"type":"array","minItems":3,"maxItems":2}"##,
            ##"{"type":"string","enum":[true]}"##, ##"{"$ref":"#/$defs/missing"}"##,
        ] {
            XCTAssertThrowsError(try JSONSchemaGrammar.validateSupportedSchema(schema), schema)
        }
        XCTAssertThrowsError(
            try JSONSchemaGrammarTokenizer(
                vocabulary: pieces, vocabularyType: .raw, stopTokenIDs: []))
        XCTAssertThrowsError(
            try JSONSchemaGrammarTokenizer(
                vocabulary: pieces, vocabularyType: .raw, stopTokenIDs: [-1]))
        XCTAssertThrowsError(
            try JSONSchemaGrammarTokenizer(
                vocabulary: ["\u{0}"], vocabularyType: .raw, stopTokenIDs: [0]))
    }
}
