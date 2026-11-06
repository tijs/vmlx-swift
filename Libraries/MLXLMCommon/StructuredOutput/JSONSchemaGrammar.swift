import CoreFoundation
import Foundation
import MLXCXGrammar

public enum JSONSchemaVocabularyType: String, Sendable {
    case raw, byteFallback, byteLevel

    fileprivate var cValue: XGVocabType {
        switch self {
        case .raw: XG_VOCAB_TYPE_RAW
        case .byteFallback: XG_VOCAB_TYPE_BYTE_FALLBACK
        case .byteLevel: XG_VOCAB_TYPE_BYTE_LEVEL
        }
    }
}

public enum JSONSchemaGrammarError: Error, LocalizedError, Sendable {
    case invalidTokenizer(String)
    case invalidSchema(String)
    case unsupportedSchema(path: String, keyword: String)
    case runtime(String)

    public var errorDescription: String? {
        switch self {
        case .invalidTokenizer(let message), .invalidSchema(let message), .runtime(let message):
            message
        case .unsupportedSchema(let path, let keyword):
            "Unsupported JSON Schema keyword or combination at \(path): \(keyword)"
        }
    }
}

private func grammarError(_ operation: String) -> JSONSchemaGrammarError {
    let detail = xg_last_error_message().map { String(cString: $0) } ?? "unknown grammar error"
    return .runtime("\(operation): \(detail)")
}

/// Immutable vocabulary information; raw pieces are decoded exactly once by XGrammar.
/// Empty pieces are explicit unavailable/special vocabulary slots, never guessed text.
public final class JSONSchemaGrammarTokenizer: @unchecked Sendable {
    fileprivate let handle: OpaquePointer
    public let vocabularySize: Int

    public init(vocabulary: [String], vocabularyType: JSONSchemaVocabularyType, stopTokenIDs: [Int])
        throws
    {
        guard !vocabulary.isEmpty, vocabulary.count <= Int(Int32.max),
            !vocabulary.contains(where: { $0.utf8.contains(0) }), !stopTokenIDs.isEmpty,
            stopTokenIDs.allSatisfy({ vocabulary.indices.contains($0) })
        else {
            throw JSONSchemaGrammarError.invalidTokenizer(
                "Vocabulary and explicit stop-token IDs must be valid; embedded NUL is unsupported")
        }
        vocabularySize = vocabulary.count
        var strings: [UnsafeMutablePointer<CChar>] = []
        defer { for string in strings { free(string) } }
        for piece in vocabulary {
            guard let string = strdup(piece) else {
                throw JSONSchemaGrammarError.invalidTokenizer("Cannot allocate grammar vocabulary")
            }
            strings.append(string)
        }
        let pointers: [UnsafePointer<CChar>?] = strings.map { UnsafePointer($0) }
        let stops = Array(Set(stopTokenIDs)).sorted().map(Int32.init)
        var result: OpaquePointer?
        let status = pointers.withUnsafeBufferPointer { pieces in
            stops.withUnsafeBufferPointer { ids in
                xg_tokenizer_info_new(
                    pieces.baseAddress, pieces.count, vocabularyType.cValue,
                    ids.baseAddress, ids.count, &result)
            }
        }
        guard status == XG_OK, let result else { throw grammarError("tokenizer creation") }
        handle = result
    }

    deinit { xg_tokenizer_info_free(handle) }
}

public struct JSONSchemaTokenMask: Sendable {
    public let words: [UInt32]
    public let vocabularySize: Int
    public let needsApply: Bool
}

private final class JSONSchemaCompiledOwner: @unchecked Sendable {
    let tokenizer: JSONSchemaGrammarTokenizer
    let compiler: OpaquePointer
    let compiled: OpaquePointer

    init(tokenizer: JSONSchemaGrammarTokenizer, schema: String) throws {
        var c: OpaquePointer?
        guard xg_grammar_compiler_new(tokenizer.handle, &c) == XG_OK, let c else {
            throw grammarError("compiler creation")
        }
        var g: OpaquePointer?
        let status = schema.withCString { xg_compile_json_schema(c, $0, &g) }
        guard status == XG_OK, let g else {
            let error = grammarError("schema compilation")
            xg_grammar_compiler_free(c)
            throw error
        }
        self.tokenizer = tokenizer
        compiler = c
        compiled = g
    }
    deinit {
        xg_compiled_grammar_free(compiled)
        xg_grammar_compiler_free(compiler)
    }
}

/// One request's grammar state. Copies replay accepted IDs into a fresh matcher;
/// they never use the unavailable XGrammar 0.1.30 fork or share mutable state.
public final class JSONSchemaGrammar: @unchecked Sendable {
    private let owner: JSONSchemaCompiledOwner
    private let matcher: OpaquePointer
    private let lock = NSLock()
    private var accepted: [Int32]

    public convenience init(tokenizer: JSONSchemaGrammarTokenizer, schema: String) throws {
        try Self.validateSupportedSchema(schema)
        let owner = try JSONSchemaCompiledOwner(tokenizer: tokenizer, schema: schema)
        try self.init(owner: owner, history: [])
    }

    private init(owner: JSONSchemaCompiledOwner, history: [Int32]) throws {
        var result: OpaquePointer?
        guard xg_matcher_new(owner.compiled, &result) == XG_OK, let result else {
            throw grammarError("matcher creation")
        }
        for token in history {
            guard xg_matcher_accept_token(result, token) == XG_OK else {
                let error = grammarError("matcher replay")
                xg_matcher_free(result)
                throw error
            }
        }
        self.owner = owner
        matcher = result
        accepted = history
    }
    deinit { xg_matcher_free(matcher) }

    public func nextTokenMask() throws -> JSONSchemaTokenMask {
        lock.lock()
        defer { lock.unlock() }
        let size = owner.tokenizer.vocabularySize
        var words = [Int32](repeating: 0, count: (size + 31) / 32)
        var needsApply: Int32 = 0
        let status = words.withUnsafeMutableBufferPointer {
            xg_matcher_fill_next_token_bitmask(
                matcher, $0.baseAddress, $0.count, Int32(size), &needsApply)
        }
        guard status == XG_OK else { throw grammarError("mask computation") }
        return JSONSchemaTokenMask(
            words: words.map(UInt32.init(bitPattern:)),
            vocabularySize: size, needsApply: needsApply != 0)
    }

    public func accept(tokenID: Int) throws {
        lock.lock()
        defer { lock.unlock() }
        guard (0 ..< owner.tokenizer.vocabularySize).contains(tokenID) else {
            throw JSONSchemaGrammarError.runtime("Token ID outside grammar vocabulary")
        }
        let id = Int32(tokenID)
        guard xg_matcher_accept_token(matcher, id) == XG_OK else {
            throw grammarError("token rejected")
        }
        accepted.append(id)
    }

    public func isTerminated() throws -> Bool {
        lock.lock()
        defer { lock.unlock() }
        var result: Int32 = 0
        guard xg_matcher_is_terminated(matcher, &result) == XG_OK else {
            throw grammarError("termination query")
        }
        return result != 0
    }

    public func independentCopy() throws -> JSONSchemaGrammar {
        lock.lock()
        defer { lock.unlock() }
        return try JSONSchemaGrammar(owner: owner, history: accepted)
    }

    /// Fail closed on constraints the pinned compiler ignores or approximates.
    /// The admitted subset is intentionally narrower than general JSON Schema.
    public static func validateSupportedSchema(_ schema: String) throws {
        guard let bytes = schema.data(using: .utf8), bytes.count <= 1_048_576,
            !schema.utf8.contains(0)
        else {
            throw JSONSchemaGrammarError.invalidSchema(
                "Schema must be UTF-8, contain no NUL and fit within 1 MiB")
        }
        let root: Any
        do {
            root = try JSONSerialization.jsonObject(with: bytes, options: [.fragmentsAllowed])
        } catch {
            throw JSONSchemaGrammarError.invalidSchema(
                "Invalid JSON Schema JSON: \(error.localizedDescription)")
        }
        var nodes = 0
        var refs: [(String, String)] = []
        var schemaPaths = Set<String>()
        var dependencies: [String: Set<String>] = [:]
        var parents: [String] = []
        func component(_ name: String) -> String {
            name.replacingOccurrences(of: "~", with: "~0").replacingOccurrences(of: "/", with: "~1")
        }
        let annotations: Set<String> = ["title", "description", "$comment", "default", "examples"]
        let definitions: Set<String> = ["$defs", "definitions"]
        let allowed = annotations.union(definitions).union([
            "type", "properties", "required", "additionalProperties", "items", "minItems",
            "maxItems",
            "enum", "const", "anyOf", "$ref",
        ])
        let types: Set<String> = [
            "object", "array", "string", "integer", "number", "boolean", "null",
        ]
        func fail(_ path: String, _ keyword: String) throws {
            throw JSONSchemaGrammarError.unsupportedSchema(path: path, keyword: keyword)
        }
        func boolean(_ value: Any) -> Bool? {
            guard let n = value as? NSNumber, CFGetTypeID(n) == CFBooleanGetTypeID() else {
                return nil
            }
            return n.boolValue
        }
        func matches(_ value: Any, _ type: String) -> Bool {
            switch type {
            case "null": return value is NSNull
            case "boolean": return boolean(value) != nil
            case "string": return value is String
            case "array": return value is [Any]
            case "object": return value is [String: Any]
            case "number", "integer":
                guard let n = value as? NSNumber, boolean(value) == nil else { return false }
                return n.doubleValue.isFinite
                    && (type == "number" || n.doubleValue.rounded() == n.doubleValue)
            default: return false
            }
        }
        func walk(_ node: Any, _ path: String, _ depth: Int) throws {
            schemaPaths.insert(path)
            if let parent = parents.last { dependencies[parent, default: []].insert(path) }
            parents.append(path)
            defer { parents.removeLast() }
            nodes += 1
            guard depth <= 64, nodes <= 10_000 else {
                throw JSONSchemaGrammarError.invalidSchema("Schema complexity limit exceeded")
            }
            if let b = boolean(node) {
                guard b else {
                    try fail(path, "false schema")
                    return
                }
                return
            }
            guard let object = node as? [String: Any] else {
                throw JSONSchemaGrammarError.invalidSchema("Expected schema object at \(path)")
            }
            for key in object.keys where !allowed.contains(key) { try fail(path, key) }
            for key in definitions where object[key] != nil {
                guard let defs = object[key] as? [String: Any] else {
                    throw JSONSchemaGrammarError.invalidSchema("Invalid \(key) at \(path)")
                }
                for (name, child) in defs {
                    try walk(child, path + "/" + key + "/" + component(name), depth + 1)
                }
            }
            let constraints = Set(object.keys).subtracting(annotations).subtracting(definitions)
            if let ref = object["$ref"] {
                guard constraints == ["$ref"], let ref = ref as? String,
                    ref == "#"
                        || (ref.hasPrefix("#/") && !ref.contains("~") && !ref.contains("%")
                            && !ref.dropFirst(2).split(
                                separator: "/", omittingEmptySubsequences: false
                            ).contains(where: { $0.isEmpty }))
                else {
                    try fail(path, "$ref or constraint siblings")
                    return
                }
                refs.append((ref, path))
                return
            }
            if let any = object["anyOf"] {
                guard constraints == ["anyOf"], let branches = any as? [Any], !branches.isEmpty,
                    branches.count <= 128
                else {
                    try fail(path, "anyOf or constraint siblings")
                    return
                }
                for (i, branch) in branches.enumerated() {
                    try walk(branch, path + "/anyOf/\(i)", depth + 1)
                }
                return
            }
            var declared: [String] = []
            if let type = object["type"] {
                if let single = type as? String {
                    declared = [single]
                } else if let multiple = type as? [String] {
                    declared = multiple
                }
                guard !declared.isEmpty, Set(declared).count == declared.count,
                    declared.allSatisfy(types.contains)
                else {
                    throw JSONSchemaGrammarError.invalidSchema("Invalid type at \(path)")
                }
            }
            for key in ["enum", "const"] where object[key] != nil {
                guard constraints.isSubset(of: [key, "type"]) else {
                    try fail(path, key + " with constraint siblings")
                    return
                }
                let values: [Any]
                if key == "enum" {
                    guard let list = object[key] as? [Any], !list.isEmpty, list.count <= 1024 else {
                        throw JSONSchemaGrammarError.invalidSchema("Invalid enum at \(path)")
                    }
                    values = list
                } else {
                    values = [object[key]!]
                }
                guard
                    declared.isEmpty
                        || values.allSatisfy({ v in declared.contains { matches(v, $0) } })
                else {
                    try fail(path, key + " inconsistent with type")
                    return
                }
                return
            }
            // Explicit types avoid the converter treating unrecognized standalone constraints as 'any'.
            let objectKeys: Set<String> = ["properties", "required", "additionalProperties"]
            let arrayKeys: Set<String> = ["items", "minItems", "maxItems"]
            for (keys, type) in [(objectKeys, "object"), (arrayKeys, "array")] {
                if !constraints.isDisjoint(with: keys), !declared.contains(type) {
                    try fail(path, "constraints without explicit " + type + " type")
                }
            }
            if let properties = object["properties"] {
                guard let properties = properties as? [String: Any] else {
                    throw JSONSchemaGrammarError.invalidSchema("Invalid properties at \(path)")
                }
                // The donor additional-property grammar can repeat a named key
                // with an unconstrained value. Require explicit closure; never
                // silently change the caller's additionalProperties semantics.
                if !properties.isEmpty,
                    object["additionalProperties"].flatMap(boolean) != false
                {
                    try fail(path, "named properties require additionalProperties:false")
                }
                for (key, child) in properties {
                    try walk(child, path + "/properties/" + component(key), depth + 1)
                }
            }
            if let required = object["required"] {
                guard let required = required as? [String], Set(required).count == required.count,
                    let properties = object["properties"] as? [String: Any],
                    required.allSatisfy({ properties[$0] != nil })
                else {
                    try fail(path, "required without declared properties")
                    return
                }
            }
            if let additional = object["additionalProperties"] {
                if boolean(additional) == nil {
                    try walk(additional, path + "/additionalProperties", depth + 1)
                }
            }
            if let items = object["items"] { try walk(items, path + "/items", depth + 1) }
            for (low, high, cap) in [("minItems", "maxItems", 1024)] {
                var bounds: [String: Int] = [:]
                for key in [low, high] where object[key] != nil {
                    guard let n = object[key] as? NSNumber, boolean(n) == nil,
                        n.doubleValue.isFinite, n.doubleValue.rounded() == n.doubleValue,
                        n.doubleValue >= 0, n.doubleValue <= Double(cap)
                    else {
                        throw JSONSchemaGrammarError.invalidSchema(
                            "Invalid or excessive \(key) at \(path)")
                    }
                    bounds[key] = n.intValue
                }
                if let a = bounds[low], let b = bounds[high], a > b {
                    throw JSONSchemaGrammarError.invalidSchema("Inverted bounds at \(path)")
                }
            }
        }
        try walk(root, "#", 0)
        // Inspect numeric source spellings before Foundation/picojson can round
        // them. Enum/const values and array bounds require plain integer literals.
        let source = Array(bytes)
        var cursor = 0
        func skipSpace() {
            while cursor < source.count, [UInt8(32), 9, 10, 13].contains(source[cursor]) {
                cursor += 1
            }
        }
        func stringToken() throws -> String {
            let start = cursor
            cursor += 1
            while cursor < source.count {
                let byte = source[cursor]
                cursor += 1
                if byte == 92 { cursor += 1 } else if byte == 34 { break }
            }
            return try JSONSerialization.jsonObject(
                with: Data(source[start ..< cursor]), options: [.fragmentsAllowed]) as! String
        }
        func inspectNumbers(_ exact: Bool, _ path: String) throws {
            skipSpace()
            guard cursor < source.count else { return }
            switch source[cursor] {
            case 34: _ = try stringToken()
            case 123:
                cursor += 1
                skipSpace()
                while cursor < source.count, source[cursor] != 125 {
                    let key = try stringToken()
                    skipSpace()
                    cursor += 1
                    try inspectNumbers(
                        exact
                            || (schemaPaths.contains(path)
                                && ["enum", "const", "minItems", "maxItems"].contains(key)),
                        path + "/" + component(key))
                    skipSpace()
                    if cursor < source.count, source[cursor] == 44 {
                        cursor += 1
                        skipSpace()
                    } else {
                        break
                    }
                }
                cursor += 1
            case 91:
                cursor += 1
                skipSpace()
                var index = 0
                while cursor < source.count, source[cursor] != 93 {
                    try inspectNumbers(exact, path + "/" + String(index))
                    index += 1
                    skipSpace()
                    if cursor < source.count, source[cursor] == 44 { cursor += 1 } else { break }
                }
                cursor += 1
            default:
                let start = cursor
                while cursor < source.count,
                    ![UInt8(32), 9, 10, 13, 44, 93, 125].contains(source[cursor])
                { cursor += 1 }
                let token = String(decoding: source[start ..< cursor], as: UTF8.self)
                if exact, let first = token.first, first == "-" || first.isNumber {
                    guard let integer = Int64(token),
                        (-9_007_199_254_740_991 ... 9_007_199_254_740_991).contains(integer)
                    else {
                        throw JSONSchemaGrammarError.unsupportedSchema(
                            path: "#",
                            keyword:
                                "enum/const numbers and array bounds require safe integer literals")
                    }
                }
            }
        }
        try inspectNumbers(false, "#")
        // Resolve only paths already visited as schemas. Annotation/instance-data
        // objects must never become schemas merely because a reference names them.
        for (ref, path) in refs {
            guard schemaPaths.contains(ref) else {
                try fail(path, "unresolved or non-schema $ref target")
                return
            }
            dependencies[path, default: []].insert(ref)
        }
        var visiting = Set<String>()
        var visited = Set<String>()
        func checkAcyclic(_ path: String, _ depth: Int) throws {
            guard !visiting.contains(path) else {
                try fail(path, "recursive $ref")
                return
            }
            if visited.contains(path) { return }
            guard depth <= 64 else {
                throw JSONSchemaGrammarError.invalidSchema("Reference depth limit exceeded")
            }
            visiting.insert(path)
            for child in dependencies[path, default: []] { try checkAcyclic(child, depth + 1) }
            visiting.remove(path)
            visited.insert(path)
        }
        try checkAcyclic("#", 0)
    }
}
