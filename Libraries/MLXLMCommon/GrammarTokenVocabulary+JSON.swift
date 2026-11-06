import Foundation

extension GrammarTokenVocabulary {
    /// Read the declared plain ByteLevel decoder contract. Unsupported or
    /// ambiguous metadata returns nil; individual token decoding is never used
    /// to infer bytes (it can destroy incomplete UTF-8 sequences).
    public static func fromTokenizerJSON(
        _ data: Data, tokenizerConfig: Data
    ) -> GrammarTokenVocabulary? {
        struct Model: Decodable {
            let type: String
            let vocab: [String: Int]
            let continuing_subword_prefix: String?
            let end_of_word_suffix: String?
        }
        struct Decoder: Decodable { let type: String }
        struct Configuration: Decodable { let clean_up_tokenization_spaces: Bool }
        struct Added: Decodable {
            let id: Int
            let content: String
            let special: Bool
        }
        struct Document: Decodable {
            let model: Model
            let decoder: Decoder
            let added_tokens: [Added]?
        }
        guard let document = try? JSONDecoder().decode(Document.self, from: data),
            document.model.type == "BPE", document.decoder.type == "ByteLevel",
            document.model.continuing_subword_prefix?.isEmpty != false,
            document.model.end_of_word_suffix?.isEmpty != false,
            let config = try? JSONSerialization.jsonObject(with: tokenizerConfig) as? [String: Any],
            let settings = try? JSONDecoder().decode(Configuration.self, from: tokenizerConfig),
            !settings.clean_up_tokenization_spaces,
            !document.model.vocab.isEmpty
        else { return nil }

        // The byte-to-Unicode alphabet specified by the ByteLevel decoder.
        // Added tokens bypass that decoder, so encode their literal UTF-8 bytes
        // into this same alphabet before handing the vocabulary to XGrammar.
        let initial = Array(33 ... 126) + Array(161 ... 172) + Array(174 ... 255)
        var byteScalars: [Int: Unicode.Scalar] = [:]
        for value in initial { byteScalars[value] = Unicode.Scalar(value)! }
        var extra = 0
        for value in 0 ... 255 where byteScalars[value] == nil {
            byteScalars[value] = Unicode.Scalar(256 + extra)!
            extra += 1
        }
        let alphabet = Set(byteScalars.values)
        func literalPiece(_ text: String) -> String {
            String(String.UnicodeScalarView(text.utf8.map { byteScalars[Int($0)]! }))
        }
        var pieces: [Int: String] = [:]
        var originalPieces: [Int: String] = [:]
        for (piece, id) in document.model.vocab {
            guard id >= 0, originalPieces[id] == nil else { return nil }
            originalPieces[id] = piece
            pieces[id] = piece
        }
        var special = Set<Int>()
        var addedIDs = Set<Int>()
        for added in document.added_tokens ?? [] {
            guard added.id >= 0, addedIDs.insert(added.id).inserted,
                originalPieces[added.id] == nil || originalPieces[added.id] == added.content
            else { return nil }
            originalPieces[added.id] = added.content
            pieces[added.id] = literalPiece(added.content)
            if added.special { special.insert(added.id) }
        }
        // Config special-token declarations also count when an exporter omitted
        // the corresponding added_tokens special flag.
        func specialContent(_ value: Any?) -> String? {
            if let text = value as? String { return text }
            return (value as? [String: Any])?["content"] as? String
        }
        for key in [
            "bos_token", "eos_token", "unk_token", "pad_token", "sep_token", "cls_token",
            "mask_token",
        ] {
            if let content = specialContent(config[key]) {
                for (id, piece) in originalPieces where piece == content { special.insert(id) }
            }
        }
        for value in config["additional_special_tokens"] as? [Any] ?? [] {
            if let content = specialContent(value) {
                for (id, piece) in originalPieces where piece == content { special.insert(id) }
            }
        }
        for (id, piece) in pieces where !special.contains(id) {
            guard !piece.isEmpty, piece.unicodeScalars.allSatisfy({ alphabet.contains($0) }) else {
                return nil
            }
        }
        // Bound sparse, untrusted IDs before allocating an ID-indexed array.
        // This is a supported vocabulary limit, not a model/token remapping.
        guard let maximum = pieces.keys.max(), maximum < (1 << 22) else { return nil }
        var vocabulary = Array(repeating: "", count: maximum + 1)
        for id in vocabulary.indices {
            if let piece = pieces[id] { vocabulary[id] = piece } else { special.insert(id) }  // ID holes must never become admissible empty tokens.
        }
        return GrammarTokenVocabulary(
            vocabulary: vocabulary, vocabularyType: .byteLevel,
            specialTokenIDs: special)
    }
}
