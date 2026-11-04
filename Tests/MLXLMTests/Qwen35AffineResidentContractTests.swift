// Copyright © 2026 Apple Inc.
import Testing
@testable import MLXLMCommon

struct Qwen35AffineResidentContractTests {
    private let routed = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
    private let embedding = "language_model.model.embed_tokens.weight"
    private let linear = "language_model.model.layers.0.mlp.gate"

    private func contract(tied: Bool = true) throws -> Qwen35AffineResidentContract {
        var shapes = [routed: [2, 64, 64], embedding: [128, 64],
            linear + ".weight": [64, 64], linear + ".bias": [64]]
        if !tied { shapes["language_model.lm_head.weight"] = [128, 64] }
        return try Qwen35AffineResidentContract(expected: shapes,
            affine: [linear: .init(bits: 4, groupSize: 64)], routed: [routed], tiedHead: tied)
    }
    private func tensors() -> [String: Qwen35AffineResidentContract.Tensor] {
        [embedding: .init(shape: [128, 64], dtype: "BF16"),
         linear + ".weight": .init(shape: [64, 8], dtype: "U32"),
         linear + ".scales": .init(shape: [64, 1], dtype: "BF16"),
         linear + ".biases": .init(shape: [64, 1], dtype: "BF16"),
         linear + ".bias": .init(shape: [64], dtype: "BF16")]
    }
    @Test("resident affine coverage enforces shapes, companions and ordinary bias")
    func coverage() throws {
        let validator = try contract()
        try validator.validate(tensors())
        for key in [embedding, linear + ".scales", linear + ".biases", linear + ".bias"] {
            var bad = tensors(); bad.removeValue(forKey: key)
            #expect(throws: (any Error).self) { try validator.validate(bad) }
        }
        var bad = tensors(); bad[linear + ".weight"] = .init(shape: [64, 4], dtype: "U32")
        #expect(throws: (any Error).self) { try validator.validate(bad) }
        bad = tensors(); bad[linear + ".biases"] = .init(shape: [64, 1], dtype: "F32")
        #expect(throws: (any Error).self) { try validator.validate(bad) }
        bad = tensors(); bad[routed] = .init(shape: [2, 64, 64], dtype: "BF16")
        #expect(throws: (any Error).self) { try validator.validate(bad) }
    }
    @Test("untied output head is mandatory")
    func head() throws {
        let validator = try contract(tied: false)
        #expect(throws: (any Error).self) { try validator.validate(tensors()) }
        var all = tensors()
        all["language_model.lm_head.weight"] = .init(shape: [128, 64], dtype: "F16")
        try validator.validate(all)
    }
    @Test("duplicate captured parameter or module paths throw")
    func duplicatePaths() throws {
        #expect(throws: Qwen35AffineResidentContract.ContractError.self) {
            try Qwen35AffineResidentContract.uniqueEntries([("duplicate", [1]), ("duplicate", [2])])
        }
        let unique = try Qwen35AffineResidentContract.uniqueEntries([("one", 1), ("two", 2)])
        #expect(unique.count == 2)
    }
}
