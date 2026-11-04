// Copyright © 2026 Apple Inc.
import Foundation
import MLXLMCommon
import Testing
@testable import MLXVLM

struct Qwen35AffineResidentCaptureTests {
    private func model(tied: Bool) throws -> Qwen35 {
        let json = """
        {"model_type":"qwen3_5_moe","text_config":{"hidden_size":64,"num_hidden_layers":2,
         "num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"vocab_size":128,
         "linear_num_value_heads":2,"linear_num_key_heads":2,"linear_key_head_dim":16,
         "linear_value_head_dim":16,"linear_conv_kernel_dim":4,"full_attention_interval":2,
         "num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":64,
         "shared_expert_intermediate_size":64,"tie_word_embeddings":\(tied)}}
        """
        let configuration = try JSONDecoder().decode(Qwen35Configuration.self, from: Data(json.utf8))
        return try Qwen35(configuration, requesting: [.text])
    }
    private func routed() -> Set<String> {
        Set((0..<2).flatMap { layer in
            ["gate_proj", "up_proj", "down_proj"].map {
                "language_model.model.layers.\(layer).mlp.switch_mlp.\($0).weight"
            }
        })
    }
    @Test("tiny text-only hybrid Qwen35 captures complete resident tree and tied head")
    func nativeCapture() throws {
        for tied in [true, false] {
            let m = try model(tied: tied)
            let parameters = m.parameters().flattened()
            let names = Set(parameters.map(\.0))
            let excluded = routed()
            #expect(excluded.isSubset(of: names))
            #expect(names.contains("language_model.model.embed_tokens.weight"))
            #expect(names.contains("language_model.lm_head.weight") == !tied)
            #expect(names.contains(where: { $0.contains("layers.0.linear_attn") }))
            #expect(names.contains(where: { $0.contains("layers.1.self_attn") }))
            let contract = try Qwen35AffineResidentContract.capture(model: m, affine: [:],
                routed: excluded, tiedHead: tied, layerCount: 2)
            var descriptors: [String: Qwen35AffineResidentContract.Tensor] = [:]
            for (key, parameter) in parameters where !excluded.contains(key) {
                descriptors[key] = .init(shape: parameter.shape, dtype: "BF16")
            }
            try contract.validate(descriptors)
            descriptors.removeValue(forKey: "language_model.model.embed_tokens.weight")
            #expect(throws: (any Error).self) { try contract.validate(descriptors) }
            #expect(throws: (any Error).self) {
                try Qwen35AffineResidentContract.capture(model: m, affine: [:],
                    routed: Set(excluded.dropFirst()), tiedHead: tied, layerCount: 2)
            }
            #expect(throws: (any Error).self) {
                try Qwen35AffineResidentContract.capture(model: m, affine: [:],
                    routed: excluded, tiedHead: !tied, layerCount: 2)
            }
            #expect(throws: (any Error).self) {
                try Qwen35AffineResidentContract.capture(model: m, affine: [:],
                    routed: excluded, tiedHead: tied, layerCount: 1)
            }
            let firstLayer = Set(excluded.filter { $0.contains(".layers.0.") })
            #expect(throws: (any Error).self) {
                try Qwen35AffineResidentContract.capture(model: m, affine: [:],
                    routed: firstLayer, tiedHead: tied, layerCount: 1)
            }
            var extra = excluded
            extra.insert("language_model.model.layers.2.mlp.switch_mlp.gate_proj.weight")
            #expect(throws: (any Error).self) {
                try Qwen35AffineResidentContract.capture(model: m, affine: [:],
                    routed: extra, tiedHead: tied, layerCount: 2)
            }
        }
    }
    @Test("capture accepts actual native affine leaf and rejects non-linear target")
    func affineCapture() throws {
        let m = try model(tied: true)
        let path = "language_model.model.layers.0.mlp.gate"
        let policy = Qwen35AffineResidentContract.Affine(bits: 4, groupSize: 64)
        let contract = try Qwen35AffineResidentContract.capture(model: m, affine: [path: policy],
            routed: routed(), tiedHead: true, layerCount: 2)
        var descriptors: [String: Qwen35AffineResidentContract.Tensor] = [:]
        for (key, parameter) in m.parameters().flattened() where !routed().contains(key) {
            descriptors[key] = .init(shape: parameter.shape, dtype: "BF16")
        }
        descriptors[path + ".weight"] = .init(shape: [2, 8], dtype: "U32")
        descriptors[path + ".scales"] = .init(shape: [2, 1], dtype: "BF16")
        descriptors[path + ".biases"] = .init(shape: [2, 1], dtype: "BF16")
        try contract.validate(descriptors)
        #expect(throws: (any Error).self) {
            try Qwen35AffineResidentContract.capture(model: m,
                affine: ["language_model.model.norm": policy], routed: routed(), tiedHead: true, layerCount: 2)
        }
    }
    @Test("capture accepts an affine embedding from a fresh text-only model")
    func embeddingCapture() throws {
        let m = try model(tied: true)
        let path = "language_model.model.embed_tokens"
        let policy = Qwen35AffineResidentContract.Affine(bits: 4, groupSize: 64)
        let contract = try Qwen35AffineResidentContract.capture(model: m, affine: [path: policy],
            routed: routed(), tiedHead: true, layerCount: 2)
        var descriptors: [String: Qwen35AffineResidentContract.Tensor] = [:]
        for (key, parameter) in m.parameters().flattened() where !routed().contains(key) {
            descriptors[key] = .init(shape: parameter.shape, dtype: "BF16")
        }
        descriptors[path + ".weight"] = .init(shape: [128, 8], dtype: "U32")
        descriptors[path + ".scales"] = .init(shape: [128, 1], dtype: "BF16")
        descriptors[path + ".biases"] = .init(shape: [128, 1], dtype: "BF16")
        try contract.validate(descriptors)
        descriptors.removeValue(forKey: path + ".scales")
        #expect(throws: (any Error).self) { try contract.validate(descriptors) }
    }
}
