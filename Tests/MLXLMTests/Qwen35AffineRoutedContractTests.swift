// Copyright © 2026 Apple Inc.
import Foundation
import MLXLMCommon
import Testing
@testable import MLXVLM

struct Qwen35AffineRoutedContractTests {
    private func fresh() throws -> Qwen35 {
        let json = """
        {"model_type":"qwen3_5_moe","text_config":{"hidden_size":64,"num_hidden_layers":2,
         "num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"vocab_size":128,
         "linear_num_value_heads":2,"linear_num_key_heads":2,"linear_key_head_dim":16,
         "linear_value_head_dim":16,"linear_conv_kernel_dim":4,"full_attention_interval":2,
         "num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":128,
         "shared_expert_intermediate_size":64,"tie_word_embeddings":true}}
        """
        let config = try JSONDecoder().decode(Qwen35Configuration.self, from: Data(json.utf8))
        return try Qwen35(config, requesting: [.text])
    }
    private func paths() -> [String] {
        (0..<2).flatMap { layer in ["gate_proj", "up_proj", "down_proj"].map {
            "language_model.model.layers.\(layer).mlp.switch_mlp.\($0)"
        }}
    }
    @Test("pristine hybrid routed geometry validates complete pinned 3D affine descriptors")
    func validate() throws {
        let model = try fresh()
        let policy = Qwen35AffineRoutedContract.Policy(bits: 4, groupSize: 64)
        let policies = Dictionary(uniqueKeysWithValues: paths().map { ($0, policy) })
        let contract = try Qwen35AffineRoutedContract.capture(model: model, layerCount: 2,
            numExperts: 2, inputDims: 64, hiddenDims: 128, policies: policies)
        var tensors: [String: Qwen35AffineRoutedContract.Tensor] = [:]
        for path in paths() {
            let down = path.hasSuffix(".down_proj")
            tensors[path + ".weight"] = .init(shape: down ? [2, 64, 16] : [2, 128, 8], dtype: "U32")
            tensors[path + ".scales"] = .init(shape: down ? [2, 64, 2] : [2, 128, 1], dtype: "BF16")
            tensors[path + ".biases"] = .init(shape: down ? [2, 64, 2] : [2, 128, 1], dtype: "BF16")
        }
        try contract.validate(tensors)
        for suffix in [".weight", ".scales", ".biases"] {
            var missing = tensors; missing.removeValue(forKey: paths()[0] + suffix)
            #expect(throws: (any Error).self) { try contract.validate(missing) }
        }
        var wrong = tensors
        wrong[paths()[0] + ".weight"] = .init(shape: [2, 128, 4], dtype: "U32")
        #expect(throws: (any Error).self) { try contract.validate(wrong) }
        wrong = tensors
        wrong[paths()[0] + ".biases"] = .init(shape: [2, 128, 1], dtype: "F16")
        #expect(throws: (any Error).self) { try contract.validate(wrong) }
        wrong = tensors
        wrong["unexpected"] = .init(shape: [1], dtype: "BF16")
        #expect(throws: (any Error).self) { try contract.validate(wrong) }
        #expect(throws: (any Error).self) {
            try Qwen35AffineRoutedContract.capture(model: model, layerCount: 2,
                numExperts: 3, inputDims: 64, hiddenDims: 128, policies: policies)
        }
        var incomplete = policies; incomplete.removeValue(forKey: paths()[0])
        #expect(throws: (any Error).self) {
            try Qwen35AffineRoutedContract.capture(model: model, layerCount: 2,
                numExperts: 2, inputDims: 64, hiddenDims: 128, policies: incomplete)
        }
        var invalid = policies; invalid[paths()[0]] = .init(bits: 3, groupSize: 64)
        #expect(throws: (any Error).self) {
            try Qwen35AffineRoutedContract.capture(model: model, layerCount: 2,
                numExperts: 2, inputDims: 64, hiddenDims: 128, policies: invalid)
        }
    }
}
