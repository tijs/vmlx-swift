import Cmlx
import Foundation
import MLX
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

final class Qwen4ExpJANGHPreparationTests: XCTestCase {
    private let roles = ["gate_proj", "up_proj", "down_proj"]

    private func configuration() -> [String: Any] {
        [
            "model_type": "qwen4_exp",
            "text_config": [
                "model_type": "qwen4_exp_text", "dtype": "bfloat16",
                "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 64,
                "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 16,
                "linear_num_value_heads": 4, "linear_num_key_heads": 1,
                "linear_key_head_dim": 16, "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4, "vocab_size": 128,
                "num_experts": 16, "num_experts_per_tok": 10,
                "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
                "layer_types": ["linear_attention", "full_attention"],
                "hc_count": 4, "hc_lowrank": 8, "ple_layer_ids": [],
                "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 8,
                "indexer_budget": 32, "indexer_compress_ratio": 4,
            ],
            "vision_config": [
                "model_type": "qwen3_vl", "depth": 2, "hidden_size": 64,
                "intermediate_size": 128, "out_hidden_size": 64, "num_heads": 4,
                "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2,
                "num_position_embeddings": 64,
            ],
        ]
    }

    private func fixture(_ directory: URL, widths: [Int] = [2, 3, 4]) throws -> Data {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var root = configuration()
        var quant: [String: Any] = ["group_size": 64, "bits": 8,
            "language_model.embed_tokens": ["bits": 4, "group_size": 64],
            "visual.blocks.0.attn.qkv": ["bits": 6, "group_size": 64],
        ]
        var books: [String: Any] = [:], headers: [String: Any] = [:], map: [String: String] = [:]
        var payload = Data()
        for layer in 0 ..< 2 {
            for (role, bits) in zip(roles, widths) {
                let module = "model.layers.\(layer).mlp.switch_mlp.\(role)"
                books[String(bits)] = ["alpha": 0.125, "beta": 0,
                    "levels": (0 ..< (1 << bits)).map { (Double($0) - Double((1 << bits) - 1) / 2) * 0.125 }]
                quant[module] = ["mode": "jangtq2", "bits": bits, "rotation": "hadamard32"]
                let input = role == "down_proj" ? 32 : 64
                let output = role == "down_proj" ? 64 : 32
                let shapes = [[16, output, input * bits / 32], [16, output]]
                let bytes = [Data(repeating: 0, count: 16 * output * input * bits / 8),
                    Data((0 ..< 16 * output).flatMap { _ in [UInt8(0), UInt8(0x3c)] })]
                for i in 0 ..< 2 {
                    let name = module + (i == 0 ? ".tq2_packed" : ".tq2_scales")
                    headers[name] = ["dtype": i == 0 ? "U32" : "F16", "shape": shapes[i],
                        "data_offsets": [payload.count, payload.count + bytes[i].count]]
                    map[name] = "model.safetensors"
                    payload.append(bytes[i])
                }
            }
        }
        var header = try JSONSerialization.data(withJSONObject: headers, options: .sortedKeys)
        let padded = (header.count + 7) / 8 * 8
        header.append(Data(repeating: 0x20, count: padded - header.count))
        var length = UInt64(header.count).littleEndian
        var file = withUnsafeBytes(of: &length) { Data($0) }
        file.append(header); file.append(payload)
        try file.write(to: directory.appendingPathComponent("model.safetensors"))
        try JSONSerialization.data(withJSONObject: ["weight_map": map]).write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))
        root["quantization"] = quant; root["quantization_config"] = quant
        root["jangtq"] = ["version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
            "rotation": "hadamard32", "codebook_family": "odd-cubic", "codebooks": books]
        return try JSONSerialization.data(withJSONObject: root)
    }

    func testOrdinaryAffineAndMisleadingFolderNameKeepGenericLoading() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("Allosaurus-JANGH2-" + UUID().uuidString)
        var root = configuration()
        root["quantization"] = ["bits": 4, "group_size": 64]
        XCTAssertNil(try Qwen4ExpJANGHPreparation.loadIfDeclared(directory: directory,
            configurationData: JSONSerialization.data(withJSONObject: root)))
    }

    func testCustomAdmissionPreservesOrdinaryQuantAndMapsOnlyAfterConstruction() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            let data = try fixture(directory)
            XCTAssertThrowsError(try JSONDecoder().decode(BaseConfiguration.self, from: data))
            let before = mlx_safetensors_mmap_tracked_buffer_bytes()
            let prepared = try XCTUnwrap(Qwen4ExpJANGHPreparation.loadIfDeclared(directory: directory, configurationData: data))
            XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(), before)
            XCTAssertEqual(prepared.banks.excludedTensorNames.count, 12)
            let ordinary = try XCTUnwrap(JSONSerialization.jsonObject(with: prepared.banks.ordinaryConfiguration) as? [String: Any])
            let q = try XCTUnwrap(ordinary["quantization"] as? [String: Any])
            XCTAssertEqual((q["language_model.embed_tokens"] as? [String: Int])?["bits"], 4)
            XCTAssertEqual((q["visual.blocks.0.attn.qkv"] as? [String: Int])?["bits"], 6)
            for layer in 0 ..< 2 {
                for role in roles { XCTAssertEqual(q["model.layers.\(layer).mlp.switch_mlp.\(role)"] as? Bool, false) }
            }
            let model = try prepared.construct(configurationData: prepared.banks.ordinaryConfiguration, requesting: [.text])
            for name in prepared.banks.excludedTensorNames {
                XCTAssertTrue(model.excludeFromGenericSafetensorsLoad(key: name))
            }
            for name in ["lm_head.weight", "language_model.layers.0.mlp.shared_expert.gate_proj.weight",
                         "language_model.layers.0.mlp.gate.weight", "mtp.layers.0.mlp.switch_mlp.gate_proj.weight"] {
                XCTAssertFalse(model.excludeFromGenericSafetensorsLoad(key: name))
            }
            XCTAssertTrue(model.excludeFromGenericSafetensorsLoad(key: "language_model.layers.1.ple.ngram_embedding.shards.0.weight"))
        }
    }

    func testDeclaredBadGeometryWidthsAndCoverageFailClosed() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let data = try fixture(directory)
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        for routes in [-1, 0, 17] {
            var changed = root
            var text = try XCTUnwrap(changed["text_config"] as? [String: Any])
            text["num_experts_per_tok"] = routes; changed["text_config"] = text
            XCTAssertThrowsError(try Qwen4ExpJANGHPreparation.loadIfDeclared(directory: directory,
                configurationData: JSONSerialization.data(withJSONObject: changed)))
        }
        var missing = root
        var quant = try XCTUnwrap(root["quantization"] as? [String: Any])
        quant.removeValue(forKey: "model.layers.1.mlp.switch_mlp.down_proj")
        missing["quantization"] = quant; missing["quantization_config"] = quant
        XCTAssertThrowsError(try Qwen4ExpJANGHPreparation.loadIfDeclared(directory: directory,
            configurationData: JSONSerialization.data(withJSONObject: missing)))
        let unsupported = try fixture(directory, widths: [2, 5, 4])
        XCTAssertThrowsError(try Qwen4ExpJANGHPreparation.loadIfDeclared(directory: directory,
            configurationData: unsupported))
    }
}
