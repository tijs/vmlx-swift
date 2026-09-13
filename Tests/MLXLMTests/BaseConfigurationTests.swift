// Copyright © 2025 Apple Inc.

import Foundation
import MLXLMCommon
import XCTest

public class BaseConfigurationTests: XCTestCase {

    func testQuantization() throws {
        let json =
            """
            {
                "model_type": "Test",
                "quantization": {
                    "group_size": 128,
                    "bits": 4
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "x"), .init(groupSize: 128, bits: 4))
    }

    func testHeterogenousQuantization() throws {
        // from https://huggingface.co/mlx-community/Qwen3-1.7B-4bit-AWQ/blob/main/config.json#L20
        let json =
            """
            {
                "model_type": "Test",
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "model.embed_tokens": {
                        "group_size": 32,
                        "bits": 4
                    },
                    "model.layers.0.self_attn.q_norm": false,
                    "true_layer": true
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        // a random layer -- no specific configuration gets default
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "x"),
            .init(groupSize: 64, bits: 4))

        // layer with an override
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "model.embed_tokens"),
            .init(groupSize: 32, bits: 4))

        // layer with an override -- not quant
        XCTAssertNil(
            config.perLayerQuantization?.quantization(layer: "model.layers.0.self_attn.q_norm"))

        // layer with an override -- true, use the default
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "true_layer"),
            .init(groupSize: 64, bits: 4))
    }

    func testDSV4RoutedExpertBitPlanIsQuantizationMetadata() throws {
        let json =
            """
            {
                "model_type": "deepseek_v4",
                "weight_format": "mxtq",
                "quantization": {
                    "bits": 8,
                    "group_size": 32,
                    "mode": "affine",
                    "routed_expert_bits": 2,
                    "routed_expert_bit_plan": {
                        "default_bits": 2,
                        "codec": "mxtq",
                        "routed_layer_bits": {
                            "23": 4,
                            "25": 4,
                            "28": 4,
                            "34": 4,
                            "36": 4
                        }
                    },
                    "mxtq_bits": {
                        "routed_expert": 2,
                        "attention": 8,
                        "shared_expert": 8
                    }
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        let fallback = config.perLayerQuantization?.quantization(layer: "model.layers.23.mlp.experts")
        XCTAssertEqual(fallback?.groupSize, 32)
        XCTAssertEqual(fallback?.bits, 8)
        XCTAssertEqual(fallback?.mode, .affine)
        XCTAssertTrue(
            config.perLayerQuantization?.perLayerQuantization.isEmpty ?? false,
            "DSV4 routed_expert_bit_plan is consumed by DeepseekV4Configuration/JANGTQ, not by BaseConfiguration per-layer affine overrides")
    }

    func testMXFPQuantizationMetadataDoesNotDecodeAsLayerOverride() throws {
        let json =
            """
            {
                "model_type": "qwen3_5",
                "quantization": {
                    "bits": 8,
                    "group_size": 32,
                    "mode": "mxfp8",
                    "quantization_backend": "mx.quantize",
                    "vision": "fp16_passthrough",
                    "mtp": "preserved",
                    "mtp_policy": "native_mxfp8_linears_fp16_norms",
                    "passthrough_bit_widths_used": [16],
                    "norm_convention": "qwen3_5_language_mlx_plus_one",
                    "model.embed_tokens": {
                        "bits": 8,
                        "group_size": 32,
                        "mode": "mxfp8"
                    },
                    "model.layers.0.self_attn.q_norm": false
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        let fallback = config.perLayerQuantization?.quantization(layer: "model.layers.0.mlp.down_proj")
        XCTAssertEqual(fallback?.groupSize, 32)
        XCTAssertEqual(fallback?.bits, 8)
        XCTAssertEqual(fallback?.mode, .mxfp8)
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "model.embed_tokens")?.mode,
            .mxfp8)
        XCTAssertNil(
            config.perLayerQuantization?.quantization(layer: "model.layers.0.self_attn.q_norm"))
        XCTAssertEqual(
            config.perLayerQuantization?.perLayerQuantization.count,
            2,
            "Only real per-layer dictionary overrides and false skips should enter the override map.")
    }

    func testQuantizationConfigAliasDecodesMixedPerLayerBits() throws {
        // Shape mirrors mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit
        // (revision 70a3aa32) which stamps the quantization map under the
        // Hugging Face name "quantization_config" with `language_model.model.`
        // prefixed per-layer overrides and no per-layer mode.
        let json =
            """
            {
                "model_type": "qwen3_5_moe",
                "quantization_config": {
                    "group_size": 64,
                    "bits": 4,
                    "mode": "affine",
                    "language_model.model.embed_tokens": {
                        "bits": 8,
                        "group_size": 64
                    },
                    "language_model.model.layers.0.linear_attn.in_proj_qkv": {
                        "bits": 4,
                        "group_size": 64
                    },
                    "language_model.model.layers.0.mlp.switch_mlp.gate_proj": {
                        "bits": 8,
                        "group_size": 64
                    }
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        // unscoped layer gets the quantization_config default
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "x"),
            .init(groupSize: 64, bits: 4, mode: .affine))

        // mixed per-layer 4/8-bit affine overrides are exposed
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "language_model.model.embed_tokens"),
            .init(groupSize: 64, bits: 8))
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "model.layers.0.linear_attn.in_proj_qkv"),
            .init(groupSize: 64, bits: 4))
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "model.layers.0.mlp.switch_mlp.gate_proj"),
            .init(groupSize: 64, bits: 8))

        XCTAssertEqual(
            config.perLayerQuantization?.perLayerQuantization.count,
            3,
            "Only the per-layer dictionary overrides should enter the override map.")

        // re-encoding keeps the historical "quantization" spelling
        let encoder = JSONEncoder()
        let reencoded = try encoder.encode(config)
        let object = try JSONSerialization.jsonObject(with: reencoded) as? [String: Any]
        XCTAssertNotNil(object?["quantization"], "encode(to:) must preserve the legacy key")
        XCTAssertNil(object?["quantization_config"], "encode(to:) must not emit the alias key")
    }

    func testQuantizationConfigAliasEquivalenceWhenBothKeysPresent() throws {
        // Some checkpoints (the Qwen3.6 OptiQ candidate included) carry both
        // spellings with identical content. Decoding must accept that instead
        // of guessing.
        let json =
            """
            {
                "model_type": "qwen3_5_moe",
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "mode": "affine",
                    "model.layers.0.mlp.gate_proj": {
                        "bits": 8,
                        "group_size": 64
                    }
                },
                "quantization_config": {
                    "group_size": 64,
                    "bits": 4,
                    "mode": "affine",
                    "model.layers.0.mlp.gate_proj": {
                        "bits": 8,
                        "group_size": 64
                    }
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "x"),
            .init(groupSize: 64, bits: 4, mode: .affine))
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "model.layers.0.mlp.gate_proj"),
            .init(groupSize: 64, bits: 8))
    }

    func testQuantizationConfigAliasConflictRejects() throws {
        // Both keys present BUT different: decoding must fail loudly rather
        // than silently prefer one spelling.
        for json in [
            """
            {
                "model_type": "qwen3_5_moe",
                "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
                "quantization_config": {"group_size": 32, "bits": 8, "mode": "affine"}
            }
            """,
            """
            {
                "model_type": "qwen3_5_moe",
                "quantization": {
                    "group_size": 64, "bits": 4,
                    "model.layers.0.mlp.gate_proj": {"bits": 8, "group_size": 64}
                },
                "quantization_config": {
                    "group_size": 64, "bits": 4,
                    "model.layers.0.mlp.gate_proj": {"bits": 4, "group_size": 64}
                }
            }
            """,
        ] {
            XCTAssertThrowsError(
                try JSONDecoder().decode(
                    BaseConfiguration.self, from: json.data(using: .utf8)!)
            ) { error in
                guard case DecodingError.dataCorrupted = error else {
                    return XCTFail("Expected a dataCorrupted decoding error, got \(error)")
                }
            }
        }
    }
}
