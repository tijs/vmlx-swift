import Foundation
import XCTest

@testable import MLXLMCommon

final class NaiveN05ArchitectureContractTests: XCTestCase {
    private func parse(_ values: [String: Any] = [:]) throws -> NaiveN05ArchitectureContract {
        var config = values
        config["model_type"] = config["model_type"] ?? "naive_n05_flash"
        return try JSONDecoder().decode(
            NaiveN05ArchitectureContract.self,
            from: JSONSerialization.data(withJSONObject: config))
    }

    func testVendorDefaultSchedulesAndAttentionGeometry() throws {
        let c = try parse()
        XCTAssertEqual(c.layerCount, 48)
        XCTAssertEqual(
            c.attentionKinds.enumerated().filter { $0.element == .sparse }.map(\.offset),
            [0, 5, 11, 17, 23, 29, 35, 41, 47])
        XCTAssertEqual(c.routedLayers.filter { $0 }.count, 47)
        XCTAssertEqual(c.fullAttention.kvHeads, 4)
        XCTAssertEqual(c.slidingAttention.kvHeads, 8)
        XCTAssertEqual(c.fullAttention.rotaryDimensions, 64)
        XCTAssertEqual(c.slidingAttention.rotaryDimensions, 64)
        XCTAssertEqual(c.fullAttention.ropeTheta, 10_000_000)
        XCTAssertEqual(c.slidingAttention.ropeTheta, 10_000)
        XCTAssertFalse(c.fullAttention.hasSink)
        XCTAssertTrue(c.slidingAttention.hasSink)
        XCTAssertEqual(c.valueScale, 0.707)
        XCTAssertEqual(c.indexerPrecision, .fp8E4M3)
    }

    func testNullableAndFalsyRoutingScaleFollowVendor() throws {
        let missing = try parse()
        XCTAssertEqual(missing.routingScale, 1)
        XCTAssertFalse(missing.routingScaleWasProvided)
        for raw in [NSNull() as Any, 0.0] {
            let c = try parse(["routed_scaling_factor": raw])
            XCTAssertEqual(c.routingScale, 1)
            XCTAssertTrue(c.routingScaleWasProvided)
        }
        let custom = try parse([
            "routed_scaling_factor": 2.5, "attention_value_scale": NSNull(),
            "indexer_activation_dtype": "bf16",
        ])
        XCTAssertEqual(custom.routingScale, 2.5)
        XCTAssertEqual(custom.requestedRoutingScale, 2.5)
        XCTAssertNil(custom.valueScale)
        XCTAssertEqual(custom.indexerPrecision, .bf16)
    }

    func testExplicitSchedulesAndVendorDerivedLayerTypes() throws {
        let c = try parse([
            "num_hidden_layers": 3, "hybrid_layer_pattern": [1, 0, 1],
            "moe_layer_freq": [1, 0, 0], "layer_types": ["ignored", "by", "vendor"],
        ])
        XCTAssertEqual(c.attentionKinds, [.sliding, .sparse, .sliding])
        XCTAssertEqual(c.routedLayers, [true, false, false])
        // Vendor config regenerates layer_types from hybrid_layer_pattern in validate().
        let empty = try parse([
            "num_hidden_layers": 3, "hybrid_layer_pattern": [], "moe_layer_freq": NSNull(),
        ])
        XCTAssertEqual(empty.attentionKinds, [.sparse, .sliding, .sliding])
        XCTAssertEqual(empty.routedLayers, [false, true, true])
    }

    func testUnsupportedArchitectureVariantsFailExplicitly() throws {
        let cases: [[String: Any]] = [
            ["attention_projection_layout": "fused"], ["index_n_kv_heads": 2],
            ["enable_dsa": false], ["index_top_k": 0], ["scoring_func": "softmax"],
            ["n_group": 2], ["topk_group": 2], ["hidden_act": "gelu"],
            ["n_shared_experts": 1], ["tie_word_embeddings": true],
            ["_attn_implementation": "sdpa"], ["num_attention_heads": 63],
            ["partial_rotary_factor": 0.01], ["index_head_dim": 32],
            ["hybrid_layer_pattern": [0]], ["moe_layer_freq": [1]],
            ["num_experts_per_tok": 300], ["num_hidden_layers": 0],
            ["num_hidden_layers": Int.max], ["routed_scaling_factor": -1.0],
        ]
        for value in cases {
            XCTAssertThrowsError(try parse(value)) { error in
                XCTAssertTrue(
                    error is NaiveN05ArchitectureContract.ContractError, "\(value): \(error)")
            }
        }
    }

    func testWrongTypesAndExplicitNullGeometryAreNotDefaulted() throws {
        for value: [String: Any] in [
            ["hidden_size": NSNull()], ["num_attention_heads": "64"],
            ["indexer_activation_dtype": "fp8_e5m2"], ["partial_rotary_factor": "NaN"],
        ] {
            XCTAssertThrowsError(try parse(value))
        }
        XCTAssertThrowsError(try parse(["model_type": "glm5_next"]))
    }
}
