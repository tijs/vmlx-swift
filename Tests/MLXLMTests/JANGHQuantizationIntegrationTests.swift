import Foundation
import XCTest

@testable import MLXLMCommon

final class JANGHQuantizationIntegrationTests: XCTestCase {
    private func partition(ordinaryMode: String, includeGroupSize: Bool = true) throws
        -> JANGHConfigurationPartition
    {
        var ordinary: [String: Any] = ["mode": ordinaryMode, "bits": 8]
        if includeGroupSize { ordinary["group_size"] = 32 }
        var plan: [String: Any] = [
            "mode": "affine", "bits": 8, "group_size": 64,
            "model.layers.0.self_attn.q_proj": ordinary,
        ]
        for role in ["gate_proj", "up_proj", "down_proj"] {
            plan["model.layers.0.mlp.switch_mlp." + role] = [
                "mode": "jangtq2", "bits": 2, "rotation": "hadamard32",
            ]
        }
        let config: [String: Any] = [
            "model_type": "glm5_next", "quantization": plan, "quantization_config": plan,
            "jangtq": [
                "version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
                "codebook_family": "odd-cubic", "rotation": "hadamard32",
                "codebooks": ["2": ["alpha": 1.0, "beta": 0.0,
                                      "levels": [-1.5, -0.5, 0.5, 1.5]]],
            ],
        ]
        return try JANGHConfigurationPartition(
            configuration: JSONSerialization.data(withJSONObject: config))
    }

    func testStrictDecoderPreservesCustomSkipsAcrossRuntimeWrappers() throws {
        let split = try partition(ordinaryMode: "mxfp8")
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: split.ordinaryConfiguration)
        let plan = try XCTUnwrap(base.perLayerQuantization)
        for role in ["gate_proj", "up_proj", "down_proj"] {
            for wrapper in ["", "model.", "language_model.", "language_model.model."] {
                XCTAssertNil(plan.quantization(layer: wrapper + "layers.0.mlp.switch_mlp." + role))
            }
        }
        XCTAssertEqual(plan.quantization(layer: "layers.0.self_attn.q_proj")?.mode, .mxfp8)
        XCTAssertEqual(plan.quantization(layer: "layers.0.self_attn.k_proj")?.mode, .affine)
    }

    func testStrictDecoderRejectsUnknownOrdinaryModeAfterCustomPartition() throws {
        for includeGroupSize in [false, true] {
            let split = try partition(ordinaryMode: "unrecognized-mode", includeGroupSize: includeGroupSize)
            XCTAssertThrowsError(try JSONDecoder().decode(
                BaseConfiguration.self, from: split.ordinaryConfiguration))
        }
    }
}
