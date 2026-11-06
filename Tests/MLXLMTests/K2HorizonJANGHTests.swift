import Cmlx
import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class K2HorizonJANGHTests: XCTestCase {
    private let module = "model.layers.0.mlp.down_proj"

    /// Two layers with an affine exception in layer 1, matching the mixed bundle.
    private func configuration() -> [String: Any] {
        let book: [String: Any] = [
            "alpha": 0.25, "beta": 0,
            "levels": (0 ..< 16).map { (Double($0) - 7.5) * 0.25 },
        ]
        var plan: [String: Any] = ["mode": "affine", "bits": 8, "group_size": 32]
        for layer in 0 ..< 2 {
            for role in ["gate_proj", "up_proj"] {
                plan["model.layers.\(layer).mlp.\(role)"] = [
                    "mode": "affine", "bits": 6, "group_size": 32,
                ]
            }
        }
        plan[module] = ["mode": "jangtq2", "bits": 4, "rotation": "hadamard32"]
        plan["model.layers.1.mlp.down_proj"] = ["mode": "affine", "bits": 8, "group_size": 32]
        return [
            "model_type": "k2_horizon", "mlp_layout": "dense_jangh_down", "hidden_size": 32,
            "intermediate_size": 64, "num_hidden_layers": 2, "vocab_size": 64,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "layernorm_num_groups": 4, "rms_norm_eps": 1e-6,
            "jangtq": [
                "version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
                "codebook_family": "odd-cubic", "rotation": "hadamard32", "codebooks": ["4": book],
            ],
            "quantization": plan, "quantization_config": plan,
        ]
    }
    private func data(_ value: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: value)
    }

    private func fixture(_ directory: URL) throws -> Data {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let scalarCount: Int = 32 * 64
        var codes = [UInt32]()
        codes.reserveCapacity(scalarCount)
        for index in 0 ..< scalarCount {
            let row: Int = index / 64
            let code: Int = (index * 7 + row) % 16
            codes.append(UInt32(code))
        }
        var words = [UInt32](repeating: 0, count: codes.count / 8)
        for (i, code) in codes.enumerated() { words[i / 8] |= code << ((i % 8) * 4) }
        let packed = words.withUnsafeBytes { Data($0) }
        let scales = [Float16](repeating: 0.5, count: 32).withUnsafeBytes { Data($0) }
        let headers: [String: Any] = [
            module + ".tq2_packed": [
                "dtype": "U32", "shape": [1, 32, 8], "data_offsets": [0, packed.count],
            ],
            module + ".tq2_scales": [
                "dtype": "F16", "shape": [1, 32],
                "data_offsets": [packed.count, packed.count + scales.count],
            ],
        ]
        var header = try data(headers)
        header.append(Data(repeating: 0x20, count: 4088 - header.count))
        var length = UInt64(header.count).littleEndian
        var file = withUnsafeBytes(of: &length) { Data($0) }
        file.append(header)
        file.append(packed)
        file.append(scales)
        try file.write(to: directory.appendingPathComponent("model.safetensors"))
        try data([
            "weight_map": [
                module + ".tq2_packed": "model.safetensors",
                module + ".tq2_scales": "model.safetensors",
            ]
        ])
        .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        return try data(configuration())
    }

    func testMixedPartitionPreservesAffineWidthsAndException() throws {
        let partition = try JANGHConfigurationPartition(configuration: data(configuration()))
        XCTAssertEqual(partition.customModules, [module])
        let root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: partition.ordinaryConfiguration) as? [String: Any])
        let plan = try XCTUnwrap(root["quantization"] as? [String: Any])
        XCTAssertEqual(plan[module] as? Bool, false)
        XCTAssertEqual((plan["model.layers.0.mlp.gate_proj"] as? [String: Any])?["bits"] as? Int, 6)
        XCTAssertEqual((plan["model.layers.1.mlp.down_proj"] as? [String: Any])?["bits"] as? Int, 8)
        let base = try JSONDecoder().decode(
            BaseConfiguration.self, from: partition.ordinaryConfiguration)
        XCTAssertNotNil(base.perLayerQuantization)
    }

    func testDenseAdmissionIsArchitectureAndRoleSpecific() throws {
        var wrongFamily = configuration()
        wrongFamily["model_type"] = "llama"
        XCTAssertThrowsError(try JANGHConfigurationPartition(configuration: data(wrongFamily)))
        var wrongLayout = configuration()
        wrongLayout["mlp_layout"] = "dense"
        XCTAssertThrowsError(try JANGHConfigurationPartition(configuration: data(wrongLayout)))
        var wrongRole = configuration()
        var plan = wrongRole["quantization"] as! [String: Any]
        plan["model.layers.0.mlp.gate_proj"] = plan.removeValue(forKey: module)
        wrongRole["quantization"] = plan
        wrongRole["quantization_config"] = plan
        XCTAssertThrowsError(try JANGHConfigurationPartition(configuration: data(wrongRole)))
    }

    func testHeaderAdmissionAndDenseExecutionAtDecodePrefillBoundary() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            let config = try fixture(directory)
            let before = mlx_safetensors_mmap_tracked_buffer_bytes()
            let prep = try JANGHDenseModelPreparation(
                directory: directory, configuration: config, sidecar: nil,
                hiddenSize: 32, intermediateSize: 64, layerCount: 2)
            XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(), before)
            XCTAssertEqual(
                prep.excludedTensorNames, [module + ".tq2_packed", module + ".tq2_scales"])
            let projections = try prep.makeProjections()
            XCTAssertNil(projections[1])
            let op = try XCTUnwrap(projections[0])
            XCTAssertTrue(op.parameters().flattenedValues().isEmpty)
            XCTAssertEqual(op.supplementalParameterCount, 32 * 64)
            // Independent scalar Sylvester transform and codebook matmul.
            for count in [1, 63, 64, 65] {
                let values = (0 ..< (count * 64)).map { Float(($0 * 5) % 17 - 8) / 32 }
                let result = op(MLXArray(values, [count, 64])).asArray(Float.self)
                for token in 0 ..< count {
                    var rotated = [Float](repeating: 0, count: 64)
                    for k in 0 ..< 64 {
                        for j in 0 ..< 32 {
                            let sign: Float =
                                ((k % 32) & j).nonzeroBitCount.isMultiple(of: 2) ? 1 : -1
                            rotated[k] +=
                                values[token * 64 + k / 32 * 32 + j] * sign / sqrt(Float(32))
                        }
                    }
                    for row in 0 ..< 32 {
                        var expected: Float = 0
                        for k in 0 ..< 64 {
                            let index = row * 64 + k
                            let code = (index * 7 + row) % 16
                            expected += rotated[k] * (Float(code) - 7.5) * 0.25 * 0.5
                        }
                        XCTAssertEqual(result[token * 32 + row], expected, accuracy: 2e-4)
                    }
                }
            }
        }
    }

    func testBF16DenseDispatchPreservesFloatDecodeAndRoundedPrefill() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            let config = try fixture(directory)
            let prep = try JANGHDenseModelPreparation(
                directory: directory, configuration: config, sidecar: nil,
                hiddenSize: 32, intermediateSize: 64, layerCount: 2)
            let op = try XCTUnwrap(prep.makeProjections()[0])
            var floatRotationDiscriminatingOutputs = 0
            for count in [1, 63, 64] {
                var values = [Float]()
                for index in 0 ..< (count * 64) { values.append(Float((index * 5) % 17 - 8) / 32) }
                let input = MLXArray(values, [count, 64]).asType(.bfloat16)
                let inputValues = input.asType(.float32).asArray(Float.self)
                var expected = [Float]()
                var roundedRotationExpected = [Float]()
                for token in 0 ..< count {
                    var rotated = [Float](repeating: 0, count: 64)
                    for k in 0 ..< 64 {
                        for j in 0 ..< 32 {
                            let sign: Float =
                                ((k % 32) & j).nonzeroBitCount.isMultiple(of: 2) ? 1 : -1
                            rotated[k] += inputValues[token * 64 + k / 32 * 32 + j] * sign
                        }
                        rotated[k] /= sqrt(Float(32))
                    }
                    if count >= 64 {
                        // Native prefill rounds its H32 tile to the activation dtype.
                        rotated = MLXArray(rotated).asType(.bfloat16).asType(.float32).asArray(
                            Float.self)
                    }
                    let roundedRotation = MLXArray(rotated).asType(.bfloat16).asType(.float32)
                        .asArray(Float.self)
                    for row in 0 ..< 32 {
                        var value: Float = 0
                        var roundedValue: Float = 0
                        for k in 0 ..< 64 {
                            let index = row * 64 + k
                            let code = (index * 7 + row) % 16
                            value += rotated[k] * (Float(code) - 7.5) * 0.25 * 0.5
                            roundedValue += roundedRotation[k] * (Float(code) - 7.5) * 0.25 * 0.5
                        }
                        expected.append(value)
                        roundedRotationExpected.append(roundedValue)
                    }
                }
                let target = MLXArray(expected, [count, 32]).asType(.bfloat16)
                let actual = op(input)
                XCTAssertEqual(actual.dtype, .bfloat16)
                let got = actual.asType(.float32).asArray(Float.self)
                let wanted = target.asType(.float32).asArray(Float.self)
                let mismatches = got.indices.filter { got[$0] != wanted[$0] }
                if count < 64 {
                    let rounded = MLXArray(roundedRotationExpected).asType(.bfloat16).asType(
                        .float32
                    ).asArray(Float.self)
                    // Require a material output difference, excluding cancellation around zero.
                    floatRotationDiscriminatingOutputs +=
                        got.indices.filter {
                            abs(got[$0]) > 1e-3 && got[$0] != rounded[$0]
                        }.count
                }
                for index in mismatches {
                    let token = index / 32
                    let row = index % 32
                    var exact = 0.0
                    for k in 0 ..< 64 {
                        var transformed = 0.0
                        for j in 0 ..< 32 {
                            let sign = ((k % 32) & j).nonzeroBitCount.isMultiple(of: 2) ? 1.0 : -1.0
                            transformed += Double(inputValues[token * 64 + k / 32 * 32 + j]) * sign
                        }
                        transformed /= sqrt(32.0)
                        let packedIndex = row * 64 + k
                        let code = (packedIndex * 7 + row) % 16
                        exact += transformed * (Double(code) - 7.5) * 0.25 * 0.5
                    }
                    // The fixture's exact cancellations are zero to Double precision (8.3e-17).
                    // Sequential Float accumulation leaves -1.49e-8 while SIMD returns zero.
                    // Permit only this near-zero reduction noise; all nonzero outputs remain exact BF16.
                    XCTAssertLessThanOrEqual(
                        abs(exact), 1e-12, "nonzero BF16 mismatch rows=\(count) index=\(index)")
                    XCTAssertLessThanOrEqual(abs(got[index]), 3e-8)
                    XCTAssertLessThanOrEqual(abs(wanted[index]), 3e-8)
                }
            }
            XCTAssertGreaterThan(
                floatRotationDiscriminatingOutputs, 0,
                "fixture must reject activation-dtype H32 rounding during decode")
        }
    }

    private func switchConfiguration(hidden: Int, intermediate: Int) -> [String: Any] {
        var root = configuration()
        root["mlp_layout"] = "switch1"
        root["hidden_size"] = hidden
        root["intermediate_size"] = intermediate
        root["head_dim"] = hidden / 4
        root["num_hidden_layers"] = 1
        var plan: [String: Any] = ["mode": "affine", "bits": 8, "group_size": 32]
        for role in ["gate_proj", "up_proj", "down_proj"] {
            plan["model.layers.0.mlp.switch_mlp." + role] = [
                "mode": "jangtq2", "bits": 4, "rotation": "hadamard32",
            ]
        }
        root["quantization"] = plan
        root["quantization_config"] = plan
        return root
    }

    func testSwitch1PrefillGeometryRejectsBeforeReadingTensorIndex() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        // Both are valid H32/decode dimensions, but one prefill projection's K is unsupported.
        for (hidden, intermediate) in [(32, 64), (64, 96)] {
            let root = switchConfiguration(hidden: hidden, intermediate: intermediate)
            XCTAssertThrowsError(
                try K2HorizonJANGHPreparation.loadIfDeclared(
                    directory: directory, configurationData: data(root))
            ) { error in
                guard let contract = error as? K2HorizonConfiguration.ContractError,
                    case .unsupported(let message) = contract
                else {
                    return XCTFail("expected typed prefill geometry rejection, got \(error)")
                }
                XCTAssertEqual(
                    message,
                    "K2 switch1 JANGH requires hidden_size and intermediate_size divisible by 64 for prefill"
                )
            }
        }
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: directory.path).isEmpty)
        // No custom format: the JANGH gate must not restrict ordinary switch1 dimensions.
        var ordinary = switchConfiguration(hidden: 32, intermediate: 64)
        ordinary.removeValue(forKey: "jangtq")
        let affine: [String: Any] = ["mode": "affine", "bits": 8, "group_size": 32]
        ordinary["quantization"] = affine
        ordinary["quantization_config"] = affine
        XCTAssertNil(
            try K2HorizonJANGHPreparation.loadIfDeclared(
                directory: directory, configurationData: data(ordinary)))
    }

    func testSwitch1PrefillGeometryAcceptsAlignedHeadersWithoutConstructingModel() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var headers: [String: Any] = [:]
        var index: [String: String] = [:]
        var payload = Data()
        for role in ["gate_proj", "up_proj", "down_proj"] {
            let input = role == "down_proj" ? 128 : 64
            let output = role == "down_proj" ? 64 : 128
            let path = "model.layers.0.mlp.switch_mlp." + role
            let packed = Data(repeating: 0, count: input * output / 2)
            let scales = [Float16](repeating: 0.5, count: output).withUnsafeBytes { Data($0) }
            for (suffix, bytes, dtype, shape) in [
                ("tq2_packed", packed, "U32", [1, output, input / 8]),
                ("tq2_scales", scales, "F16", [1, output]),
            ] {
                let start = payload.count
                payload.append(bytes)
                headers[path + "." + suffix] = [
                    "dtype": dtype, "shape": shape, "data_offsets": [start, payload.count],
                ]
                index[path + "." + suffix] = "model.safetensors"
            }
        }
        var header = try data(headers)
        let padded = ((header.count + 8 + 4095) / 4096) * 4096 - 8
        header.append(Data(repeating: 32, count: padded - header.count))
        var length = UInt64(header.count).littleEndian
        var file = withUnsafeBytes(of: &length) { Data($0) }
        file.append(header)
        file.append(payload)
        try file.write(to: directory.appendingPathComponent("model.safetensors"))
        try data(["weight_map": index]).write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))
        let preparation = try XCTUnwrap(
            K2HorizonJANGHPreparation.loadIfDeclared(
                directory: directory,
                configurationData: data(switchConfiguration(hidden: 64, intermediate: 128))))
        XCTAssertNotNil(preparation.routed)
        XCTAssertNil(preparation.dense)
        XCTAssertEqual(preparation.routed?.excludedTensorNames.count, 6)
        // Header admission only: do not call construct/makeRoutedExperts or execute any MLX primitive.
    }

    func testMissingAffineExceptionAndOutOfRangeLayerFailClosed() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        _ = try fixture(directory)
        for path in ["model.layers.1.mlp.down_proj", module] {
            var root = configuration()
            var plan = root["quantization"] as! [String: Any]
            let old = plan.removeValue(forKey: path)
            if path == module { plan["model.layers.2.mlp.down_proj"] = old }
            root["quantization"] = plan
            root["quantization_config"] = plan
            XCTAssertThrowsError(
                try JANGHDenseModelPreparation(
                    directory: directory, configuration: data(root),
                    sidecar: nil, hiddenSize: 32, intermediateSize: 64, layerCount: 2))
        }
    }
}
