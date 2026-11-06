import Cmlx
import Foundation
import XCTest

@testable import MLXLMCommon

final class JANGHModelPreparationTests: XCTestCase {
    private let parent = "model.layers.0.mlp.switch_mlp"
    private let shard = "model.safetensors"
    private let index = "model.safetensors.index.json"

    private func fixture(_ directory: URL) throws -> Data {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var books: [String: Any] = [:]
        var quant: [String: Any] = [:]
        var headers: [String: Any] = [:]
        var map: [String: String] = [:]
        var payload = Data()
        for (role, bits) in [("gate_proj", 2), ("up_proj", 3), ("down_proj", 4)] {
            let module = parent + "." + role
            books[String(bits)] = ["alpha": 0.25, "beta": 0,
                                   "levels": (0..<(1 << bits)).map { (Double($0) - Double((1 << bits) - 1) / 2) * 0.25 }]
            quant[module] = ["mode": "jangtq2", "bits": bits, "rotation": "none"]
            for (suffix, dtype, shape, data) in [
                ("tq2_packed", "U32", [2, 32, bits], Data(repeating: 0, count: 2 * 32 * bits * 4)),
                ("tq2_scales", "F16", [2, 32], Data((0..<64).flatMap { _ in [UInt8(0), UInt8(0x3c)] })),
            ] {
                let name = module + "." + suffix
                headers[name] = ["dtype": dtype, "shape": shape, "data_offsets": [payload.count, payload.count + data.count]]
                map[name] = shard
                payload.append(data)
            }
        }
        var header = try JSONSerialization.data(withJSONObject: headers, options: .sortedKeys)
        XCTAssertLessThan(header.count, 4088)
        header.append(Data(repeating: 0x20, count: 4088 - header.count))
        var length = UInt64(header.count).littleEndian
        var file = withUnsafeBytes(of: &length) { Data($0) }
        file.append(header)
        file.append(payload)
        try file.write(to: directory.appendingPathComponent(shard), options: .atomic)
        try JSONSerialization.data(withJSONObject: ["weight_map": map]).write(
            to: directory.appendingPathComponent(index), options: .atomic)
        let config: [String: Any] = ["model_type": "glm5_next", "jangtq": ["version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
                                                       "codebook_family": "odd-cubic", "rotation": "none", "codebooks": books],
                                     "quantization": quant]
        return try JSONSerialization.data(withJSONObject: config)
    }


    private func layout(modelType: String = "glm5_next", hidden: Int = 32,
                        intermediate: Int = 32, experts: Int = 2,
                        layers: Set<Int> = [0]) -> JANGHRoutedModelLayout {
        .init(modelType: modelType, hiddenSize: hidden, intermediateSize: intermediate,
              expertCount: experts, sparseLayers: layers)
    }

    func testAdmissionExcludesOnlyCustomTensorsWithoutMappingPayload() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            let config = try fixture(directory)
            let before = mlx_safetensors_mmap_tracked_buffer_bytes()
            let prepared = try JANGHModelPreparation(directory: directory, configuration: config,
                                                     sidecar: nil, layout: layout())
            XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(), before)
            XCTAssertEqual(prepared.moduleByLayer, [0: parent])
            var expected = Set<String>()
            for role in ["gate_proj", "up_proj", "down_proj"] {
                let module = parent + "." + role
                expected.insert(module + ".tq2_packed")
                expected.insert(module + ".tq2_scales")
            }
            XCTAssertEqual(prepared.excludedTensorNames, expected)
            let ordinary = try XCTUnwrap(JSONSerialization.jsonObject(with: prepared.ordinaryConfiguration)
                                        as? [String: Any])
            let quantization = try XCTUnwrap(ordinary["quantization"] as? [String: Any])
            for role in ["gate_proj", "up_proj", "down_proj"] {
                XCTAssertEqual(quantization[parent + "." + role] as? Bool, false)
            }
        }
    }

    func testArchitectureGeometryAndCoverageMustMatchBeforeMapping() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            let config = try fixture(directory)
            let before = mlx_safetensors_mmap_tracked_buffer_bytes()
            for wrong in [layout(modelType: "naive_n05_flash"), layout(hidden: 64),
                          layout(intermediate: 64), layout(experts: 3), layout(layers: []),
                          layout(layers: [0, 1]), layout(layers: [-1]), layout(hidden: 0)] {
                XCTAssertThrowsError(try JANGHModelPreparation(directory: directory,
                    configuration: config, sidecar: nil, layout: wrong))
            }
            XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(), before)
        }
    }
    func testOrdinaryUnknownModeStillFailsBeforeMapping() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            var config = try XCTUnwrap(JSONSerialization.jsonObject(with: fixture(directory)) as? [String: Any])
            var quantization = try XCTUnwrap(config["quantization"] as? [String: Any])
            quantization["mode"] = "affine"
            quantization["bits"] = 4
            quantization["group_size"] = 64
            quantization["model.layers.0.self_attn.q_proj"] = ["mode": "unrecognized-custom", "bits": 4]
            config["quantization"] = quantization
            let before = mlx_safetensors_mmap_tracked_buffer_bytes()
            let prepared = try JANGHModelPreparation(directory: directory,
                configuration: JSONSerialization.data(withJSONObject: config), sidecar: nil, layout: layout())
            XCTAssertThrowsError(try JSONDecoder().decode(BaseConfiguration.self,
                                                        from: prepared.ordinaryConfiguration))
            XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(), before)
        }
    }

    func testMappedBanksReportLogicalSizeWithoutReflectedWeightArrays() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            let config = try fixture(directory)
            let before = mlx_safetensors_mmap_tracked_buffer_bytes()
            try autoreleasepool {
                let prepared = try JANGHModelPreparation(directory: directory, configuration: config,
                                                         sidecar: nil, layout: layout())
                let banks = try prepared.makeRoutedExperts(activationLimit: nil)
                let bank = try XCTUnwrap(banks[0])
                XCTAssertTrue(bank.parameters().flattenedValues().isEmpty)
                let counts = bank.modelWeightAccounting()
                XCTAssertEqual(counts.parameterArrayBytes, 0)
                XCTAssertEqual(counts.supplementalMappedBytes, 2688)
                XCTAssertEqual(bank.numParameters(), 6144)
                XCTAssertGreaterThan(mlx_safetensors_mmap_tracked_buffer_bytes(), before)
            }
            XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(), before)
        }
    }

    func testLegacyRuntimeAndOverlayRefusedBeforeMapping() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            let config = try fixture(directory)
            let before = mlx_safetensors_mmap_tracked_buffer_bytes()
            for name in ["jangtq_runtime.safetensors", "jangtq_stacked.safetensors"] {
                let legacy = directory.appendingPathComponent(name)
                try Data().write(to: legacy)
                XCTAssertThrowsError(try JANGHModelPreparation(directory: directory,
                    configuration: config, sidecar: nil, layout: layout())) { error in
                    XCTAssertTrue(String(describing: error).contains("legacy JANGTQ"))
                }
                XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(), before)
                try FileManager.default.removeItem(at: legacy)
            }
            XCTAssertNoThrow(try JANGHModelPreparation(directory: directory,
                configuration: config, sidecar: nil, layout: layout()))
        }
    }

    func testSelectedDiagnosticAdmissionCountsWeightsWithoutWholeBankMapping() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            let config = try fixture(directory)
            let before = mlx_safetensors_mmap_tracked_buffer_bytes()
            let prepared = try JANGHModelPreparation(directory: directory, configuration: config,
                                                     sidecar: nil, layout: layout())
            let layers = try prepared.makeSelectedRoutedExperts(activationLimit: nil)
            let layer = try XCTUnwrap(layers[0])
            XCTAssertTrue(layer is JANGHSelectedRoutedExpertLayer)
            XCTAssertTrue(layer.parameters().flattenedValues().isEmpty)
            XCTAssertEqual(layer.modelWeightAccounting().supplementalMappedBytes, 2688)
            XCTAssertEqual(layer.numParameters(), 6144)
            XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(), before)
        }
    }

    func testSelectedDiagnosticCacheCapRejectsMalformedAndUnboundedValues() throws {
        XCTAssertEqual(try JANGHModelPreparation.selectedDiagnosticCacheBytes(environment: [:]), 128 * 1024 * 1024)
        XCTAssertEqual(try JANGHModelPreparation.selectedDiagnosticCacheBytes(environment: ["VMLX_JANGH_SELECTED_CACHE_MIB": "0"]), 0)
        XCTAssertEqual(try JANGHModelPreparation.selectedDiagnosticCacheBytes(environment: ["VMLX_JANGH_SELECTED_CACHE_MIB": "8192"]), 8 * 1024 * 1024 * 1024)
        for value in ["-1", "16385", "abc", "99999999999999999999999999999999"] {
            XCTAssertThrowsError(try JANGHModelPreparation.selectedDiagnosticCacheBytes(environment: ["VMLX_JANGH_SELECTED_CACHE_MIB": value]))
        }
    }

}
