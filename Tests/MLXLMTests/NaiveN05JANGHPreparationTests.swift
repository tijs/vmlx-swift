import Cmlx
import Foundation
import MLX
import MLXNN
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon

final class NaiveN05JANGHPreparationTests: XCTestCase {
    private static let parent = "model.layers.1.mlp.switch_mlp"
    private static let shard = "model.safetensors"
    private static let index = "model.safetensors.index.json"
    private static func withLock(_ body: () throws -> Void) rethrows {
        try MLXMetalTestLock.withLock {
            try NativeMTPActivation.$explicitRequestOverride.withValue(false, operation:body)
        }
    }
    private static func architecture() -> [String: Any] {
        ["model_type":"naive_n05_flash", "hidden_size":32, "intermediate_size":64,
         "num_hidden_layers":2, "vocab_size":64, "num_attention_heads":2,
         "num_key_value_heads":1, "head_dim":8, "v_head_dim":4,
         "swa_num_attention_heads":2, "swa_num_key_value_heads":1,
         "swa_head_dim":8, "swa_v_head_dim":4, "partial_rotary_factor":0.5,
         "n_routed_experts":2, "num_experts_per_tok":1, "moe_intermediate_size":32,
         "hybrid_layer_pattern":[0,1], "moe_layer_freq":[0,1],
         "index_n_heads":2, "index_head_dim":8, "index_top_k":3, "sliding_window":3]
    }
    private static func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        return directory
    }
    private static func fixture(_ directory: URL) throws -> Data {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var books: [String: Any] = [:]
        var quant: [String: Any] = ["mode": "affine", "bits": 4, "group_size": 32]
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
        var config: [String: Any] = ["model_type": "naive_n05_flash", "jangtq": ["version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
                                                       "codebook_family": "odd-cubic", "rotation": "none", "codebooks": books],
                                     "quantization": quant]
        config.merge(architecture()) { _, new in new }
        return try JSONSerialization.data(withJSONObject: config)
    }

    func testCompleteBundleAdmissionAndExactCustomConstruction() throws {
        try Self.withLock {
            let directory = try Self.temporaryDirectory()
            defer { try? FileManager.default.removeItem(at:directory) }
            let config = try Self.fixture(directory)
            let before = mlx_safetensors_mmap_tracked_buffer_bytes()
            let prepared = try XCTUnwrap(NaiveN05JANGHPreparation.loadIfDeclared(
                directory:directory,configurationData:config))
            XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(),before)
            XCTAssertEqual(prepared.banks.moduleByLayer,[1:Self.parent])
            let model = try prepared.construct(requesting:[.text])
            let names = Set(model.parameters().flattened().map(\.0))
            XCTAssertFalse(names.contains { $0.hasPrefix(Self.parent + ".") })
            XCTAssertTrue(names.contains("model.layers.1.mlp.gate.weight"))
            XCTAssertEqual(prepared.banks.excludedTensorNames.count,6)
            for key in prepared.banks.excludedTensorNames {
                XCTAssertTrue(model.excludeFromGenericSafetensorsLoad(key:key))
            }
            XCTAssertFalse(model.excludeFromGenericSafetensorsLoad(key:"model.layers.1.mlp.gate.weight"))
            XCTAssertTrue(model.requiresExactTensorMmapBuffers)
            XCTAssertEqual(model.maximumSupportedDecodeBatchSize,1)
        }
    }

    func testMalformedCustomAndOrdinaryModesRefuseBeforeMapping() throws {
        try Self.withLock {
            let directory = try Self.temporaryDirectory()
            defer { try? FileManager.default.removeItem(at:directory) }
            let root = try XCTUnwrap(JSONSerialization.jsonObject(with:Self.fixture(directory)) as? [String:Any])
            let before = mlx_safetensors_mmap_tracked_buffer_bytes()
            for module in [Self.parent + ".gate_proj", "lm_head"] {
                var broken = root
                var quant = try XCTUnwrap(root["quantization"] as? [String:Any])
                quant[module] = ["mode":"unsupported-mode", "bits":4]
                broken["quantization"] = quant
                XCTAssertThrowsError(try NaiveN05JANGHPreparation.loadIfDeclared(directory:directory,
                    configurationData:JSONSerialization.data(withJSONObject:broken)))
            }
            XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(),before)
        }
    }

    func testModalitiesAndNativeMTPRefuseBeforeMapping() throws {
        try Self.withLock {
            let directory = try Self.temporaryDirectory()
            defer { try? FileManager.default.removeItem(at:directory) }
            let config = try Self.fixture(directory)
            let before = mlx_safetensors_mmap_tracked_buffer_bytes()
            let prepared = try XCTUnwrap(NaiveN05JANGHPreparation.loadIfDeclared(directory:directory,
                configurationData:config))
            XCTAssertThrowsError(try prepared.construct(requesting:[.vision]))
            XCTAssertThrowsError(try prepared.construct(requesting:[]))
            try NativeMTPActivation.$explicitRequestOverride.withValue(true) {
                XCTAssertThrowsError(try NaiveN05JANGHPreparation.loadIfDeclared(directory:directory,
                    configurationData:config))
            }
            XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(),before)
        }
    }

    func testActualGenericLoadPreservesCustomBanksF32RouterAndOrdinaryAffineHead() throws {
        try Self.withLock {
            let directory = try Self.temporaryDirectory()
            defer { try? FileManager.default.removeItem(at:directory) }
            var config = try XCTUnwrap(JSONSerialization.jsonObject(with:Self.fixture(directory)) as? [String:Any])
            var quant = try XCTUnwrap(config["quantization"] as? [String:Any])
            quant["lm_head"] = ["mode":"affine", "group_size":32, "bits":4]
            config["quantization"] = quant
            let data = try JSONSerialization.data(withJSONObject:config)
            let prepared = try XCTUnwrap(NaiveN05JANGHPreparation.loadIfDeclared(directory:directory,
                configurationData:data))
            let original = try prepared.construct(requesting:[.text])
            // Native mixed checkpoint: ordinary BF16, router/correction F32.
            // Values below are deliberately not representable in BF16.
            try original.update(parameters:ModuleParameters.unflattened(
                original.parameters().flattened().map { ($0.0,$0.1.asType(.bfloat16)) }),verify:[.all])
            let routerName = "model.layers.1.mlp.gate.weight"
            let correctionName = "model.layers.1.mlp.gate.e_score_correction_bias"
            let nativeRouter = MLXArray.full([2,32],values:MLXArray(Float(0.1234567)),dtype:.float32)
            let nativeCorrection = MLXArray([Float(0.2345678),-0.1234567])
            try original.update(parameters:ModuleParameters.unflattened([
                routerName:nativeRouter,correctionName:nativeCorrection]),verify:[])
            quantize(model:original,groupSize:32,bits:4,mode:.affine,filter:{ path,_ in path == "lm_head" })
            let weights = Dictionary(uniqueKeysWithValues:original.parameters().flattened())
            XCTAssertNotNil(weights["lm_head.scales"])
            let ordinaryShard = "ordinary.safetensors"
            try MLX.save(arrays:weights,url:directory.appendingPathComponent(ordinaryShard))
            let indexURL = directory.appendingPathComponent(Self.index)
            var index = try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:indexURL)) as? [String:Any])
            var map = try XCTUnwrap(index["weight_map"] as? [String:String])
            for key in weights.keys { map[key] = ordinaryShard }
            index["weight_map"] = map
            try JSONSerialization.data(withJSONObject:index).write(to:indexURL)
            // Index mutation invalidates the previous lease intentionally.
            // Admit the complete final index before the second construction.
            let reloadedPreparation = try XCTUnwrap(NaiveN05JANGHPreparation.loadIfDeclared(
                directory:directory,configurationData:data))
            let loaded = try reloadedPreparation.construct(requesting:[.text])
            try loadWeights(modelDirectory:directory,model:loaded,
                quantization:prepared.baseConfiguration.quantizationContainer?.quantization,
                perLayerQuantization:prepared.baseConfiguration.perLayerQuantization)
            XCTAssertTrue(loaded.head is QuantizedLinear)
            XCTAssertTrue(loaded.preservesCheckpointParameterDTypes)
            let actual = Dictionary(uniqueKeysWithValues:loaded.parameters().flattened())
            XCTAssertEqual(Set(actual.keys),Set(weights.keys))
            for (name,expected) in weights {
                let value = try XCTUnwrap(actual[name])
                XCTAssertEqual(value.shape,expected.shape)
                XCTAssertEqual(value.dtype,expected.dtype,name)
                XCTAssertTrue((value .== expected).all().item(Bool.self),name)
            }
            let moe = try XCTUnwrap(loaded.model.layers[1].mlp as? NaiveN05FlashMoE)
            XCTAssertTrue(moe.experts is JANGHRoutedExpertLayer)
        }
    }

    func testRegistryAdmitsOrdinaryTextButRejectsCustomWithoutDirectory() async throws {
        try await MLXMetalTestLock.withLock {
            try await NativeMTPActivation.withExplicitRequest(false) {
                let data = try JSONSerialization.data(withJSONObject:Self.architecture())
                let ordinary = try await LLMTypeRegistry.shared.createModel(configuration:data,
                    modelType:"naive_n05_flash",requesting:[.text])
                XCTAssertTrue(ordinary is NaiveN05FlashModel)
                let directory = try Self.temporaryDirectory()
                defer { try? FileManager.default.removeItem(at:directory) }
                let custom = try Self.fixture(directory)
                do {
                    _ = try await LLMTypeRegistry.shared.createModel(configuration:custom,
                        modelType:"naive_n05_flash",requesting:[.text])
                    XCTFail("Directory-free custom construction must refuse")
                } catch { }
                do {
                    _ = try await LLMTypeRegistry.shared.createModel(configuration:data,
                        modelType:"naive_n05_flash",requesting:[.vision])
                    XCTFail("Text architecture must refuse image requests")
                } catch { }
            }
        }
    }
    func testUnrelatedJSON5ConfigurationBypassesCustomProbe() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at:directory) }
        let data = Data("{model_type: 'llama', // ordinary JSON5\n hidden_size: 32,}".utf8)
        XCTAssertNil(try NaiveN05JANGHPreparation.loadIfDeclared(directory:directory,configurationData:data))
    }

}
