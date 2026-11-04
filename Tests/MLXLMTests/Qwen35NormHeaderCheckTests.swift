// Copyright © 2026 Apple Inc.
import Foundation
import MLX
import Testing
@testable import MLXLMCommon

struct Qwen35NormHeaderCheckTests {
    private func fixture(_ values: [(String, [Float])]) throws -> (URL, Qwen35AffineResearchManifest.Inspection) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let shard = root.appendingPathComponent("model.safetensors")
        var header: [String: Any] = ["__metadata__": ["format": "mlx"]]
        var payload = Data()
        func add(_ key: String, _ dtype: String, _ shape: [Int], _ bytes: Data) {
            let start = payload.count
            payload.append(bytes)
            header[key] = ["dtype": dtype, "shape": shape, "data_offsets": [start, payload.count]] as [String: Any]
        }
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            add("language_model.model.layers.0.mlp.switch_mlp.\(projection).weight", "U32", [1, 64, 8], Data(repeating: 0, count: 2048))
        }
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            for suffix in ["scales", "biases"] {
                add("language_model.model.layers.0.mlp.switch_mlp.\(projection).\(suffix)", "BF16", [1, 64, 1], Data(repeating: 0, count: 128))
            }
        }
        for (index, (dtype, numbers)) in values.enumerated() {
            var bytes = Data()
            for value in numbers {
                if dtype == "F32" {
                    var bits = value.bitPattern.littleEndian
                    bytes.append(Data(bytes: &bits, count: 4))
                } else if dtype == "F16" {
                    var bits = Float16(value).bitPattern.littleEndian
                    bytes.append(Data(bytes: &bits, count: 2))
                } else if dtype == "BF16" {
                    var bits = UInt16(value.bitPattern >> 16).littleEndian
                    bytes.append(Data(bytes: &bits, count: 2))
                } else { bytes.append(UInt8(0)) }
            }
            let suffix = index == 0 ? "input_layernorm" : "post_attention_layernorm"
            add("language_model.model.layers.0.\(suffix).weight", dtype, [numbers.count], bytes)
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        json.append(Data(repeating: 32, count: (4096 - (8 + json.count) % 4096) % 4096))
        var size = UInt64(json.count).littleEndian
        var data = Data(bytes: &size, count: 8); data.append(json); data.append(payload)
        try data.write(to: shard)
        let keys = header.keys.filter { $0 != "__metadata__" }
        let map = Dictionary(uniqueKeysWithValues: keys.map { ($0, "model.safetensors") })
        try JSONSerialization.data(withJSONObject: ["weight_map": map])
            .write(to: root.appendingPathComponent("model.safetensors.index.json"))
        return (root, try Qwen35AffineResearchManifest.inspect(modelDirectory: root, selectedShards: [shard],
            layerCount: 1, numExperts: 1, inputDims: 64, hiddenDims: 64))
    }
    @Test("CPU vote handles BF16/F16/F32, ties and convention precedence")
    func votes() throws {
        for dtype in ["BF16", "F16", "F32"] {
            let (root, manifest) = try fixture([(dtype, [0, 0]), (dtype, [1, 1])])
            defer { try? FileManager.default.removeItem(at: root) }
            #expect(try Qwen35NormHeaderCheck.requiresShift(inspection: manifest, modelDirectory: root))
            #expect(try Qwen35NormHeaderCheck.requiresShift(inspection: manifest, modelDirectory: root,
                configConvention: "mlx_plus_one"))
            let (shiftedRoot, shifted) = try fixture([(dtype, [1, 1]), (dtype, [1, 1])])
            defer { try? FileManager.default.removeItem(at: shiftedRoot) }
            #expect(!(try Qwen35NormHeaderCheck.requiresShift(inspection: shifted, modelDirectory: shiftedRoot)))
        }
        let (root, manifest) = try fixture([("F32", [0.5, 0.5]), ("F32", [0.5, 0.5])])
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(!(try Qwen35NormHeaderCheck.requiresShift(inspection: manifest, modelDirectory: root)))
        #expect(try Qwen35NormHeaderCheck.requiresShift(inspection: manifest, modelDirectory: root, fallbackWhenNoProbe: true))
    }
    @Test("ambiguous, nonfinite, unsupported, truncated and cancelled checks fail closed")
    func failures() throws {
        for (dtype, numbers) in [("F32", [Float(0.5).nextUp]), ("F32", [Float.infinity]), ("U8", [Float(0)])] {
            let (root, manifest) = try fixture([(dtype, numbers)])
            defer { try? FileManager.default.removeItem(at: root) }
            #expect(throws: (any Error).self) { try Qwen35NormHeaderCheck.requiresShift(inspection: manifest, modelDirectory: root) }
        }
        let (root, manifest) = try fixture([("BF16", [1, 1])])
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(throws: CancellationError.self) {
            try Qwen35NormHeaderCheck.requiresShift(inspection: manifest, modelDirectory: root, shouldCancel: { true })
        }
        #expect(throws: (any Error).self) {
            try Qwen35NormHeaderCheck.requiresShift(inspection: manifest, modelDirectory: root, maxProbeBytes: 2)
        }
        let handle = try FileHandle(forWritingTo: root.appendingPathComponent("model.safetensors"))
        let size = try handle.seekToEnd()
        try handle.truncate(atOffset: size - 1)
        try handle.close()
        #expect(throws: (any Error).self) { try Qwen35NormHeaderCheck.requiresShift(inspection: manifest, modelDirectory: root) }
    }

    @Test("CPU votes agree with native MLX on representative finite values")
    func nativeParity() throws {
        try MLXMetalTestLock.withLock {
            let examples: [[Float]] = [
                [0, 0], [1, 1], [0.5, 0.5],
                [Float.leastNonzeroMagnitude, -Float.leastNonzeroMagnitude],
            ]
            for dtype in ["BF16", "F16", "F32"] {
                for first in examples {
                    for second in examples {
                        let (root, manifest) = try fixture([(dtype, first), (dtype, second)])
                        defer { try? FileManager.default.removeItem(at: root) }
                        let mlxType: DType = dtype == "BF16" ? .bfloat16 : (dtype == "F16" ? .float16 : .float32)
                        let weights = [
                            "language_model.model.layers.0.input_layernorm.weight": MLXArray(first).asType(mlxType),
                            "language_model.model.layers.0.post_attention_layernorm.weight": MLXArray(second).asType(mlxType),
                        ]
                        let reference = NormConventionResolver.shouldApplyPlusOneShift(
                            metadataConvention: nil, configConvention: nil, declaredConvention: nil,
                            weights: weights, probeSuffixes: [".input_layernorm.weight", ".post_attention_layernorm.weight"],
                            fallbackWhenNoProbe: { false })
                        #expect(try Qwen35NormHeaderCheck.requiresShift(inspection: manifest, modelDirectory: root) == reference)
                    }
                }
            }
        }
    }
    @Test("manifest inspection checks cancellation at a header boundary")
    func inspectCancellation() throws {
        let (root, _) = try fixture([("BF16", [1, 1])])
        defer { try? FileManager.default.removeItem(at: root) }
        var checks = 0
        #expect(throws: CancellationError.self) {
            try Qwen35AffineResearchManifest.inspect(modelDirectory: root,
                selectedShards: [root.appendingPathComponent("model.safetensors")], layerCount: 1,
                numExperts: 1, inputDims: 64, hiddenDims: 64, shouldCancel: { checks += 1; return checks >= 3 })
        }
    }
}
