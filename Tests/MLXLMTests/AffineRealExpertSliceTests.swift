// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Opt-in read-only sample of a staged Qwen35-layout checkpoint. This compares
/// only selected expert slices with separately read bytes; it does not load a
/// model, rehash a checkpoint, or establish generation or memory admission.
@Suite("Qwen35 real expert slice sampling", .serialized)
struct AffineRealExpertSliceTests {
    private struct Descriptor {
        let url: URL
        let absoluteStart: UInt64
        let length: UInt64
        let shape: [Int]
        let dtype: String
    }

    private func readExactly(_ file: FileHandle, count: Int) throws -> Data {
        guard count >= 0 else { throw AffineStreamingExpertError.invalidInput("negative read size") }
        var result = Data()
        while result.count < count {
            guard let chunk = try file.read(upToCount: count - result.count), !chunk.isEmpty else {
                throw AffineStreamingExpertError.invalidInput("truncated independent read")
            }
            result.append(chunk)
        }
        return result
    }

    private func descriptor(key: String, directory: URL, weightMap: [String: String]) throws -> Descriptor {
        let shardName = try #require(weightMap[key])
        guard !shardName.isEmpty, !shardName.contains("/"), shardName != ".", shardName != ".." else {
            throw AffineStreamingExpertError.invalidInput("invalid independent shard name")
        }
        let root = directory.resolvingSymlinksInPath().standardizedFileURL
        let shard = root.appendingPathComponent(shardName).resolvingSymlinksInPath().standardizedFileURL
        guard shard.deletingLastPathComponent() == root else {
            throw AffineStreamingExpertError.invalidInput("independent shard escapes model directory")
        }
        let file = try FileHandle(forReadingFrom: shard)
        defer { try? file.close() }
        let headerSize = try readExactly(file, count: 8).withUnsafeBytes {
            $0.loadUnaligned(as: UInt64.self).littleEndian
        }
        guard headerSize > 0, headerSize <= 16 * 1024 * 1024 else {
            throw AffineStreamingExpertError.invalidInput("independent header size out of range")
        }
        let header = try #require(JSONSerialization.jsonObject(with: readExactly(file, count: Int(headerSize)))
            as? [String: Any])
        let tensor = try #require(header[key] as? [String: Any])
        let dtype = try #require(tensor["dtype"] as? String)
        let shape = try #require(tensor["shape"] as? [Int])
        let offsets = try #require(tensor["data_offsets"] as? [UInt64])
        guard offsets.count == 2, offsets[0] <= offsets[1],
            offsets[0] <= UInt64.max - 8 - headerSize
        else { throw AffineStreamingExpertError.invalidInput("invalid independent tensor offsets") }
        return Descriptor(url: shard, absoluteStart: 8 + headerSize + offsets[0],
            length: offsets[1] - offsets[0], shape: shape, dtype: dtype)
    }

    @Test("sample real Qwen35 affine expert slices against independent file reads",
        .enabled(if: !(ProcessInfo.processInfo.environment["VMLX_QWEN35_REAL_SLICE_MODEL"] ?? "").isEmpty))
    func sampleRealExpertSlices() throws {
        guard let path = ProcessInfo.processInfo.environment["VMLX_QWEN35_REAL_SLICE_MODEL"],
            !path.isEmpty else { throw AffineStreamingExpertError.invalidInput("model directory required") }
        try MLXMetalTestLock.withLock {
            let directory = URL(fileURLWithPath: path, isDirectory: true)
            let config = try #require(JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appendingPathComponent("config.json"))) as? [String: Any])
            let text = try #require(config["text_config"] as? [String: Any])
            guard config["model_type"] as? String == "qwen3_5_moe",
                text["num_hidden_layers"] as? Int == 40,
                text["num_experts"] as? Int == 256,
                text["num_experts_per_tok"] as? Int == 8,
                text["hidden_size"] as? Int == 2048,
                text["moe_intermediate_size"] as? Int == 512
            else { throw AffineStreamingExpertError.invalidInput("unexpected checkpoint geometry") }
            let index = try #require(JSONSerialization.jsonObject(with: Data(contentsOf:
                directory.appendingPathComponent("model.safetensors.index.json"))) as? [String: Any])
            let weightMap = try #require(index["weight_map"] as? [String: String])
            let catalog = AffineStreamingExpertCatalog(numExperts: 256, inputDims: 2048,
                hiddenDims: 512, layout: .qwen35)
            try catalog.configure(modelDirectory: directory, layerCount: 40)
            var checkedSlices = 0
            var checkedBytes: UInt64 = 0
            for layer in [0, 39] {
                for expert in [0, 255] {
                    let mapped = try catalog.loadExpert(layer: layer, expert: expert)
                    for (projectionName, projection) in [("gate_proj", mapped.gate),
                        ("up_proj", mapped.up), ("down_proj", mapped.down)] {
                        for (suffix, array) in [("weight", projection.weight),
                            ("scales", projection.scales), ("biases", projection.biases)] {
                            let key = "language_model.model.layers.\(layer).mlp.switch_mlp.\(projectionName).\(suffix)"
                            let entry = try descriptor(key: key, directory: directory, weightMap: weightMap)
                            guard entry.shape.first == 256, entry.length % 256 == 0 else {
                                throw AffineStreamingExpertError.invalidInput("invalid independent expert geometry")
                            }
                            let sliceLength = entry.length / 256
                            let (delta, mulOverflow) = UInt64(expert).multipliedReportingOverflow(by: sliceLength)
                            let (offset, addOverflow) = entry.absoluteStart.addingReportingOverflow(delta)
                            guard !mulOverflow, !addOverflow, sliceLength > 0, sliceLength <= UInt64(Int.max) else {
                                throw AffineStreamingExpertError.invalidInput("independent expert slice overflow")
                            }
                            let file = try FileHandle(forReadingFrom: entry.url)
                            defer { try? file.close() }
                            try file.seek(toOffset: offset)
                            let expected = try readExactly(file, count: Int(sliceLength))
                            MLX.eval(array)
                            let actual = array.asData().data
                            #expect(array.shape == Array(entry.shape.dropFirst()))
                            #expect((suffix == "weight" && entry.dtype == "U32" && array.dtype == .uint32)
                                || (suffix != "weight" && entry.dtype == "BF16" && array.dtype == .bfloat16))
                            #expect(actual == expected, "slice mismatch: \(key) expert=\(expert)")
                            checkedSlices += 1
                            checkedBytes += sliceLength
                        }
                    }
                }
            }
            #expect(checkedSlices == 36)
            print("QWEN35_REAL_SLICE_SAMPLE directory=\(directory.standardizedFileURL.path) "
                + "layers=0,39 experts=0,255 slices=\(checkedSlices) payload_bytes=\(checkedBytes)")
        }
    }
}
