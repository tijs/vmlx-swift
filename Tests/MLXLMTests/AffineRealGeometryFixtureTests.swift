// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

@testable import MLXLMCommon

/// Opt-in, synthetic-only diagnostic. The caller must provide an owned scratch
/// directory and supervise disk, pressure, swap and process memory externally.
/// The fixture writes roughly 432 MiB to disk and retains the native expert bank
/// for parity; it is not a test of a model checkpoint or full-model admission.
@Suite("Qwen35 synthetic realistic affine geometry", .serialized)
struct AffineRealGeometryFixtureTests {
    private let experts = 256
    private let inputDims = 2048
    private let hiddenDims = 512
    private let prefix = "language_model.model.layers.0.mlp.switch_mlp."

    private struct Tensor {
        let name: String
        let dtype: String
        let shape: [Int]
        let bytesPerExpert: Int
        let offset: Int
        let projection: Int
        let kind: Int // 0 = packed U32, 1 = BF16 scale, 2 = BF16 bias
    }

    /// Writes at most one expert slice in memory at a time. Packed bytes vary
    /// with expert and projection; scales and biases encode finite BF16 values.
    private func makeFixture(root: URL) throws -> (directory: URL, shard: URL) {
        let directory = root.appendingPathComponent("affine-real-geometry-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        let shard = directory.appendingPathComponent("model.safetensors")
        do {
            var tensors: [Tensor] = []
            var offset = 0
            for (projection, name) in ["gate_proj", "up_proj", "down_proj"].enumerated() {
                let input = projection == 2 ? hiddenDims : inputDims
                let output = projection == 2 ? inputDims : hiddenDims
                for (kind, suffix) in ["weight", "scales", "biases"].enumerated() {
                    let last = kind == 0 ? input / 8 : input / 64
                    let bytes = output * last * (kind == 0 ? 4 : 2)
                    tensors.append(Tensor(name: prefix + name + "." + suffix,
                        dtype: kind == 0 ? "U32" : "BF16", shape: [experts, output, last],
                        bytesPerExpert: bytes, offset: offset, projection: projection, kind: kind))
                    offset += bytes * experts
                }
            }
            var headerObject: [String: Any] = [:]
            var weightMap: [String: String] = [:]
            for tensor in tensors {
                headerObject[tensor.name] = ["dtype": tensor.dtype, "shape": tensor.shape,
                    "data_offsets": [tensor.offset, tensor.offset + experts * tensor.bytesPerExpert]]
                weightMap[tensor.name] = "model.safetensors"
            }
            var header = try JSONSerialization.data(withJSONObject: headerObject, options: [.sortedKeys])
            let page = Int(getpagesize())
            // Absolute tensor offsets are 8 bytes past a page boundary, while
            // remaining naturally aligned for U32 and BF16 access.
            let padding = (page + 8 - ((8 + header.count) % page)) % page
            header.append(Data(repeating: 0x20, count: padding))
            guard (8 + header.count) % page == 8 else {
                throw AffineStreamingExpertError.invalidInput("fixture header alignment failed")
            }
            guard FileManager.default.createFile(atPath: shard.path, contents: nil) else {
                throw AffineStreamingExpertError.invalidInput("could not create fixture shard")
            }
            let file = try FileHandle(forWritingTo: shard)
            defer { try? file.close() }
            var size = UInt64(header.count).littleEndian
            try file.write(contentsOf: withUnsafeBytes(of: &size) { Data($0) })
            try file.write(contentsOf: header)
            for tensor in tensors {
                for expert in 0 ..< experts {
                    let slice: Data
                    if tensor.kind == 0 {
                        // Each byte encodes two affine 4-bit values. A distinct
                        // byte for every expert makes swapped slices observable.
                        let byte = UInt8(truncatingIfNeeded: expert + tensor.projection * 53)
                        slice = Data(repeating: byte, count: tensor.bytesPerExpert)
                    } else {
                        // BF16 0.015625 and -0.125, respectively, little endian.
                        let word: UInt16 = tensor.kind == 1 ? 0x3c80 : 0xbe00
                        let values = [UInt16](repeating: word.littleEndian, count: tensor.bytesPerExpert / 2)
                        slice = values.withUnsafeBytes { Data($0) }
                    }
                    try file.write(contentsOf: slice)
                }
            }
            try JSONSerialization.data(withJSONObject: ["weight_map": weightMap], options: [.sortedKeys])
                .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
            print("AFFINE_FULL_GEOMETRY_FIXTURE bytes=\(offset) header_bytes=\(header.count) experts=\(experts)")
            return (directory, shard)
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    @Test("256 distinct mapped experts at realistic Qwen35 geometry match native",
        .enabled(if: !(ProcessInfo.processInfo.environment["VMLX_AFFINE_REAL_GEOMETRY_SCRATCH"] ?? "").isEmpty))
    func realisticFullExpertGeometry() throws {
        guard let scratch = ProcessInfo.processInfo.environment["VMLX_AFFINE_REAL_GEOMETRY_SCRATCH"],
            !scratch.isEmpty else {
            throw AffineStreamingExpertError.invalidInput("scratch directory is required")
        }
        try MLXMetalTestLock.withLock {
            let fixture = try makeFixture(root: URL(fileURLWithPath: scratch, isDirectory: true))
            defer { try? FileManager.default.removeItem(at: fixture.directory) }
            let tensors = try MLX.loadArrays(url: fixture.shard)
            func projection(_ name: String, input: Int, output: Int) throws -> QuantizedSwitchLinear {
                QuantizedSwitchLinear(inputDims: input, outputDims: output, numExperts: experts,
                    weight: try #require(tensors[prefix + name + ".weight"]),
                    scales: try #require(tensors[prefix + name + ".scales"]),
                    biases: try #require(tensors[prefix + name + ".biases"]),
                    groupSize: 64, bits: 4, mode: .affine)
            }
            let native = SwitchGLU(inputDims: inputDims, hiddenDims: hiddenDims, numExperts: experts,
                allowFusedGateUpCache: false, compileSeparatedDecode: false)
            try native.update(modules: ModuleChildren.unflattened([
                ("gate_proj", try projection("gate_proj", input: inputDims, output: hiddenDims)),
                ("up_proj", try projection("up_proj", input: inputDims, output: hiddenDims)),
                ("down_proj", try projection("down_proj", input: hiddenDims, output: inputDims)),
            ]), verify: .all)
            let catalog = AffineStreamingExpertCatalog(numExperts: experts, inputDims: inputDims,
                hiddenDims: hiddenDims, layout: .qwen35)
            try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
            let reducer = AffineStreamingSwitchGLU(inputDims: inputDims, hiddenDims: hiddenDims,
                numExperts: experts, layerIndex: 0, catalog: catalog)
            let limits = AffineStreamingSwitchGLU.ResearchLimits(maxTokens: 65_536,
                maxRoutesPerToken: 8, maxWindowTokens: 128, maxGroupSize: 32,
                maxMappedExpertBytes: 8 * 1024 * 1024,
                maxRetainedOutputBytes: 2 * 1024 * 1024 * 1024)
            let oldPolicy = SwitchLayerRuntime.propagatePromotedOutput
            SwitchLayerRuntime.propagatePromotedOutput = false
            defer { SwitchLayerRuntime.propagatePromotedOutput = oldPolicy }
            for tokens in [1, 8, 128] {
                let ids = (0 ..< tokens).flatMap { token in
                    (0 ..< 8).map { slot in Int32((token * 3 + slot * 17) % experts) }
                }
                let indices = MLXArray(ids, [1, tokens, 8]).asType(.uint32)
                let values = (0 ..< tokens * inputDims).map { index in
                    Float((index * 17) % 101 - 50) / Float(37)
                }
                let x = MLXArray(values, [1, tokens, inputDims]).asType(.bfloat16)
                let scoreRow: [Float] = [0.037, 0.067, 0.093, 0.111, 0.127, 0.151, 0.18, 0.234]
                let scores = MLXArray(Array(repeating: scoreRow, count: tokens).flatMap { $0 },
                    [1, tokens, 8])
                let routed = native(x, indices)
                let expected = (routed * scores.asType(routed.dtype)[.ellipsis, .newAxis]).sum(axis: -2)
                let actual = try reducer.researchQwen35GroupedReduced(x, indices: indices,
                    scores: scores, groupLimit: 32, tokenWindow: 8, limits: limits)
                MLX.eval(actual, expected)
                let maxDifference = MLX.max(MLX.abs(actual.asType(.float32)
                    - expected.asType(.float32))).item(Float.self)
                print("AFFINE_FULL_GEOMETRY_PARITY tokens=\(tokens) max_abs=\(maxDifference)")
                #expect(actual.shape == expected.shape && actual.dtype == expected.dtype)
                #expect(MLX.arrayEqual(actual, expected).item(Bool.self),
                    "tokens=\(tokens) max_abs=\(maxDifference)")
            }
        }
    }
}
