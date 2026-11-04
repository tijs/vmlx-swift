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

@Suite("native affine active-expert SSD routing", .serialized)
struct AffineStreamingSwitchGLUTests {
    private let researchLimits = AffineStreamingSwitchGLU.ResearchLimits(
        maxTokens: 65_536, maxRoutesPerToken: 256, maxWindowTokens: 65_536,
        maxGroupSize: 2_048, maxMappedExpertBytes: 64 * 1024 * 1024,
        maxRetainedOutputBytes: 2 * 1024 * 1024 * 1024)

    private final class ConcurrentResults: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var shapes: [[Int]] = []
        private(set) var failures: [String] = []

        func append(shape: [Int]) {
            lock.lock()
            shapes.append(shape)
            lock.unlock()
        }

        func append(failure: String) {
            lock.lock()
            failures.append(failure)
            lock.unlock()
        }
    }

    private struct Fixture {
        let directory: URL
        let tensors: [String: MLXArray]
    }

    private func makeFixture(
        numExperts: Int = 4, dimensions: Int = 32, hiddenDimensions: Int? = nil, seedShift: Int = 0,
        layout: AffineStreamingExpertCatalog.Layout = .qwen4,
        nonPageAligned: Bool = false, companionOverride: DType? = nil
    ) throws -> Fixture {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("affine-streaming-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        func dense(expert: Int, projection: Int) -> MLXArray {
            let hidden = hiddenDimensions ?? dimensions
            let input = projection == 2 ? hidden : dimensions
            let output = projection == 2 ? dimensions : hidden
            let count = input * output
            let values = (0 ..< count).map { index -> Float in
                let centered = Float((index * 13 + expert * 17 + projection * 23 + seedShift) % 101) - 50
                return centered / Float(200 + projection * 40)
            }
            return MLXArray(values, [output, input])
        }

        var tensors: [String: MLXArray] = [:]
        for (projectionIndex, projection) in ["gate_proj", "up_proj", "down_proj"].enumerated() {
            var weights: [MLXArray] = []
            var scales: [MLXArray] = []
            var biases: [MLXArray] = []
            for expert in 0 ..< numExperts {
                let quantized = MLX.quantized(
                    dense(expert: expert, projection: projectionIndex),
                    groupSize: layout == .qwen35 ? 64 : dimensions, bits: 4, mode: .affine)
                weights.append(quantized.wq)
                let companion: DType = companionOverride ?? (layout == .qwen35 ? .bfloat16 : .float16)
                scales.append(quantized.scales.asType(companion))
                biases.append(quantized.biases!.asType(companion))
            }
            let root = layout == .qwen35 ? "language_model.model.layers" : "language_model.layers"
            let base = "\(root).0.mlp.switch_mlp.\(projection)"
            tensors[base + ".weight"] = stacked(weights, axis: 0)
            tensors[base + ".scales"] = stacked(scales, axis: 0)
            tensors[base + ".biases"] = stacked(biases, axis: 0)
        }
        MLX.eval(Array(tensors.values))

        let shardName = "model.safetensors"
        let shardURL = directory.appendingPathComponent(shardName)
        try MLX.save(arrays: tensors, url: shardURL)
        // Production Qwen4 uses JangPressPrestacker's alignment overlay.
        // Reproduce that contract here by padding the safetensors JSON header
        // so the data segment begins on a page boundary without changing any
        // tensor-relative data_offsets.
        let saved = try Data(contentsOf: shardURL)
        let oldHeaderLength = saved.prefix(8).withUnsafeBytes {
            $0.loadUnaligned(as: UInt64.self).littleEndian
        }
        var header = saved.subdata(in: 8 ..< (8 + Int(oldHeaderLength)))
        let page = nonPageAligned ? Int(getpagesize()) : 4096
        let target = nonPageAligned ? 8 : 0
        let padding = (page + target - ((8 + header.count) % page)) % page
        header.append(Data(repeating: 0x20, count: padding))
        var aligned = Data()
        var newHeaderLength = UInt64(header.count).littleEndian
        aligned.append(contentsOf: withUnsafeBytes(of: &newHeaderLength) { Array($0) })
        aligned.append(header)
        aligned.append(saved.suffix(from: 8 + Int(oldHeaderLength)))
        try aligned.write(to: shardURL)
        if nonPageAligned {
            #expect((8 + header.count) % page == 8)
        }
        let weightMap = Dictionary(uniqueKeysWithValues: tensors.keys.map { ($0, shardName) })
        let index = ["weight_map": weightMap]
        try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        return Fixture(directory: directory, tensors: tensors)
    }

    @Test("loader exclusion is exact and does not swallow shared experts")
    func exactLoaderExclusion() {
        #expect(AffineStreamingExpertCatalog.isRoutedTensorKey(
            "language_model.layers.4.mlp.switch_mlp.gate_proj.weight"))
        #expect(AffineStreamingExpertCatalog.isRoutedTensorKey(
            "model.language_model.layers.4.mlp.switch_mlp.down_proj.biases"))
        #expect(!AffineStreamingExpertCatalog.isRoutedTensorKey(
            "language_model.layers.4.mlp.shared_expert.gate_proj.weight"))
        #expect(!AffineStreamingExpertCatalog.isRoutedTensorKey(
            "language_model.layers.4.mlp.switch_mlp.gate_proj.tq_packed"))
        #expect(AffineStreamingExpertCatalog.isRoutedTensorKey(
            "language_model.model.layers.4.mlp.switch_mlp.gate_proj.weight", layout: .qwen35))
        #expect(!AffineStreamingExpertCatalog.isRoutedTensorKey(
            "language_model.model.layers.4.mlp.shared_expert.gate_proj.weight", layout: .qwen35))
        #expect(!AffineStreamingExpertCatalog.isRoutedTensorKey(
            "language_model.model.layers.4.mlp.switch_mlp.gate_proj.weight"))
        #expect(!AffineStreamingExpertCatalog.isRoutedTensorKey(
            "language_model.layers.4.mlp.switch_mlp.gate_proj.weight", layout: .qwen35))
    }

    @Test("Qwen35 BF16 expert slices retain exact bytes with non-page-aligned offsets")
    func qwen35ExactRegions() throws {
        let fixture = try makeFixture(dimensions: 64, layout: .qwen35, nonPageAligned: true)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let catalog = AffineStreamingExpertCatalog(
            numExperts: 4, inputDims: 64, hiddenDims: 64, layout: .qwen35)
        try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
        for index in 0 ..< 4 {
            let expert = try catalog.loadExpert(layer: 0, expert: index)
            for (name, projection) in [("gate_proj", expert.gate), ("up_proj", expert.up),
                                       ("down_proj", expert.down)] {
                let base = "language_model.model.layers.0.mlp.switch_mlp.\(name)"
                #expect(projection.bits == 4 && projection.groupSize == 64)
                #expect(projection.scales.dtype == .bfloat16 && projection.biases.dtype == .bfloat16)
                for (suffix, mapped) in [("weight", projection.weight), ("scales", projection.scales),
                                         ("biases", projection.biases)] {
                    let expected = try #require(fixture.tensors[base + "." + suffix])[index]
                    MLX.eval(mapped, expected)
                    #expect(mapped.shape == expected.shape && mapped.dtype == expected.dtype)
                    #expect(mapped.asData().data == expected.asData().data)
                }
                let x = (MLXArray(0 ..< 64).asType(.float32) / 64).asType(.bfloat16).reshaped([1, 64])
                let actual = MLX.quantizedMM(x, projection.weight, scales: projection.scales,
                    biases: projection.biases, groupSize: 64, bits: 4, mode: .affine)
                let expected = MLX.quantizedMM(x, try #require(fixture.tensors[base + ".weight"])[index],
                    scales: try #require(fixture.tensors[base + ".scales"])[index],
                    biases: try #require(fixture.tensors[base + ".biases"])[index],
                    groupSize: 64, bits: 4, mode: .affine)
                MLX.eval(actual, expected)
                #expect(actual.dtype == expected.dtype && MLX.arrayEqual(actual, expected).item(Bool.self))
            }
        }
    }

    @Test("Qwen35 catalog rejects a mismatched companion dtype")
    func qwen35RejectsF16Companions() throws {
        let fixture = try makeFixture(dimensions: 64, layout: .qwen35, nonPageAligned: true,
                                      companionOverride: .float16)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let catalog = AffineStreamingExpertCatalog(
            numExperts: 4, inputDims: 64, hiddenDims: 64, layout: .qwen35)
        #expect(throws: AffineStreamingExpertError.self) {
            try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
        }
    }

    @Test("Qwen35 catalog rejects zero dimensions and layer counts")
    func qwen35RejectsZeroGeometry() throws {
        let directory = FileManager.default.temporaryDirectory
        for (experts, input, hidden, layers) in [
            (0, 64, 64, 1), (4, 0, 64, 1), (4, 64, 0, 1), (4, 64, 64, 0),
        ] {
            let catalog = AffineStreamingExpertCatalog(
                numExperts: experts, inputDims: input, hiddenDims: hidden, layout: .qwen35)
            do {
                try catalog.configure(modelDirectory: directory, layerCount: layers)
                Issue.record("zero geometry was accepted")
            } catch let AffineStreamingExpertError.invalidTensor(name, reason) {
                #expect(name == "model geometry")
                #expect(reason.contains("must be positive"))
            }
        }
    }

    @Test("Qwen35 catalog rejects parent and symlink shard escapes")
    func qwen35RejectsEscapedShards() throws {
        let fixture = try makeFixture(dimensions: 64, layout: .qwen35, nonPageAligned: true)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let parent = fixture.directory.deletingLastPathComponent()
        let external = parent.appendingPathComponent("affine-external-\(UUID().uuidString).safetensors")
        try FileManager.default.copyItem(
            at: fixture.directory.appendingPathComponent("model.safetensors"), to: external)
        defer { try? FileManager.default.removeItem(at: external) }
        let symlink = fixture.directory.appendingPathComponent("outside.safetensors")
        try FileManager.default.createSymbolicLink(at: symlink, withDestinationURL: external)
        let indexURL = fixture.directory.appendingPathComponent("model.safetensors.index.json")
        let index = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as? [String: Any])
        let originalMap = try #require(index["weight_map"] as? [String: String])
        let key = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"

        for (name, reason) in [("../\(external.lastPathComponent)", "invalid shard path"),
                               (symlink.lastPathComponent, "shard escapes model directory")] {
            var map = originalMap
            map[key] = name
            try JSONSerialization.data(withJSONObject: ["weight_map": map], options: [.sortedKeys])
                .write(to: indexURL)
            // Use a new catalog each time; an already configured catalog caches
            // its directory and would not re-read the mutated index.
            let catalog = AffineStreamingExpertCatalog(
                numExperts: 4, inputDims: 64, hiddenDims: 64, layout: .qwen35)
            do {
                try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
                Issue.record("escaped shard was accepted")
            } catch let AffineStreamingExpertError.invalidTensor(name, actual) {
                #expect(name == key)
                #expect(actual == reason)
            }
        }
    }

    @Test("Qwen35 BF16 routed output matches native top-8 across prefill chunks")
    func qwen35RoutedBF16Parity() throws {
        let fixture = try makeFixture(
            numExperts: 12, dimensions: 64, layout: .qwen35, nonPageAligned: true)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let catalog = AffineStreamingExpertCatalog(
            numExperts: 12, inputDims: 64, hiddenDims: 64, layout: .qwen35)
        try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)

        // Each row selects eight distinct experts, while experts recur across
        // tokens. Keep the unsorted 40-slot path used by native SwitchGLU.
        let ids: [Int32] = [
            0, 1, 2, 3, 4, 5, 6, 7,
            4, 5, 6, 7, 8, 9, 10, 11,
            8, 9, 10, 11, 0, 1, 2, 3,
            3, 5, 7, 9, 11, 1, 2, 4,
            11, 10, 9, 8, 7, 6, 5, 4,
        ]
        let indices = MLXArray(ids, [1, 5, 8]).asType(.uint32)
        let values: [Float] = (0 ..< 320).map { index in
            let centered: Int = (index * 17) % 101 - 50
            return Float(centered) / Float(37)
        }
        let x = MLXArray(values, [1, 5, 64]).asType(.bfloat16)
        let row: [Float] = [0.037, 0.067, 0.093, 0.111, 0.127, 0.151, 0.18, 0.234]
        let scores = MLXArray(Array(repeating: row, count: 5).flatMap { $0 }, [1, 5, 8])

        func tensor(_ projection: String, _ suffix: String) throws -> MLXArray {
            try #require(fixture.tensors[
                "language_model.model.layers.0.mlp.switch_mlp.\(projection).\(suffix)"])
        }
        let expanded = expandedDimensions(x, axes: [-2, -3])
        func nativeProject(_ projection: String, _ input: MLXArray) throws -> MLXArray {
            // QuantizedSwitchLinear casts each projection back to its BF16
            // input dtype before the activation and the next projection.
            let result = MLX.gatherQuantizedMM(
                input, try tensor(projection, "weight"),
                scales: try tensor(projection, "scales"),
                biases: try tensor(projection, "biases"), rhsIndices: indices,
                transpose: true, groupSize: 64, bits: 4, mode: .affine)
            return result.asType(input.dtype)
        }
        let up = try nativeProject("up_proj", expanded)
        let gate = try nativeProject("gate_proj", expanded)
        let activated = silu(gate) * up
        let nativeDown = try nativeProject("down_proj", activated)
        let routed = squeezed(nativeDown, axis: -2)
        // Qwen35SparseMoeBlock casts router scores to the routed dtype before
        // weighting and sums in that dtype; do not normalize in this test.
        for scoreType in [DType.bfloat16, DType.float32] {
            let routerScores = scores.asType(scoreType)
            let expected = (routed * routerScores.asType(routed.dtype)[.ellipsis, .newAxis])
                .sum(axis: -2)
            MLX.eval(expected)
            for chunkSize in [1, 2, 3, 5] {
                let streaming = AffineStreamingSwitchGLU(
                    inputDims: 64, hiddenDims: 64, numExperts: 12, layerIndex: 0,
                    catalog: catalog, cacheExpertLimit: 3, prefillChunkSize: chunkSize)
                let actual = streaming.reduced(x, indices: indices, scores: routerScores)
                MLX.eval(actual)
                let maxDifference = MLX.max(MLX.abs(
                    actual.asType(.float32) - expected.asType(.float32))).item(Float.self)
                #expect(actual.shape == expected.shape && actual.dtype == expected.dtype)
                #expect(MLX.arrayEqual(actual, expected).item(Bool.self),
                    "scores=\(scoreType) chunk=\(chunkSize) max absolute difference=\(maxDifference)")
                #expect(streaming.cachedExpertCount <= 3)
            }
        }
    }

    @Test("Qwen35 sorted routing matches real SwitchGLU for activation and promotion policies")
    func qwen35SortedNativeSwitchParity() throws {
        try MLXMetalTestLock.withLock {
            let fixture = try makeFixture(
                numExperts: 12, dimensions: 64, layout: .qwen35, nonPageAligned: true)
            defer { try? FileManager.default.removeItem(at: fixture.directory) }
            let catalog = AffineStreamingExpertCatalog(
                numExperts: 12, inputDims: 64, hiddenDims: 64, layout: .qwen35)
            try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
            let prefix = "language_model.model.layers.0.mlp.switch_mlp."

            func projection(_ name: String) throws -> QuantizedSwitchLinear {
                QuantizedSwitchLinear(
                    inputDims: 64, outputDims: 64, numExperts: 12,
                    weight: try #require(fixture.tensors[prefix + name + ".weight"]),
                    scales: try #require(fixture.tensors[prefix + name + ".scales"]),
                    biases: try #require(fixture.tensors[prefix + name + ".biases"]),
                    groupSize: 64, bits: 4, mode: .affine)
            }
            func native(_ compiled: Bool) throws -> SwitchGLU {
                let model = SwitchGLU(inputDims: 64, hiddenDims: 64, numExperts: 12,
                    allowFusedGateUpCache: false, compileSeparatedDecode: compiled)
                try model.update(modules: ModuleChildren.unflattened([
                    ("gate_proj", try projection("gate_proj")),
                    ("up_proj", try projection("up_proj")),
                    ("down_proj", try projection("down_proj")),
                ]), verify: [])
                return model
            }

            // 9 x 8 = 72 routes, above SwitchGLU's sort threshold. Each token
            // has unique IDs; the same experts recur across different tokens.
            let tokens = 9
            let ids = (0 ..< tokens).flatMap { token in
                (0 ..< 8).map { Int32(($0 * 5 + token * 3) % 12) }
            }
            let indices = MLXArray(ids, [1, tokens, 8]).asType(.uint32)
            let valueCount = tokens * 64
            let values: [Float] = (0 ..< valueCount).map { index in
                let centered: Int = (index * 17) % 101 - 50
                return Float(centered) / Float(37)
            }
            let row: [Float] = [0.037, 0.067, 0.093, 0.111, 0.127, 0.151, 0.18, 0.234]
            let scores = MLXArray(Array(repeating: row, count: tokens).flatMap { $0 },
                                  [1, tokens, 8])
            let oldPolicy = SwitchLayerRuntime.propagatePromotedOutput
            defer { SwitchLayerRuntime.propagatePromotedOutput = oldPolicy }

            for promote in [false, true] {
                SwitchLayerRuntime.propagatePromotedOutput = promote
                for activation in [DType.bfloat16, DType.float16] {
                    let x = MLXArray(values, [1, tokens, 64]).asType(activation)
                    for compiledSetting in [false, true] {
                        // The flag is exercised as configuration; 72 routes
                        // intentionally force the sorted fallback, not compiled
                        // single-token decode (which requires <64 routes).
                        let resident = try native(compiledSetting)
                        let y = resident(x, indices)
                        let expected = (y * scores.asType(y.dtype)[.ellipsis, .newAxis])
                            .sum(axis: -2)
                        MLX.eval(expected)
                        for chunk in [1, 4, 9] {
                            let streaming = AffineStreamingSwitchGLU(
                                inputDims: 64, hiddenDims: 64, numExperts: 12, layerIndex: 0,
                                catalog: catalog, cacheExpertLimit: 3, prefillChunkSize: chunk)
                            let actual = streaming.reduced(x, indices: indices, scores: scores)
                            MLX.eval(actual)
                            let maxDifference = MLX.max(MLX.abs(
                                actual.asType(.float32) - expected.asType(.float32))).item(Float.self)
                            let context = "promote=\(promote) activation=\(activation) "
                                + "compiled-setting=\(compiledSetting) chunk=\(chunk) "
                                + "actual-dtype=\(actual.dtype) expected-dtype=\(expected.dtype) "
                                + "max-abs=\(maxDifference)"
                            #expect(actual.shape == expected.shape, "\(context)")
                            #expect(actual.dtype == expected.dtype, "\(context)")
                            #expect(MLX.arrayEqual(actual, expected).item(Bool.self), "\(context)")
                        }
                    }
                }
            }
        }
    }

    @Test("Qwen35 sorted and unsorted affine projection stages expose first divergence")
    func qwen35SortedProjectionDiagnostics() throws {
        try MLXMetalTestLock.withLock {
            let fixture = try makeFixture(
                numExperts: 12, dimensions: 64, layout: .qwen35, nonPageAligned: true)
            defer { try? FileManager.default.removeItem(at: fixture.directory) }
            let prefix = "language_model.model.layers.0.mlp.switch_mlp."
            func tensor(_ projection: String, _ suffix: String) throws -> MLXArray {
                try #require(fixture.tensors[prefix + projection + "." + suffix])
            }
            func projection(_ name: String) throws -> QuantizedSwitchLinear {
                QuantizedSwitchLinear(
                    inputDims: 64, outputDims: 64, numExperts: 12,
                    weight: try tensor(name, "weight"), scales: try tensor(name, "scales"),
                    biases: try tensor(name, "biases"), groupSize: 64, bits: 4, mode: .affine)
            }
            let resident = SwitchGLU(inputDims: 64, hiddenDims: 64, numExperts: 12,
                allowFusedGateUpCache: false, compileSeparatedDecode: false)
            try resident.update(modules: ModuleChildren.unflattened([
                ("gate_proj", try projection("gate_proj")),
                ("up_proj", try projection("up_proj")),
                ("down_proj", try projection("down_proj")),
            ]), verify: .all)
            for (name, module) in [("gate_proj", resident.gateProj),
                                   ("up_proj", resident.upProj),
                                   ("down_proj", resident.downProj)] {
                let installed = try #require(module as? QuantizedSwitchLinear)
                let expectedWeight = try tensor(name, "weight")
                let expectedScales = try tensor(name, "scales")
                let expectedBiases = try tensor(name, "biases")
                let installedBiases = try #require(installed.biases)
                #expect(installed.groupSize == 64 && installed.bits == 4 && installed.mode == .affine)
                #expect(MLX.arrayEqual(installed.weight, expectedWeight).item(Bool.self))
                #expect(MLX.arrayEqual(installed.scales, expectedScales).item(Bool.self))
                #expect(MLX.arrayEqual(installedBiases, expectedBiases).item(Bool.self))
            }

            let tokens = 9
            let ids = (0 ..< tokens).flatMap { token in
                (0 ..< 8).map { Int32(($0 * 5 + token * 3) % 12) }
            }
            let indices = MLXArray(ids, [1, tokens, 8]).asType(.uint32)
            let valueCount = tokens * 64
            let values: [Float] = (0 ..< valueCount).map { index in
                let centered: Int = (index * 17) % 101 - 50
                return Float(centered) / Float(37)
            }
            let x = MLXArray(values, [1, tokens, 64]).asType(.bfloat16)
            let expanded = expandedDimensions(x, axes: [-2, -3])
            let (sortedX, sortedIndices, inverse) = gatherSort(x: expanded, indices: indices)
            let oldPolicy = SwitchLayerRuntime.propagatePromotedOutput
            SwitchLayerRuntime.propagatePromotedOutput = false
            defer { SwitchLayerRuntime.propagatePromotedOutput = oldPolicy }
            print("AFFINE_SORT_DIAGNOSTIC compiled_swiglu_enabled=\(HardwareInfo.isCompiledDecodeSupported)")
            func direct(_ name: String, _ x: MLXArray, _ ids: MLXArray,
                        sorted: Bool) throws -> MLXArray {
                MLX.gatherQuantizedMM(x, try tensor(name, "weight"),
                    scales: try tensor(name, "scales"), biases: try tensor(name, "biases"),
                    rhsIndices: ids, transpose: true, groupSize: 64, bits: 4,
                    mode: .affine, sortedIndices: sorted).asType(x.dtype)
            }
            func unsort(_ value: MLXArray) -> MLXArray {
                scatterUnsort(x: value, invOrder: inverse, shape: indices.shape)
            }
            func compare(_ label: String, _ left: MLXArray, _ right: MLXArray) {
                MLX.eval(left, right)
                let difference = MLX.max(MLX.abs(
                    left.asType(.float32) - right.asType(.float32))).item(Float.self)
                let exact = left.dtype == right.dtype && MLX.arrayEqual(left, right).item(Bool.self)
                print("AFFINE_SORT_DIAGNOSTIC stage=\(label) exact=\(exact) max_abs=\(difference)")
                #expect(left.shape == right.shape && left.dtype == right.dtype, "stage=\(label)")
                #expect(exact, "stage=\(label) max_abs=\(difference)")
            }

            let unsortedGate = try direct("gate_proj", expanded, indices, sorted: false)
            let unsortedUp = try direct("up_proj", expanded, indices, sorted: false)
            let sortedGate = try direct("gate_proj", sortedX, sortedIndices, sorted: true)
            let sortedUp = try direct("up_proj", sortedX, sortedIndices, sorted: true)
            compare("gate", unsortedGate, unsort(sortedGate))
            compare("up", unsortedUp, unsort(sortedUp))
            let unsortedActivated = silu(unsortedGate) * unsortedUp
            let sortedActivated = silu(sortedGate) * sortedUp
            compare("activation", unsortedActivated, unsort(sortedActivated))
            let unsortedDown = try direct("down_proj", unsortedActivated, indices, sorted: false)
            let sortedDown = try direct("down_proj", sortedActivated, sortedIndices, sorted: true)
            compare("down", unsortedDown, unsort(sortedDown))
            let nativeRouted = resident(x, indices)
            compare("native-vs-sorted", nativeRouted, squeezed(unsort(sortedDown), axis: -2))
            compare("native-vs-unsorted", nativeRouted, squeezed(unsortedDown, axis: -2))
        }
    }

    @Test("Qwen35 grouped padded projection matches native dispatch threshold")
    func qwen35GroupedProjectionThresholdDiagnostics() throws {
        try MLXMetalTestLock.withLock {
            let fixture = try makeFixture(numExperts: 256, dimensions: 64, layout: .qwen35)
            defer { try? FileManager.default.removeItem(at: fixture.directory) }
            let prefix = "language_model.model.layers.0.mlp.switch_mlp.gate_proj."
            let weights = try #require(fixture.tensors[prefix + "weight"])
            let scales = try #require(fixture.tensors[prefix + "scales"])
            let biases = try #require(fixture.tensors[prefix + "biases"])
            for tokens in [8, 127, 128, 129] {
                // Include frequently repeated experts, small groups and expert
                // boundaries that do not align to a 16-row Metal tile.
                let ids = (0 ..< tokens).flatMap { token in
                    (0 ..< 8).map { slot in Int32((token * 3 + slot * 17) % 37) }
                }
                let indices = MLXArray(ids, [1, tokens, 8]).asType(.uint32)
                let values = (0 ..< tokens * 64).map { index in
                    Float((index * 17) % 101 - 50) / Float(37)
                }
                for dtype in [DType.bfloat16, DType.float16] {
                    let x = MLXArray(values, [1, tokens, 64]).asType(dtype)
                    let expanded = expandedDimensions(x, axes: [-2, -3])
                    let (sortedX, sortedIndices, inverse) = gatherSort(x: expanded, indices: indices)
                    let sortedIDs = sortedIndices.asArray(Int32.self)
                    let native = MLX.gatherQuantizedMM(sortedX, weights, scales: scales,
                        biases: biases, rhsIndices: sortedIndices, transpose: true,
                        groupSize: 64, bits: 4, mode: .affine, sortedIndices: true).asType(dtype)
                    let useRHS = tokens * 8 >= 1024
                    var segments: [MLXArray] = []
                    var start = 0
                    while start < sortedIDs.count {
                        let expert = Int(sortedIDs[start])
                        var end = start + 1
                        while end < sortedIDs.count && sortedIDs[end] == sortedIDs[start] {
                            end += 1
                        }
                        var offset = start
                        while offset < end {
                            let stop = min(offset + 32, end)
                            let count = stop - offset
                            var group = sortedX[offset ..< stop, 0..., 0...]
                            let padded = useRHS ? max(count, 16) : count
                            if padded > count {
                                let copies = (0 ..< padded - count).map { _ in
                                    sortedX[offset ..< offset + 1, 0..., 0...]
                                }
                                group = concatenated([group] + copies, axis: 0)
                            }
                            let localIDs = MLXArray([Int32](repeating: 0, count: padded)).asType(.uint32)
                            let output = MLX.gatherQuantizedMM(group,
                                weights[expert ..< expert + 1, 0..., 0...],
                                scales: scales[expert ..< expert + 1, 0..., 0...],
                                biases: biases[expert ..< expert + 1, 0..., 0...],
                                rhsIndices: localIDs, transpose: true, groupSize: 64,
                                bits: 4, mode: .affine, sortedIndices: useRHS).asType(dtype)
                            segments.append(output[0 ..< count, 0..., 0...])
                            offset = stop
                        }
                        start = end
                    }
                    let grouped = concatenated(segments, axis: 0)
                    let nativeOrdered = scatterUnsort(x: native, invOrder: inverse, shape: indices.shape)
                    let groupedOrdered = scatterUnsort(x: grouped, invOrder: inverse, shape: indices.shape)
                    MLX.eval(nativeOrdered, groupedOrdered)
                    let maxDifference = MLX.max(MLX.abs(nativeOrdered.asType(.float32)
                        - groupedOrdered.asType(.float32))).item(Float.self)
                    let exact = MLX.arrayEqual(nativeOrdered, groupedOrdered).item(Bool.self)
                    print("AFFINE_GROUP_DIAGNOSTIC tokens=\(tokens) dtype=\(dtype) rhs=\(useRHS) exact=\(exact) max_abs=\(maxDifference)")
                    #expect(nativeOrdered.shape == groupedOrdered.shape)
                    #expect(nativeOrdered.dtype == groupedOrdered.dtype)
                    #expect(exact, "tokens=\(tokens) dtype=\(dtype) max_abs=\(maxDifference)")
                }
            }
        }
    }

    @Test("Qwen35 grouped padded full SwitchGLU matches native at dispatch threshold")
    func qwen35GroupedFullSwitchThresholdDiagnostics() throws {
        try MLXMetalTestLock.withLock {
            let fixture = try makeFixture(numExperts: 256, dimensions: 64, layout: .qwen35)
            defer { try? FileManager.default.removeItem(at: fixture.directory) }
            let prefix = "language_model.model.layers.0.mlp.switch_mlp."
            func tensor(_ name: String, _ suffix: String) throws -> MLXArray {
                try #require(fixture.tensors[prefix + name + "." + suffix])
            }
            func projection(_ name: String) throws -> QuantizedSwitchLinear {
                QuantizedSwitchLinear(inputDims: 64, outputDims: 64, numExperts: 256,
                    weight: try tensor(name, "weight"), scales: try tensor(name, "scales"),
                    biases: try tensor(name, "biases"), groupSize: 64, bits: 4, mode: .affine)
            }
            let native = SwitchGLU(inputDims: 64, hiddenDims: 64, numExperts: 256,
                allowFusedGateUpCache: false, compileSeparatedDecode: false)
            try native.update(modules: ModuleChildren.unflattened([
                ("gate_proj", try projection("gate_proj")),
                ("up_proj", try projection("up_proj")),
                ("down_proj", try projection("down_proj")),
            ]), verify: .all)
            let catalog = AffineStreamingExpertCatalog(
                numExperts: 256, inputDims: 64, hiddenDims: 64, layout: .qwen35)
            try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
            let streaming = AffineStreamingSwitchGLU(inputDims: 64, hiddenDims: 64,
                numExperts: 256, layerIndex: 0, catalog: catalog, cacheExpertLimit: 3)
            let oldPolicy = SwitchLayerRuntime.propagatePromotedOutput
            defer { SwitchLayerRuntime.propagatePromotedOutput = oldPolicy }
            for tokens in [7, 8, 127, 128, 129] {
                let ids = (0 ..< tokens).flatMap { token in
                    (0 ..< 8).map { slot in Int32((token * 3 + slot * 17) % 37) }
                }
                let indices = MLXArray(ids, [1, tokens, 8]).asType(.uint32)
                let values = (0 ..< tokens * 64).map { index in
                    Float((index * 17) % 101 - 50) / Float(37)
                }
                let scoreRow: [Float] = [0.037, 0.067, 0.093, 0.111, 0.127, 0.151, 0.18, 0.234]
                let scores = MLXArray(Array(repeating: scoreRow, count: tokens).flatMap { $0 },
                    [1, tokens, 8])
                for dtype in [DType.bfloat16, DType.float16] {
                    let x = MLXArray(values, [1, tokens, 64]).asType(dtype)
                    let expanded = expandedDimensions(x, axes: [-2, -3])
                    let (sortedX, sortedIndices, inverse) = gatherSort(x: expanded, indices: indices)
                    let sortedIDs = sortedIndices.asArray(Int32.self)
                    let useRHS = tokens * 8 >= 1024
                    for promote in [false, true] {
                        SwitchLayerRuntime.propagatePromotedOutput = promote
                        let nativeRouted = native(x, indices)
                        let nativeScores = scores.asType(nativeRouted.dtype)
                        let expected = (nativeRouted * nativeScores[.ellipsis, .newAxis])
                            .sum(axis: -2)
                        MLX.eval(expected)
                        for groupLimit in [16, 32, 64, 2_048] {
                            var segments: [MLXArray] = []
                            var start = 0
                            while start < sortedIDs.count {
                                let expert = Int(sortedIDs[start])
                                var end = start + 1
                                while end < sortedIDs.count && sortedIDs[end] == sortedIDs[start] {
                                    end += 1
                                }
                                var offset = start
                                while offset < end {
                                    let stop = offset + min(groupLimit, end - offset)
                                    let count = stop - offset
                                    var group = sortedX[offset ..< stop, 0..., 0...]
                                    let padded = useRHS ? max(count, 16) : count
                                    if padded > count {
                                        let copies = (0 ..< padded - count).map { _ in
                                            sortedX[offset ..< offset + 1, 0..., 0...]
                                        }
                                        group = concatenated([group] + copies, axis: 0)
                                    }
                                    let localIDs = MLXArray([Int32](repeating: 0, count: padded))
                                        .asType(.uint32)
                                    func run(_ name: String, _ input: MLXArray) throws -> MLXArray {
                                        let weight = try tensor(name, "weight")
                                        let scales = try tensor(name, "scales")
                                        let biases = try tensor(name, "biases")
                                        let result = MLX.gatherQuantizedMM(input,
                                            weight[expert ..< expert + 1, 0..., 0...],
                                            scales: scales[expert ..< expert + 1, 0..., 0...],
                                            biases: biases[expert ..< expert + 1, 0..., 0...],
                                            rhsIndices: localIDs, transpose: true, groupSize: 64,
                                            bits: 4, mode: .affine, sortedIndices: useRHS)
                                        if !promote && result.dtype != input.dtype
                                            && (input.dtype == .bfloat16 || input.dtype == .float16) {
                                            return result.asType(input.dtype)
                                        }
                                        return result
                                    }
                                    let gate = try run("gate_proj", group)
                                    let up = try run("up_proj", group)
                                    let activated = silu(gate) * up
                                    let down = try run("down_proj", activated)
                                    segments.append(down[0 ..< count, 0..., 0...])
                                    offset = stop
                                }
                                start = end
                            }
                            let grouped = concatenated(segments, axis: 0)
                            let ordered = scatterUnsort(x: grouped, invOrder: inverse, shape: indices.shape)
                            let routed = squeezed(ordered, axis: -2)
                            let actual = (routed * scores.asType(routed.dtype)[.ellipsis, .newAxis])
                                .sum(axis: -2)
                            MLX.eval(actual)
                            let maxDifference = MLX.max(MLX.abs(actual.asType(.float32)
                                - expected.asType(.float32))).item(Float.self)
                            let exact = actual.dtype == expected.dtype
                                && MLX.arrayEqual(actual, expected).item(Bool.self)
                            let context = "tokens=\(tokens) dtype=\(dtype) promote=\(promote) "
                                + "group=\(groupLimit) rhs=\(useRHS) max_abs=\(maxDifference)"
                            print("AFFINE_GROUP_FULL_DIAGNOSTIC \(context) exact=\(exact)")
                            #expect(actual.shape == expected.shape, "\(context)")
                            #expect(actual.dtype == expected.dtype, "\(context)")
                            #expect(exact, "\(context)")
                            let streamed = try streaming.researchQwen35GroupedReduced(
                                x, indices: indices, scores: scores, groupLimit: groupLimit, tokenWindow: 256, limits: researchLimits)
                            MLX.eval(streamed)
                            let streamedDifference = MLX.max(MLX.abs(streamed.asType(.float32)
                                - expected.asType(.float32))).item(Float.self)
                            print("AFFINE_GROUP_REDUCER_DIAGNOSTIC \(context) max_abs=\(streamedDifference)")
                            #expect(streamed.shape == expected.shape, "\(context)")
                            #expect(streamed.dtype == expected.dtype, "\(context)")
                            #expect(MLX.arrayEqual(streamed, expected).item(Bool.self),
                                "\(context) streamed_max_abs=\(streamedDifference)")
                            #expect(streaming.cachedExpertCount <= 3)
                            if groupLimit == 32 {
                                for window in [1, 7, 32] {
                                    var observedRoutes = 0
                                    var maximumPadded = 0
                                    let windowed = try streaming.researchQwen35GroupedReduced(
                                        x, indices: indices, scores: scores, groupLimit: groupLimit,
                                        tokenWindow: window, limits: researchLimits, observeWindow: { routes, padded in
                                            observedRoutes += routes
                                            maximumPadded = max(maximumPadded, padded)
                                            #expect(routes <= min(window, tokens) * 8)
                                            #expect(padded <= routes * 16)
                                        })
                                    MLX.eval(windowed)
                                    let difference = MLX.max(MLX.abs(windowed.asType(.float32)
                                        - expected.asType(.float32))).item(Float.self)
                                    #expect(observedRoutes == tokens * 8)
                                    #expect(maximumPadded <= min(window, tokens) * 8 * 16)
                                    #expect(windowed.shape == expected.shape && windowed.dtype == expected.dtype)
                                    #expect(MLX.arrayEqual(windowed, expected).item(Bool.self),
                                        "\(context) window=\(window) max_abs=\(difference)")
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Test("Qwen35 mapped reducer matches native at realistic projection geometry")
    func qwen35MappedRealisticGeometry() throws {
        try MLXMetalTestLock.withLock {
            let fixture = try makeFixture(numExperts: 8, dimensions: 2048,
                hiddenDimensions: 512, layout: .qwen35, nonPageAligned: true)
            defer { try? FileManager.default.removeItem(at: fixture.directory) }
            let prefix = "language_model.model.layers.0.mlp.switch_mlp."
            func projection(_ name: String, input: Int, output: Int) throws -> QuantizedSwitchLinear {
                QuantizedSwitchLinear(inputDims: input, outputDims: output, numExperts: 8,
                    weight: try #require(fixture.tensors[prefix + name + ".weight"]),
                    scales: try #require(fixture.tensors[prefix + name + ".scales"]),
                    biases: try #require(fixture.tensors[prefix + name + ".biases"]),
                    groupSize: 64, bits: 4, mode: .affine)
            }
            let native = SwitchGLU(inputDims: 2048, hiddenDims: 512, numExperts: 8,
                allowFusedGateUpCache: false, compileSeparatedDecode: false)
            try native.update(modules: ModuleChildren.unflattened([
                ("gate_proj", try projection("gate_proj", input: 2048, output: 512)),
                ("up_proj", try projection("up_proj", input: 2048, output: 512)),
                ("down_proj", try projection("down_proj", input: 512, output: 2048)),
            ]), verify: .all)
            let catalog = AffineStreamingExpertCatalog(numExperts: 8, inputDims: 2048,
                hiddenDims: 512, layout: .qwen35)
            try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
            let streaming = AffineStreamingSwitchGLU(inputDims: 2048, hiddenDims: 512,
                numExperts: 8, layerIndex: 0, catalog: catalog, cacheExpertLimit: 3)
            let oldPolicy = SwitchLayerRuntime.propagatePromotedOutput
            SwitchLayerRuntime.propagatePromotedOutput = false
            defer { SwitchLayerRuntime.propagatePromotedOutput = oldPolicy }
            for tokens in [1, 8, 16] {
                let ids = (0 ..< tokens).flatMap { token in
                    (0 ..< 8).map { slot in Int32((token * 3 + slot * 5) % 8) }
                }
                let indices = MLXArray(ids, [1, tokens, 8]).asType(.uint32)
                let values = (0 ..< tokens * 2048).map { index in
                    Float((index * 17) % 101 - 50) / Float(37)
                }
                let x = MLXArray(values, [1, tokens, 2048]).asType(.bfloat16)
                let row: [Float] = [0.037, 0.067, 0.093, 0.111, 0.127, 0.151, 0.18, 0.234]
                let scores = MLXArray(Array(repeating: row, count: tokens).flatMap { $0 }, [1, tokens, 8])
                let routed = native(x, indices)
                let expected = (routed * scores.asType(routed.dtype)[.ellipsis, .newAxis]).sum(axis: -2)
                let actual = try streaming.researchQwen35GroupedReduced(x, indices: indices,
                    scores: scores, groupLimit: 32, tokenWindow: 7, limits: researchLimits)
                MLX.eval(expected, actual)
                let difference = MLX.max(MLX.abs(expected.asType(.float32)
                    - actual.asType(.float32))).item(Float.self)
                print("AFFINE_MAPPED_REALISTIC tokens=\(tokens) max_abs=\(difference)")
                #expect(actual.shape == expected.shape && actual.dtype == expected.dtype)
                #expect(MLX.arrayEqual(actual, expected).item(Bool.self),
                    "tokens=\(tokens) max_abs=\(difference)")
            }
        }
    }

    @Test("affine mmap failures throw without crossing concurrent error scopes")
    func qwen35MappingFailures() throws {
        try MLXMetalTestLock.withLock {
            let first = try makeFixture(numExperts: 4, dimensions: 64, layout: .qwen35)
            let second = try makeFixture(numExperts: 4, dimensions: 64, seedShift: 13, layout: .qwen35)
            defer {
                try? FileManager.default.removeItem(at: first.directory)
                try? FileManager.default.removeItem(at: second.directory)
            }
            let catalogs = [first, second].map { _ in
                AffineStreamingExpertCatalog(numExperts: 4, inputDims: 64, hiddenDims: 64, layout: .qwen35)
            }
            for (catalog, fixture) in zip(catalogs, [first, second]) {
                try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
            }
            try FileManager.default.removeItem(at: first.directory.appendingPathComponent("model.safetensors"))
            let truncated = try FileHandle(forWritingTo: second.directory.appendingPathComponent("model.safetensors"))
            try truncated.truncate(atOffset: 0)
            try truncated.close()
            let results = ConcurrentResults()
            DispatchQueue.concurrentPerform(iterations: 2) { index in
                do {
                    _ = try catalogs[index].loadExpert(layer: 0, expert: 0)
                    results.append(failure: "missing shard was accepted")
                } catch AffineStreamingExpertError.mmapFailed(let reason) {
                    let path = [first, second][index].directory.path
                    if reason.contains(path) { results.append(shape: [index]) }
                    else { results.append(failure: "mapping error did not retain its own path: \(reason)") }
                } catch { results.append(failure: "unexpected error: \(error)") }
            }
            #expect(results.failures.isEmpty)
            #expect(Set(results.shapes.map { $0[0] }) == Set([0, 1]))
            let partial = try makeFixture(numExperts: 4, dimensions: 64, layout: .qwen35)
            defer { try? FileManager.default.removeItem(at: partial.directory) }
            let indexURL = partial.directory.appendingPathComponent("model.safetensors.index.json")
            let index = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: indexURL))
                as? [String: Any])
            var map = try #require(index["weight_map"] as? [String: String])
            for key in map.keys where key.contains(".up_proj.") { map[key] = "later.safetensors" }
            try FileManager.default.copyItem(at: partial.directory.appendingPathComponent("model.safetensors"),
                to: partial.directory.appendingPathComponent("later.safetensors"))
            try JSONSerialization.data(withJSONObject: ["weight_map": map], options: [.sortedKeys])
                .write(to: indexURL)
            let partialCatalog = AffineStreamingExpertCatalog(numExperts: 4, inputDims: 64,
                hiddenDims: 64, layout: .qwen35)
            try partialCatalog.configure(modelDirectory: partial.directory, layerCount: 1)
            try FileManager.default.removeItem(at: partial.directory.appendingPathComponent("later.safetensors"))
            let beforeFailure = MLX.Memory.activeMemory
            #expect(throws: AffineStreamingExpertError.self) {
                try partialCatalog.loadExpert(layer: 0, expert: 0)
            }
            #expect(MLX.Memory.activeMemory == beforeFailure)
            let recovery = try makeFixture(numExperts: 4, dimensions: 64, layout: .qwen35)
            defer { try? FileManager.default.removeItem(at: recovery.directory) }
            let valid = AffineStreamingExpertCatalog(numExperts: 4, inputDims: 64,
                hiddenDims: 64, layout: .qwen35)
            try valid.configure(modelDirectory: recovery.directory, layerCount: 1)
            let mapped = try valid.loadExpert(layer: 0, expert: 0)
            #expect(mapped.gate.weight.shape == [64, 8])
        }
    }

    @Test("Qwen35 research limits validate 65K geometry and mapped bytes")
    func qwen35ResearchLimits() throws {
        try MLXMetalTestLock.withLock {
            try researchLimits.validate(tokens: 65_536, inputDims: 2048,
                routesPerToken: 8, window: 32, group: 32)
            #expect(throws: AffineStreamingExpertError.self) {
                try researchLimits.validate(tokens: 65_537, inputDims: 2048,
                    routesPerToken: 8, window: 32, group: 32)
            }
            #expect(throws: AffineStreamingExpertError.self) {
                try researchLimits.validate(tokens: 1, inputDims: .max,
                    routesPerToken: 1, window: 1, group: 1)
            }
            let unbounded = AffineStreamingSwitchGLU.ResearchLimits(maxTokens: .max,
                maxRoutesPerToken: .max, maxWindowTokens: .max, maxGroupSize: .max,
                maxMappedExpertBytes: .max, maxRetainedOutputBytes: .max)
            #expect(throws: AffineStreamingExpertError.self) {
                try unbounded.validate(tokens: 1, inputDims: 64, routesPerToken: 1,
                    window: 1, group: 1)
            }
            let hugeRoutes = AffineStreamingSwitchGLU.ResearchLimits(maxTokens: 2,
                maxRoutesPerToken: .max, maxWindowTokens: 2, maxGroupSize: 1,
                maxMappedExpertBytes: 1, maxRetainedOutputBytes: 1024)
            #expect(throws: AffineStreamingExpertError.self) {
                try hugeRoutes.validate(tokens: 2, inputDims: 1, routesPerToken: .max,
                    window: 2, group: 1)
            }
            #expect(throws: AffineStreamingExpertError.self) {
                try hugeRoutes.validate(tokens: 1, inputDims: 1, routesPerToken: .max,
                    window: 1, group: 1)
            }
            let fixture = try makeFixture(numExperts: 4, dimensions: 64,
                layout: .qwen35, nonPageAligned: true)
            defer { try? FileManager.default.removeItem(at: fixture.directory) }
            let catalog = AffineStreamingExpertCatalog(numExperts: 4, inputDims: 64,
                hiddenDims: 64, layout: .qwen35)
            try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
            for expert in [0, 3] {
                let bytes = try catalog.mappedExpertBytes(layer: 0, expert: expert)
                #expect(bytes > 0 && bytes % UInt64(getpagesize()) == 0)
                let beforeReject = MLX.Memory.activeMemory
                do {
                    _ = try catalog.loadExpert(layer: 0, expert: expert, maxMappedBytes: bytes - 1)
                    Issue.record("expert over the mapping limit was accepted")
                } catch AffineStreamingExpertError.invalidInput(let reason) {
                    #expect(reason == "expert mapping exceeds research byte limit")
                } catch { Issue.record("unexpected rejection error: \(error)") }
                #expect(MLX.Memory.activeMemory == beforeReject)
                let before = MLX.Memory.activeMemory
                let loaded = try catalog.loadExpert(layer: 0, expert: expert, maxMappedBytes: bytes)
                let after = MLX.Memory.activeMemory
                #expect(after - before == Int(bytes))
                #expect(loaded.gate.weight.shape == [64, 8])
            }
            let tight = AffineStreamingSwitchGLU.ResearchLimits(maxTokens: 8, maxRoutesPerToken: 1,
                maxWindowTokens: 2, maxGroupSize: 4, maxMappedExpertBytes: 1,
                maxRetainedOutputBytes: 4096)
            let reducer = AffineStreamingSwitchGLU(inputDims: 64, hiddenDims: 64,
                numExperts: 4, layerIndex: 0, catalog: catalog)
            let x = MLXArray([Float](repeating: 0, count: 64), [1, 1, 64]).asType(.bfloat16)
            let ids = MLXArray([Int32(0)], [1, 1, 1])
            let scores = MLXArray([Float(1)], [1, 1, 1])
            #expect(throws: AffineStreamingExpertError.self) {
                try reducer.researchQwen35GroupedReduced(x, indices: ids, scores: scores,
                    groupLimit: 1, tokenWindow: 1, limits: tight)
            }
            #expect(reducer.cachedExpertCount == 0)
        }
    }

    @Test("arena mapping slot is bounded and retains a cancelled mapping")
    func qwen35ArenaMapping() throws {
        try MLXMetalTestLock.withLock {
            let fixture = try makeFixture(numExperts: 4, dimensions: 64, layout: .qwen35)
            defer { try? FileManager.default.removeItem(at: fixture.directory) }
            let catalog = AffineStreamingExpertCatalog(numExperts: 4, inputDims: 64, hiddenDims: 64, layout: .qwen35)
            try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
            let reducer = AffineStreamingSwitchGLU(inputDims: 64, hiddenDims: 64, numExperts: 4,
                layerIndex: 0, catalog: catalog)
            let x = MLXArray([Float](repeating: 0.25, count: 2 * 64), [1, 2, 64]).asType(.bfloat16)
            let ids = MLXArray([Int32(0), 1], [1, 2, 1])
            let scores = MLXArray([Float](repeating: 1, count: 2), [1, 2, 1])
            let limit = max(try catalog.mappedExpertBytes(layer: 0, expert: 0),
                            try catalog.mappedExpertBytes(layer: 0, expert: 1))
            let streams = [StreamOrDevice.default.stream, MLX.Stream.cpu]
            let gate = ExperimentalRequestArenaGate()
            try ExperimentalOwnedStreamRequest.run(gate: gate,
                streams: streams.map { ExperimentalOwnedStream(stream: $0) }, operation: { arena in
                    let result = try reducer.researchQwen35GroupedReduced(x, indices: ids, scores: scores,
                        groupLimit: 2, tokenWindow: 2, limits: researchLimits,
                        arena: arena, maxRetainedMappingBytes: limit, retainedStreams: streams)
                    #expect(result.shape == x.shape)
                    #expect(arena.registeredResource(ofType: ExperimentalExpertMappingSlot.self)?.hasMapping == false)
                }, invalidate: {})
            var checks = 0
            var failDrain = true
            weak var slot: ExperimentalExpertMappingSlot?
            do {
                try ExperimentalOwnedStreamRequest.run(gate: gate,
                    streams: streams.map { ExperimentalOwnedStream(stream: $0) } + [
                        ExperimentalOwnedStream(drain: { if failDrain { throw CancellationError() } })],
                    operation: { arena in
                        defer { slot = arena.registeredResource(ofType: ExperimentalExpertMappingSlot.self) }
                        _ = try reducer.researchQwen35GroupedReduced(x, indices: ids, scores: scores,
                            groupLimit: 2, tokenWindow: 2, limits: researchLimits,
                            shouldCancel: { checks += 1; return checks == 5 },
                            arena: arena, maxRetainedMappingBytes: limit, retainedStreams: streams)
                    }, invalidate: {})
                Issue.record("failed drain was accepted")
            } catch is ExperimentalRequestArenaCleanupError {}
            #expect(slot?.hasMapping == true)
            failDrain = false
            try ExperimentalOwnedStreamRequest.recover(gate: gate)
            #expect(slot == nil)
        }
    }

    @Test("Qwen35 mapped reducer stops and recovers after cancellation")
    func qwen35MappedCancellation() throws {
        try MLXMetalTestLock.withLock {
            let fixture = try makeFixture(numExperts: 4, dimensions: 64, layout: .qwen35)
            defer { try? FileManager.default.removeItem(at: fixture.directory) }
            let catalog = AffineStreamingExpertCatalog(numExperts: 4, inputDims: 64,
                hiddenDims: 64, layout: .qwen35)
            try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
            let reducer = AffineStreamingSwitchGLU(inputDims: 64, hiddenDims: 64,
                numExperts: 4, layerIndex: 0, catalog: catalog, cacheExpertLimit: 2)
            let x = MLXArray([Float](repeating: 0.25, count: 8 * 64), [1, 8, 64]).asType(.bfloat16)
            let indices = MLXArray([Int32](repeating: 0, count: 8), [1, 8, 1])
            let scores = MLXArray([Float](repeating: 1, count: 8), [1, 8, 1])
            #expect(throws: CancellationError.self) {
                try reducer.researchQwen35GroupedReduced(x, indices: indices, scores: scores,
                    groupLimit: 4, tokenWindow: 2, limits: researchLimits, shouldCancel: { true })
            }
            #expect(reducer.cachedExpertCount == 0)
            var cancelled = false
            var observed = 0
            #expect(throws: CancellationError.self) {
                try reducer.researchQwen35GroupedReduced(x, indices: indices, scores: scores,
                    groupLimit: 4, tokenWindow: 2, limits: researchLimits, shouldCancel: { cancelled }, observeWindow: { _, _ in
                        observed += 1
                        cancelled = true
                    })
            }
            #expect(observed == 1)
            let recovered = try reducer.researchQwen35GroupedReduced(x, indices: indices,
                scores: scores, groupLimit: 4, tokenWindow: 2, limits: researchLimits)
            MLX.eval(recovered)
            #expect(recovered.shape == x.shape)
            #expect(reducer.cachedExpertCount <= 2)
        }
    }

    @Test("Qwen35 research call keeps one catalog snapshot")
    func qwen35CatalogSnapshot() throws {
        try MLXMetalTestLock.withLock {
            let first = try makeFixture(numExperts: 4, dimensions: 64, layout: .qwen35)
            let second = try makeFixture(numExperts: 4, dimensions: 64, seedShift: 13, layout: .qwen35)
            defer {
                try? FileManager.default.removeItem(at: first.directory)
                try? FileManager.default.removeItem(at: second.directory)
            }
            let catalog = AffineStreamingExpertCatalog(numExperts: 4, inputDims: 64,
                hiddenDims: 64, layout: .qwen35)
            let reducer = AffineStreamingSwitchGLU(inputDims: 64, hiddenDims: 64,
                numExperts: 4, layerIndex: 0, catalog: catalog)
            let x = MLXArray((0 ..< 4 * 64).map { Float($0 % 17) / 17 }, [1, 4, 64]).asType(.bfloat16)
            let ids = MLXArray([Int32(0), 1, 2, 3], [1, 4, 1])
            let scores = MLXArray([Float](repeating: 1, count: 4), [1, 4, 1])
            try catalog.configure(modelDirectory: first.directory, layerCount: 1)
            let expected = try reducer.researchQwen35GroupedReduced(x, indices: ids, scores: scores,
                groupLimit: 4, tokenWindow: 1, limits: researchLimits)
            try catalog.configure(modelDirectory: second.directory, layerCount: 1)
            let other = try reducer.researchQwen35GroupedReduced(x, indices: ids, scores: scores,
                groupLimit: 4, tokenWindow: 1, limits: researchLimits)
            #expect(!MLX.arrayEqual(expected, other).item(Bool.self))
            try catalog.configure(modelDirectory: first.directory, layerCount: 1)
            var changed = false
            var configurationError: Error?
            let actual = try reducer.researchQwen35GroupedReduced(x, indices: ids, scores: scores,
                groupLimit: 4, tokenWindow: 1, limits: researchLimits, observeWindow: { _, _ in
                    if !changed {
                        changed = true
                        do { try catalog.configure(modelDirectory: second.directory, layerCount: 1) }
                        catch { configurationError = error }
                    }
                })
            if let configurationError { throw configurationError }
            #expect(changed)
            #expect(MLX.arrayEqual(actual, expected).item(Bool.self))
        }
    }

    @Test("Qwen35 research admission is process-wide and releases on error")
    func qwen35ResearchAdmission() throws {
        try MLXMetalTestLock.withLock {
            let fixture = try makeFixture(numExperts: 4, dimensions: 64, layout: .qwen35)
            defer { try? FileManager.default.removeItem(at: fixture.directory) }
            let catalog = AffineStreamingExpertCatalog(numExperts: 4, inputDims: 64,
                hiddenDims: 64, layout: .qwen35)
            try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
            let first = AffineStreamingSwitchGLU(inputDims: 64, hiddenDims: 64,
                numExperts: 4, layerIndex: 0, catalog: catalog)
            let second = AffineStreamingSwitchGLU(inputDims: 64, hiddenDims: 64,
                numExperts: 4, layerIndex: 0, catalog: catalog)
            let x = MLXArray([Float](repeating: 0.25, count: 4 * 64), [1, 4, 64]).asType(.bfloat16)
            let indices = MLXArray([Int32](repeating: 0, count: 4), [1, 4, 1])
            let scores = MLXArray([Float](repeating: 1, count: 4), [1, 4, 1])
            var busyCount = 0
            let result = try first.researchQwen35GroupedReduced(x, indices: indices,
                scores: scores, groupLimit: 4, tokenWindow: 2, limits: researchLimits, observeWindow: { _, _ in
                    let overlap = ConcurrentResults()
                    let completed = DispatchSemaphore(value: 0)
                    DispatchQueue.global().async {
                        defer { completed.signal() }
                        do {
                            _ = try second.researchQwen35GroupedReduced(x, indices: indices,
                                scores: scores, groupLimit: 4, tokenWindow: 2, limits: researchLimits)
                            overlap.append(failure: "overlapping research call was admitted")
                        } catch AffineStreamingExpertError.researchBusy {
                            overlap.append(shape: [1])
                        } catch {
                            overlap.append(failure: "unexpected admission error: \(error)")
                        }
                    }
                    if completed.wait(timeout: .now() + 3) == .success {
                        #expect(overlap.failures.isEmpty)
                        busyCount += overlap.shapes.count
                    } else { Issue.record("overlapping admission did not return promptly") }
                })
            #expect(busyCount == 2)
            #expect(result.shape == x.shape)
            #expect(first.cachedExpertCount == 0 && second.cachedExpertCount == 0)
            let after = try second.researchQwen35GroupedReduced(x, indices: indices,
                scores: scores, groupLimit: 4, tokenWindow: 2, limits: researchLimits)
            #expect(MLX.arrayEqual(result, after).item(Bool.self))
            #expect(throws: AffineStreamingExpertError.self) {
                try second.researchQwen35GroupedReduced(x, indices: indices, scores: scores,
                    groupLimit: 0, tokenWindow: 2, limits: researchLimits)
            }
            let invalidLayer = AffineStreamingSwitchGLU(inputDims: 64, hiddenDims: 64,
                numExperts: 4, layerIndex: 1, catalog: catalog)
            #expect(throws: AffineStreamingExpertError.self) {
                try invalidLayer.researchQwen35GroupedReduced(x, indices: indices,
                    scores: scores, groupLimit: 4, tokenWindow: 2, limits: researchLimits)
            }
            let afterError = try first.researchQwen35GroupedReduced(x, indices: indices,
                scores: scores, groupLimit: 4, tokenWindow: 2, limits: researchLimits)
            #expect(MLX.arrayEqual(result, afterError).item(Bool.self))
        }
    }

    #if canImport(Darwin)
    @Test("Qwen35 mapped reducer repeated-window memory diagnostic")
    func qwen35MappedMemoryDiagnostic() throws {
        try MLXMetalTestLock.withLock {
            func footprint() throws -> UInt64 {
                var info = task_vm_info_data_t()
                var count = mach_msg_type_number_t(
                    MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
                let status = withUnsafeMutablePointer(to: &info) { pointer in
                    pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
                    }
                }
                guard status == KERN_SUCCESS else {
                    throw AffineStreamingExpertError.invalidInput("task_info failed: \(status)")
                }
                return UInt64(info.phys_footprint)
            }
            let directory = try makeFixture(numExperts: 8, dimensions: 2048,
                hiddenDimensions: 512, layout: .qwen35, nonPageAligned: true).directory
            defer { try? FileManager.default.removeItem(at: directory) }
            StreamOrDevice.default.stream.synchronize()
            MLX.Memory.clearCache()
            let catalog = AffineStreamingExpertCatalog(numExperts: 8, inputDims: 2048,
                hiddenDims: 512, layout: .qwen35)
            try catalog.configure(modelDirectory: directory, layerCount: 1)
            let reducer = AffineStreamingSwitchGLU(inputDims: 2048, hiddenDims: 512,
                numExperts: 8, layerIndex: 0, catalog: catalog, cacheExpertLimit: 3)
            let singleLong = ProcessInfo.processInfo.environment["AFFINE_DIAGNOSTIC_LONG"] == "1"
            let tokens = singleLong ? 1024 : 64
            let values = (0 ..< tokens * 2048).map { Float(($0 * 17) % 101 - 50) / 37 }
            let x = MLXArray(values, [1, tokens, 2048]).asType(.bfloat16)
            let ids = (0 ..< tokens).flatMap { token in
                (0 ..< 8).map { slot in Int32((token * 3 + slot * 5) % 8) }
            }
            let indices = MLXArray(ids, [1, tokens, 8]).asType(.uint32)
            let scores = MLXArray([Float](repeating: 0.125, count: tokens * 8), [1, tokens, 8])
            MLX.eval(x, indices, scores)
            do {
                let warmup = try reducer.researchQwen35GroupedReduced(x, indices: indices,
                    scores: scores, groupLimit: 32, tokenWindow: 8, limits: researchLimits)
                MLX.eval(warmup)
            }
            let baseline = try footprint()
            MLX.Memory.peakMemory = 0
            let started = Date()
            var highestFootprint = baseline
            var observedWindows = 0
            var samplingFailure: Error?
            let pooled = ProcessInfo.processInfo.environment["AFFINE_DIAGNOSTIC_AUTORELEASE"] == "1"
            let iterations = singleLong ? 1 : 40
            for iteration in 0 ..< iterations {
                func runOne() throws {
                    let result = try reducer.researchQwen35GroupedReduced(x, indices: indices,
                        scores: scores, groupLimit: 32, tokenWindow: 8, limits: researchLimits, observeWindow: { _, _ in
                            observedWindows += 1
                            do {
                                let current = try footprint()
                                highestFootprint = max(highestFootprint, current)
                                if singleLong && observedWindows % 16 == 0 {
                                    print("AFFINE_MEMORY_WINDOW window=\(observedWindows) footprint=\(current) "
                                        + "active=\(MLX.Memory.activeMemory) cache=\(MLX.Memory.cacheMemory)")
                                }
                            }
                            catch { samplingFailure = error }
                        })
                    MLX.eval(result)
                    #expect(result.shape == x.shape)
                    highestFootprint = max(highestFootprint, try footprint())
                }
                if pooled { try autoreleasepool { try runOne() } }
                else { try runOne() }
                highestFootprint = max(highestFootprint, try footprint())
                print("AFFINE_MEMORY_ITER iteration=\(iteration) footprint=\(try footprint()) "
                    + "active=\(MLX.Memory.activeMemory) cache=\(MLX.Memory.cacheMemory) "
                    + "peak=\(MLX.Memory.peakMemory)")
            }
            if let samplingFailure { throw samplingFailure }
            #expect(observedWindows == iterations * tokens / 8)
            print("AFFINE_MEMORY_RESULT pooled=\(pooled) tokens=\(tokens) iterations=\(iterations) "
                + "baseline=\(baseline) sampled_peak=\(highestFootprint) "
                + "active=\(MLX.Memory.activeMemory) cache=\(MLX.Memory.cacheMemory) "
                + "peak=\(MLX.Memory.peakMemory) windows=\(observedWindows) "
                + "elapsed_seconds=\(Date().timeIntervalSince(started))")
        }
    }
    #endif

    @Test("Qwen35 realistic projection geometry preserves grouped native output")
    func qwen35RealisticGeometryDiagnostic() throws {
        try MLXMetalTestLock.withLock {
            // Construct valid affine packed tensors directly; avoid quantizing
            // hundreds of dense experts. Banks are synthetic, not model data.
            func bank(input: Int, output: Int) -> (MLXArray, MLXArray, MLXArray) {
                let packed = MLXArray([UInt32](repeating: 0x76543210,
                    count: output * (input / 8)), [output, input / 8])
                let scale = MLXArray([Float](repeating: 0.015625,
                    count: output * (input / 64)), [output, input / 64]).asType(.bfloat16)
                let bias = MLXArray([Float](repeating: -0.125,
                    count: output * (input / 64)), [output, input / 64]).asType(.bfloat16)
                return (stacked(Array(repeating: packed, count: 256), axis: 0),
                        stacked(Array(repeating: scale, count: 256), axis: 0),
                        stacked(Array(repeating: bias, count: 256), axis: 0))
            }
            let gate = bank(input: 2048, output: 512)
            let up = bank(input: 2048, output: 512)
            let down = bank(input: 512, output: 2048)
            MLX.eval(gate.0, gate.1, gate.2, up.0, up.1, up.2, down.0, down.1, down.2)
            func module(_ b: (MLXArray, MLXArray, MLXArray), input: Int, output: Int)
                -> QuantizedSwitchLinear {
                QuantizedSwitchLinear(inputDims: input, outputDims: output, numExperts: 256,
                    weight: b.0, scales: b.1, biases: b.2, groupSize: 64, bits: 4, mode: .affine)
            }
            let native = SwitchGLU(inputDims: 2048, hiddenDims: 512, numExperts: 256,
                allowFusedGateUpCache: false, compileSeparatedDecode: false)
            try native.update(modules: ModuleChildren.unflattened([
                ("gate_proj", module(gate, input: 2048, output: 512)),
                ("up_proj", module(up, input: 2048, output: 512)),
                ("down_proj", module(down, input: 512, output: 2048)),
            ]), verify: .all)
            let oldPolicy = SwitchLayerRuntime.propagatePromotedOutput
            SwitchLayerRuntime.propagatePromotedOutput = false
            defer { SwitchLayerRuntime.propagatePromotedOutput = oldPolicy }
            for tokens in [1, 8, 128] {
                let ids = (0 ..< tokens).flatMap { token in
                    (0 ..< 8).map { slot in Int32((token * 3 + slot * 17) % 8) }
                }
                let indices = MLXArray(ids, [1, tokens, 8]).asType(.uint32)
                let values: [Float] = (0 ..< tokens * 2048).map { index in
                    let centered: Int = (index * 17) % 101 - 50
                    return Float(centered) / Float(37)
                }
                let x = MLXArray(values, [1, tokens, 2048]).asType(.bfloat16)
                let scores = MLXArray([Float](repeating: 0.125, count: tokens * 8), [1, tokens, 8])
                let expectedRouted = native(x, indices)
                let nativeScores = scores.asType(expectedRouted.dtype)
                let expected = (expectedRouted * nativeScores[.ellipsis, .newAxis]).sum(axis: -2)
                let expanded = expandedDimensions(x, axes: [-2, -3])
                let (sortedX, sortedIndices, inverse) = gatherSort(x: expanded, indices: indices)
                let sortedIDs = sortedIndices.asArray(Int32.self)
                let useRHS = tokens * 8 >= 1024
                var segments: [MLXArray] = []
                var start = 0
                while start < sortedIDs.count {
                    let expert = Int(sortedIDs[start])
                    var end = start + 1
                    while end < sortedIDs.count && sortedIDs[end] == sortedIDs[start] { end += 1 }
                    var offset = start
                    while offset < end {
                        let stop = offset + min(32, end - offset)
                        let count = stop - offset
                        let padded = useRHS ? max(count, 16) : count
                        var group = sortedX[offset ..< stop, 0..., 0...]
                        if padded > count {
                            group = concatenated([group] + (0 ..< padded - count).map { _ in
                                sortedX[offset ..< offset + 1, 0..., 0...]
                            }, axis: 0)
                        }
                        let localIDs = MLXArray([Int32](repeating: 0, count: padded)).asType(.uint32)
                        func project(_ b: (MLXArray, MLXArray, MLXArray), _ input: MLXArray) -> MLXArray {
                            MLX.gatherQuantizedMM(input, b.0[expert ..< expert + 1, 0..., 0...],
                                scales: b.1[expert ..< expert + 1, 0..., 0...],
                                biases: b.2[expert ..< expert + 1, 0..., 0...],
                                rhsIndices: localIDs, transpose: true, groupSize: 64, bits: 4,
                                mode: .affine, sortedIndices: useRHS).asType(input.dtype)
                        }
                        let a = project(gate, group)
                        let b = project(up, group)
                        let output = project(down, silu(a) * b)
                        segments.append(output[0 ..< count, 0..., 0...])
                        offset = stop
                    }
                    start = end
                }
                let grouped = concatenated(segments, axis: 0)
                let ordered = scatterUnsort(x: grouped, invOrder: inverse, shape: indices.shape)
                let routed = squeezed(ordered, axis: -2)
                let actual = (routed * scores.asType(routed.dtype)[.ellipsis, .newAxis]).sum(axis: -2)
                MLX.eval(actual, expected)
                let difference = MLX.max(MLX.abs(actual.asType(.float32)
                    - expected.asType(.float32))).item(Float.self)
                print("AFFINE_REALISTIC_DIAGNOSTIC tokens=\(tokens) max_abs=\(difference)")
                #expect(actual.shape == expected.shape && actual.dtype == expected.dtype)
                #expect(MLX.arrayEqual(actual, expected).item(Bool.self),
                    "tokens=\(tokens) max_abs=\(difference)")
            }
        }
    }

    @Test("Qwen35 research reducer rejects invalid inputs and unmapped catalogs")
    func qwen35GroupedReducerValidation() throws {
        let catalog = AffineStreamingExpertCatalog(
            numExperts: 4, inputDims: 64, hiddenDims: 64, layout: .qwen35)
        let reducer = AffineStreamingSwitchGLU(inputDims: 64, hiddenDims: 64,
            numExperts: 4, layerIndex: 0, catalog: catalog)
        let x = MLXArray([Float](repeating: 0, count: 64), [1, 1, 64]).asType(.bfloat16)
        let ids = MLXArray([Int32(0)], [1, 1, 1])
        let scores = MLXArray([Float(1)], [1, 1, 1])
        #expect(throws: AffineStreamingExpertError.self) {
            try reducer.researchQwen35GroupedReduced(x, indices: ids, scores: scores, groupLimit: 0, tokenWindow: 32, limits: researchLimits)
        }
        #expect(throws: AffineStreamingExpertError.self) {
            try reducer.researchQwen35GroupedReduced(x, indices: ids, scores: scores,
                groupLimit: 16, tokenWindow: 0, limits: researchLimits)
        }
        #expect(throws: AffineStreamingExpertError.self) {
            try reducer.researchQwen35GroupedReduced(x, indices: ids, scores: scores,
                groupLimit: 16, tokenWindow: .max, limits: researchLimits)
        }
        #expect(throws: AffineStreamingExpertError.self) {
            try reducer.researchQwen35GroupedReduced(x, indices: MLXArray([Int32(4)], [1, 1, 1]),
                scores: scores, groupLimit: 16, tokenWindow: 32, limits: researchLimits)
        }
        #expect(throws: AffineStreamingExpertError.self) {
            try reducer.researchQwen35GroupedReduced(x, indices: ids,
                scores: MLXArray([Float(1), 1], [1, 1, 2]), groupLimit: 16, tokenWindow: 32, limits: researchLimits)
        }
        #expect(throws: AffineStreamingExpertError.self) {
            try reducer.researchQwen35GroupedReduced(x, indices: ids, scores: scores, groupLimit: 16, tokenWindow: 32, limits: researchLimits)
        }
        // Geometry mismatch must be rejected before trying to map an expert.
        let wrongGeometry = AffineStreamingSwitchGLU(inputDims: 64, hiddenDims: 64,
            numExperts: 5, layerIndex: 0, catalog: catalog)
        do {
            _ = try wrongGeometry.researchQwen35GroupedReduced(
                x, indices: ids, scores: scores, groupLimit: 16, tokenWindow: 32, limits: researchLimits)
            Issue.record("mismatched catalog geometry was accepted")
        } catch AffineStreamingExpertError.invalidInput {
            // Validation precedes the unconfigured catalog lookup.
        } catch {
            Issue.record("expected geometry validation, got \(error)")
        }
        let qwen4 = AffineStreamingExpertCatalog(numExperts: 4, inputDims: 64, hiddenDims: 64)
        let wrongLayout = AffineStreamingSwitchGLU(inputDims: 64, hiddenDims: 64,
            numExperts: 4, layerIndex: 0, catalog: qwen4)
        #expect(throws: AffineStreamingExpertError.self) {
            try wrongLayout.researchQwen35GroupedReduced(x, indices: ids, scores: scores, groupLimit: 16, tokenWindow: 32, limits: researchLimits)
        }
    }

    @Test("exact expert regions preserve projection geometry")
    func exactRegionGeometry() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let catalog = AffineStreamingExpertCatalog(
            numExperts: 4, inputDims: 32, hiddenDims: 32)
        try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
        let expert = try catalog.loadExpert(layer: 0, expert: 3)
        #expect(expert.gate.weight.shape == [32, 4])
        #expect(expert.gate.scales.shape == [32, 1])
        #expect(expert.gate.biases.shape == [32, 1])
        #expect(expert.gate.bits == 4)
        #expect(expert.gate.groupSize == 32)
    }

    @Test("selected SSD experts match the full native affine bank")
    func numericalParityAndBoundedCache() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let catalog = AffineStreamingExpertCatalog(
            numExperts: 4, inputDims: 32, hiddenDims: 32)
        try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
        let streaming = AffineStreamingSwitchGLU(
            inputDims: 32, hiddenDims: 32, numExperts: 4, layerIndex: 0,
            catalog: catalog, cacheExpertLimit: 2, prefillChunkSize: 1)

        let input = (MLXArray(0 ..< 32).asType(.float32) / 64).reshaped([1, 1, 32])
        let indices = MLXArray([Int32(3), Int32(1)], [1, 1, 2]).asType(.uint32)
        let scores = MLXArray([Float(0.65), Float(0.35)], [1, 1, 2])
        let actual = streaming.reduced(input, indices: indices, scores: scores)

        func tensor(_ projection: String, _ suffix: String) -> MLXArray {
            fixture.tensors[
                "language_model.layers.0.mlp.switch_mlp.\(projection).\(suffix)"]!
        }
        let expanded = expandedDimensions(input, axes: [-2, -3])
        func project(_ projection: String, _ value: MLXArray) -> MLXArray {
            MLX.gatherQuantizedMM(
                value, tensor(projection, "weight"),
                scales: tensor(projection, "scales"),
                biases: tensor(projection, "biases"), rhsIndices: indices,
                transpose: true, groupSize: 32, bits: 4, mode: .affine)
        }
        let activated = silu(project("gate_proj", expanded))
            * project("up_proj", expanded)
        let routed = squeezed(project("down_proj", activated), axis: -2)
        let expected = (routed * scores[.ellipsis, .newAxis]).sum(axis: -2)
        MLX.eval(expected)

        #expect(MLX.allClose(actual, expected, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        #expect(streaming.cachedExpertCount == 2)

        _ = streaming.reduced(
            input,
            indices: MLXArray([Int32(0), Int32(2)], [1, 1, 2]).asType(.uint32),
            scores: scores)
        #expect(streaming.cachedExpertCount == 2)
    }

    @Test("catalog supports concurrent exact-region opens")
    func concurrentRegionAccess() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let catalog = AffineStreamingExpertCatalog(
            numExperts: 4, inputDims: 32, hiddenDims: 32)
        try catalog.configure(modelDirectory: fixture.directory, layerCount: 1)
        let results = ConcurrentResults()
        DispatchQueue.concurrentPerform(iterations: 16) { iteration in
            do {
                let arrays = try catalog.loadExpert(layer: 0, expert: iteration % 4)
                results.append(shape: arrays.down.weight.shape)
            } catch {
                results.append(failure: String(describing: error))
            }
        }
        #expect(results.failures.isEmpty)
        #expect(results.shapes.count == 16)
        #expect(results.shapes.allSatisfy { $0 == [32, 4] })
    }

    @Test("optional real Qwen4 overlay validates every affine expert descriptor")
    func optionalRealBundleGeometry() throws {
        guard let path = ProcessInfo.processInfo.environment["QWEN4_REAL_MODEL"],
            !path.isEmpty
        else { return }
        let catalog = AffineStreamingExpertCatalog(
            numExperts: 512, inputDims: 2560, hiddenDims: 640)
        try catalog.configure(
            modelDirectory: URL(fileURLWithPath: path, isDirectory: true),
            layerCount: 48)
        let first = try catalog.loadExpert(layer: 0, expert: 0)
        let last = try catalog.loadExpert(layer: 47, expert: 511)
        #expect(first.gate.weight.shape == [640, 320])
        #expect(first.down.weight.shape == [2560, 80])
        #expect(last.gate.groupSize == 64)
        #expect(last.gate.bits == 4)
        #expect(last.down.groupSize == 64)
        #expect(last.down.bits == 4)
    }
}
