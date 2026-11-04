// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation
import MLX
import MLXNN

/// Errors raised before a file-backed affine expert is admitted to the model
/// graph. These are deliberately structural: a malformed bundle must fail at
/// load instead of silently falling back to a full resident expert bank.
public enum AffineStreamingExpertError: Error, CustomStringConvertible {
    case missingIndex(URL)
    case malformedIndex(URL)
    case missingTensor(String)
    case invalidTensor(String, String)
    case mmapFailed(String)
    case notConfigured
    case invalidExpert(layer: Int, expert: Int)
    case invalidInput(String)

    public var description: String {
        switch self {
        case .missingIndex(let url): return "missing safetensors index: \(url.path)"
        case .malformedIndex(let url): return "malformed safetensors index: \(url.path)"
        case .missingTensor(let name): return "missing affine routed tensor: \(name)"
        case .invalidTensor(let name, let reason):
            return "invalid affine routed tensor \(name): \(reason)"
        case .mmapFailed(let name): return "unable to mmap affine routed tensor region: \(name)"
        case .invalidInput(let reason): return "invalid affine routed input: \(reason)"
        case .notConfigured: return "affine routed expert catalog is not configured"
        case .invalidExpert(let layer, let expert):
            return "invalid routed expert \(expert) for layer \(layer)"
        }
    }
}

/// Header-only catalog for Qwen4's stacked native-affine expert tensors.
///
/// The generic loader must exclude these tensors. At runtime this catalog maps
/// one exact expert slice at a time with `mlx_array_new_mmap_file_region`, so a
/// Metal command never receives the 512-expert backing buffer that caused the
/// entire routed bank to become Wired Memory.
public final class AffineStreamingExpertCatalog: @unchecked Sendable {
    /// Explicit layouts avoid treating a partially matching checkpoint as a
    /// supported model. Qwen4 remains the default for existing callers.
    public enum Layout: Sendable, Equatable {
        case qwen4
        case qwen35
    }
    public enum Projection: String, CaseIterable, Sendable {
        case gate = "gate_proj"
        case up = "up_proj"
        case down = "down_proj"
    }

    public struct ProjectionArrays {
        public let weight: MLXArray
        public let scales: MLXArray
        public let biases: MLXArray
        public let groupSize: Int
        public let bits: Int
    }

    public struct ExpertArrays {
        public let gate: ProjectionArrays
        public let up: ProjectionArrays
        public let down: ProjectionArrays
    }

    private struct IndexFile: Decodable {
        let weightMap: [String: String]
        enum CodingKeys: String, CodingKey { case weightMap = "weight_map" }
    }

    private struct Region: Sendable {
        let url: URL
        let name: String
        let dtype: DType
        let shape: [Int]
        let dataOffset: UInt64
        let dataLength: UInt64
    }

    private struct ProjectionSpec: Sendable {
        let weight: Region
        let scales: Region
        let biases: Region
        let groupSize: Int
        let bits: Int
    }

    private struct LayerSpec: Sendable {
        let gate: ProjectionSpec
        let up: ProjectionSpec
        let down: ProjectionSpec
    }

    public let numExperts: Int
    public let inputDims: Int
    public let hiddenDims: Int
    public let layout: Layout

    private let lock = NSLock()
    private var layers: [Int: LayerSpec] = [:]
    private var configuredDirectory: URL?

    public init(numExperts: Int, inputDims: Int, hiddenDims: Int, layout: Layout = .qwen4) {
        self.numExperts = numExperts
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.layout = layout
    }

    /// Exact key predicate used by Qwen4's generic-loader exclusion gate.
    public static func isRoutedTensorKey(_ originalKey: String, layout: Layout = .qwen4) -> Bool {
        if case .qwen35 = layout {
            let parts = originalKey.split(separator: ".", omittingEmptySubsequences: false)
            guard parts.count == 8, let layer = Int(parts[3]), layer >= 0 else { return false }
            return parts[0] == "language_model" && parts[1] == "model"
                && parts[2] == "layers" && parts[4] == "mlp"
                && parts[5] == "switch_mlp"
                && ["gate_proj", "up_proj", "down_proj"].contains(String(parts[6]))
                && ["weight", "scales", "biases"].contains(String(parts[7]))
        }
        var key = originalKey
        if key.hasPrefix("model.") { key.removeFirst("model.".count) }
        let parts = key.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 7,
            parts[0] == "language_model", parts[1] == "layers",
            Int(parts[2]) != nil, parts[3] == "mlp", parts[4] == "switch_mlp",
            ["gate_proj", "up_proj", "down_proj"].contains(String(parts[5])),
            ["weight", "scales", "biases"].contains(String(parts[6]))
        else { return false }
        return true
    }

    public func configure(modelDirectory: URL, layerCount: Int) throws {
        guard numExperts > 0, inputDims > 0, hiddenDims > 0, layerCount > 0 else {
            throw AffineStreamingExpertError.invalidTensor(
                "model geometry", "expert count, dimensions and layer count must be positive")
        }
        let resolved = modelDirectory.resolvingSymlinksInPath()
        lock.lock()
        if configuredDirectory == resolved, layers.count == layerCount {
            lock.unlock()
            return
        }
        lock.unlock()

        let indexURL = resolved.appendingPathComponent("model.safetensors.index.json")
        guard let data = try? Data(contentsOf: indexURL) else {
            throw AffineStreamingExpertError.missingIndex(indexURL)
        }
        guard let index = try? JSONDecoder().decode(IndexFile.self, from: data) else {
            throw AffineStreamingExpertError.malformedIndex(indexURL)
        }

        var shards: [URL: JangPressShard] = [:]
        var built: [Int: LayerSpec] = [:]
        built.reserveCapacity(layerCount)

        func canonicalName(layer: Int, projection: Projection, suffix: String) -> String {
            let root = layout == .qwen35 ? "language_model.model.layers" : "language_model.layers"
            return "\(root).\(layer).mlp.switch_mlp.\(projection.rawValue).\(suffix)"
        }

        func locate(_ canonical: String) throws -> Region {
            let indexedName: String
            let filename: String
            if let found = index.weightMap[canonical] {
                indexedName = canonical
                filename = found
            } else if layout == .qwen4, let found = index.weightMap["model.\(canonical)"] {
                indexedName = "model.\(canonical)"
                filename = found
            } else {
                throw AffineStreamingExpertError.missingTensor(canonical)
            }
            let components = filename.split(separator: "/", omittingEmptySubsequences: false)
            guard !filename.hasPrefix("/"),
                !components.isEmpty,
                components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." })
            else {
                throw AffineStreamingExpertError.invalidTensor(indexedName, "invalid shard path")
            }
            let url = resolved.appendingPathComponent(filename).resolvingSymlinksInPath()
            guard url.path.hasPrefix(resolved.path + "/") else {
                throw AffineStreamingExpertError.invalidTensor(indexedName, "shard escapes model directory")
            }
            let shard: JangPressShard
            if let existing = shards[url] {
                shard = existing
            } else {
                shard = try JangPressShard(path: url)
                shards[url] = shard
            }
            guard let descriptor = shard.descriptor(for: indexedName),
                let dtype = Self.dtype(named: descriptor.dtype)
            else {
                throw AffineStreamingExpertError.invalidTensor(
                    indexedName, "missing descriptor or unsupported dtype")
            }
            let expectedBytes = try Self.checkedByteCount(
                shape: descriptor.shape, dtype: dtype, name: indexedName)
            guard descriptor.dataLength == expectedBytes,
                descriptor.dataOffset <= shard.fileSize,
                descriptor.dataLength <= shard.fileSize - descriptor.dataOffset,
                descriptor.dataOffset % UInt64(dtype.size) == 0
            else {
                throw AffineStreamingExpertError.invalidTensor(
                    indexedName, "shape byte count, file bounds, or dtype alignment mismatch")
            }
            return Region(
                url: url, name: indexedName, dtype: dtype, shape: descriptor.shape,
                dataOffset: descriptor.dataOffset, dataLength: descriptor.dataLength)
        }

        func projectionSpec(layer: Int, projection: Projection) throws -> ProjectionSpec {
            let weight = try locate(canonicalName(layer: layer, projection: projection, suffix: "weight"))
            let scales = try locate(canonicalName(layer: layer, projection: projection, suffix: "scales"))
            let biases = try locate(canonicalName(layer: layer, projection: projection, suffix: "biases"))
            let inFeatures = projection == .down ? hiddenDims : inputDims
            let outFeatures = projection == .down ? inputDims : hiddenDims
            let label = weight.name
            let companionType: DType = layout == .qwen35 ? .bfloat16 : .float16
            guard weight.dtype == .uint32, scales.dtype == companionType,
                biases.dtype == companionType,
                weight.shape.count == 3, scales.shape.count == 3,
                biases.shape == scales.shape,
                weight.shape[0] == numExperts, scales.shape[0] == numExperts,
                weight.shape[1] == outFeatures, scales.shape[1] == outFeatures,
                weight.shape[2] > 0, scales.shape[2] > 0
            else {
                let reason = layout == .qwen35
                    ? "expected stacked U32 weight and matching BF16 affine companions"
                    : "expected stacked U32 weight and matching F16 affine companions"
                throw AffineStreamingExpertError.invalidTensor(label, reason)
            }
            let packedBits = weight.shape[2].multipliedReportingOverflow(by: 32)
            guard !packedBits.overflow, packedBits.partialValue % inFeatures == 0 else {
                throw AffineStreamingExpertError.invalidTensor(label, "non-integral packed bit width")
            }
            let bits = packedBits.partialValue / inFeatures
            // MLX's native affine quantized kernels accept 2/4/8-bit packed
            // weights. PLE's independent SSD row decoder supports odd widths,
            // but routed matmul must reject those rather than reaching Metal.
            guard [2, 4, 8].contains(bits),
                inFeatures % scales.shape[2] == 0
            else {
                throw AffineStreamingExpertError.invalidTensor(
                    label, "unsupported bit width or affine group geometry")
            }
            let groupSize = inFeatures / scales.shape[2]
            if layout == .qwen35 && ![32, 64, 128].contains(groupSize) {
                throw AffineStreamingExpertError.invalidTensor(label, "unsupported affine group size")
            }
            return ProjectionSpec(
                weight: weight, scales: scales, biases: biases,
                groupSize: groupSize, bits: bits)
        }

        for layer in 0 ..< layerCount {
            built[layer] = try LayerSpec(
                gate: projectionSpec(layer: layer, projection: .gate),
                up: projectionSpec(layer: layer, projection: .up),
                down: projectionSpec(layer: layer, projection: .down))
        }

        lock.lock()
        configuredDirectory = resolved
        layers = built
        lock.unlock()

        let mappedBytes = shards.values.reduce(UInt64(0)) { $0 + $1.fileSize }
        FileHandle.standardError.write(Data(
            "[Qwen4AffineExperts] exact-region catalog ready layers=\(layerCount) experts=\(numExperts) files=\(shards.count) indexed_file_bytes=\(mappedBytes) cache=file-backed\n".utf8))
    }

    public func loadExpert(layer: Int, expert: Int) throws -> ExpertArrays {
        guard expert >= 0, expert < numExperts else {
            throw AffineStreamingExpertError.invalidExpert(layer: layer, expert: expert)
        }
        lock.lock()
        let layerSpec = layers[layer]
        let configured = configuredDirectory != nil
        lock.unlock()
        guard configured else { throw AffineStreamingExpertError.notConfigured }
        guard let layerSpec else {
            throw AffineStreamingExpertError.invalidExpert(layer: layer, expert: expert)
        }
        return try ExpertArrays(
            gate: loadProjection(layerSpec.gate, expert: expert),
            up: loadProjection(layerSpec.up, expert: expert),
            down: loadProjection(layerSpec.down, expert: expert))
    }

    private func loadProjection(_ spec: ProjectionSpec, expert: Int) throws -> ProjectionArrays {
        ProjectionArrays(
            weight: try mapExpertRegion(spec.weight, expert: expert),
            scales: try mapExpertRegion(spec.scales, expert: expert),
            biases: try mapExpertRegion(spec.biases, expert: expert),
            groupSize: spec.groupSize,
            bits: spec.bits)
    }

    private func mapExpertRegion(_ region: Region, expert: Int) throws -> MLXArray {
        guard region.shape.first == numExperts,
            region.dataLength % UInt64(numExperts) == 0
        else {
            throw AffineStreamingExpertError.invalidTensor(
                region.name, "expert dimension does not divide payload")
        }
        let length = region.dataLength / UInt64(numExperts)
        let offsetResult = UInt64(expert).multipliedReportingOverflow(by: length)
        guard !offsetResult.overflow,
            region.dataOffset <= UInt64.max - offsetResult.partialValue
        else {
            throw AffineStreamingExpertError.invalidTensor(region.name, "expert offset overflow")
        }
        let offset = region.dataOffset + offsetResult.partialValue
        let shape = Array(region.shape.dropFirst())
        var cShape = shape.map(Int32.init)
        var result = mlx_array_new()
        let rc = region.url.withUnsafeFileSystemRepresentation { path -> Int32 in
            guard let path else { return 1 }
            return mlx_array_new_mmap_file_region(
                &result, path, offset, Int(length), &cShape, Int32(cShape.count),
                region.dtype.cmlxDtype)
        }
        guard rc == 0 else {
            mlx_array_free(result)
            throw AffineStreamingExpertError.mmapFailed("\(region.name)[\(expert)]")
        }
        return MLXArray(result)
    }

    private static func dtype(named name: String) -> DType? {
        switch name.uppercased() {
        case "U32": return .uint32
        case "F16": return .float16
        case "BF16": return .bfloat16
        case "F32": return .float32
        default: return nil
        }
    }

    private static func checkedByteCount(shape: [Int], dtype: DType, name: String) throws -> UInt64 {
        var count = UInt64(dtype.size)
        for dimension in shape {
            guard dimension >= 0 else {
                throw AffineStreamingExpertError.invalidTensor(name, "negative dimension")
            }
            let next = count.multipliedReportingOverflow(by: UInt64(dimension))
            guard !next.overflow else {
                throw AffineStreamingExpertError.invalidTensor(name, "shape byte count overflow")
            }
            count = next.partialValue
        }
        return count
    }
}

/// Qwen4-only native affine MoE path backed by exact safetensors expert slices.
/// The cache retains file mappings, not copied tensors, and is strictly bounded
/// per layer. Each chunk is evaluated before returning so mappings from old
/// routing decisions can be evicted and unpinned before the next layer runs.
public final class AffineStreamingSwitchGLU: Module {
    private struct CachedExpert {
        let arrays: AffineStreamingExpertCatalog.ExpertArrays
        var access: UInt64
    }

    public let inputDims: Int
    public let hiddenDims: Int
    public let numExperts: Int
    public let layerIndex: Int
    public let cacheExpertLimit: Int
    public let prefillChunkSize: Int

    private let catalog: AffineStreamingExpertCatalog
    private let lock = NSLock()
    private var cache: [Int: CachedExpert] = [:]
    private var accessClock: UInt64 = 0

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        layerIndex: Int,
        catalog: AffineStreamingExpertCatalog,
        cacheExpertLimit: Int = 16,
        prefillChunkSize: Int = 4
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.layerIndex = layerIndex
        self.catalog = catalog
        self.cacheExpertLimit = max(1, cacheExpertLimit)
        self.prefillChunkSize = max(1, prefillChunkSize)
        super.init()
    }

    public var cachedExpertCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return cache.count
    }

    public func reduced(_ x: MLXArray, indices: MLXArray, scores: MLXArray) -> MLXArray {
        let totalTokens = x.size / inputDims
        let kSlots = indices.dim(-1)
        let xFlat = x.reshaped([totalTokens, inputDims])
        let indicesFlat = indices.reshaped([totalTokens, kSlots])
        let scoresFlat = scores.reshaped([totalTokens, kSlots])
        var chunks: [MLXArray] = []
        chunks.reserveCapacity((totalTokens + prefillChunkSize - 1) / prefillChunkSize)

        var start = 0
        while start < totalTokens {
            let end = min(totalTokens, start + prefillChunkSize)
            let chunkIndices = indicesFlat[start ..< end, 0...]
            let ids = chunkIndices.reshaped([-1]).asArray(Int32.self).map(Int.init)
            guard !ids.isEmpty, ids.allSatisfy({ $0 >= 0 && $0 < numExperts }) else {
                fatalError("[Qwen4AffineExperts] invalid routed IDs at layer \(layerIndex)")
            }
            let unique = Array(Set(ids)).sorted()
            let selected = unique.map { expert($0) }
            var remap: [Int: Int32] = [:]
            remap.reserveCapacity(unique.count)
            for (local, global) in unique.enumerated() { remap[global] = Int32(local) }
            let localIndices = MLXArray(
                ids.map { remap[$0]! }, chunkIndices.shape).asType(.uint32)

            func bank(
                _ projection: (AffineStreamingExpertCatalog.ExpertArrays)
                    -> AffineStreamingExpertCatalog.ProjectionArrays,
                _ value: (AffineStreamingExpertCatalog.ProjectionArrays) -> MLXArray
            ) -> MLXArray {
                let arrays = selected.map { value(projection($0)) }
                return arrays.count == 1
                    ? expandedDimensions(arrays[0], axis: 0)
                    : stacked(arrays, axis: 0)
            }

            let gate0 = selected[0].gate
            let up0 = selected[0].up
            let down0 = selected[0].down
            let input = expandedDimensions(xFlat[start ..< end, 0...], axes: [-2, -3])
            func nativeProjectionDType(_ result: MLXArray, input: MLXArray) -> MLXArray {
                guard catalog.layout == .qwen35,
                    !SwitchLayerRuntime.propagatePromotedOutput,
                    result.dtype != input.dtype,
                    input.dtype == .bfloat16 || input.dtype == .float16
                else { return result }
                return result.asType(input.dtype)
            }
            let upRaw = MLX.gatherQuantizedMM(
                input, bank({ $0.up }, { $0.weight }),
                scales: bank({ $0.up }, { $0.scales }),
                biases: bank({ $0.up }, { $0.biases }), rhsIndices: localIndices,
                transpose: true, groupSize: up0.groupSize, bits: up0.bits,
                mode: .affine)
            let gateRaw = MLX.gatherQuantizedMM(
                input, bank({ $0.gate }, { $0.weight }),
                scales: bank({ $0.gate }, { $0.scales }),
                biases: bank({ $0.gate }, { $0.biases }), rhsIndices: localIndices,
                transpose: true, groupSize: gate0.groupSize, bits: gate0.bits,
                mode: .affine)
            let up = nativeProjectionDType(upRaw, input: input)
            let gate = nativeProjectionDType(gateRaw, input: input)
            let activated = silu(gate) * up
            let downRaw = MLX.gatherQuantizedMM(
                activated, bank({ $0.down }, { $0.weight }),
                scales: bank({ $0.down }, { $0.scales }),
                biases: bank({ $0.down }, { $0.biases }), rhsIndices: localIndices,
                transpose: true, groupSize: down0.groupSize, bits: down0.bits,
                mode: .affine)
            let down = nativeProjectionDType(downRaw, input: activated)
            let routed = squeezed(down, axis: -2)
            let chunkScores = scoresFlat[start ..< end, 0...]
            // Qwen35's resident MoE casts routing scores to the expert
            // output dtype before multiplication. Keep the Qwen4 behavior
            // unchanged and preserve that rounding point for Qwen35.
            let weightedScores = catalog.layout == .qwen35
                ? chunkScores.asType(routed.dtype) : chunkScores
            let reduced = (routed * weightedScores[.ellipsis, .newAxis])
                .sum(axis: -2)
            MLX.eval(reduced)
            chunks.append(reduced)
            start = end
        }

        let result = chunks.count == 1 ? chunks[0] : concatenated(chunks, axis: 0)
        // This path REQUIRES f16 scales, so gatherQuantizedMM promotes bf16
        // activations to fp32 (promote_types) and the whole MoE block leaked
        // fp32 into the residual stream. Pin to the activation dtype.
        let reshaped = result.reshaped(x.shape)
        return catalog.layout == .qwen35 ? reshaped : reshaped.asType(x.dtype)
    }

    /// Experimental Qwen35-only reducer. It intentionally is not connected to
    /// model loading. Callers must handle validation or mapping failures.
    /// cacheExpertLimit bounds mappings per layer, not global mapped bytes.
    public func researchQwen35GroupedReduced(
        _ x: MLXArray, indices: MLXArray, scores: MLXArray, groupLimit: Int,
        tokenWindow: Int, observeWindow: ((Int, Int) -> Void)? = nil
    ) throws -> MLXArray {
        #if canImport(ObjectiveC)
        return try autoreleasepool {
            try researchQwen35GroupedReducedImpl(x, indices: indices, scores: scores,
                groupLimit: groupLimit, tokenWindow: tokenWindow, observeWindow: observeWindow)
        }
        #else
        return try researchQwen35GroupedReducedImpl(x, indices: indices, scores: scores,
            groupLimit: groupLimit, tokenWindow: tokenWindow, observeWindow: observeWindow)
        #endif
    }

    private func researchQwen35GroupedReducedImpl(
        _ x: MLXArray, indices: MLXArray, scores: MLXArray, groupLimit: Int,
        tokenWindow: Int, observeWindow: ((Int, Int) -> Void)?
    ) throws -> MLXArray {
        guard catalog.layout == .qwen35 else {
            throw AffineStreamingExpertError.invalidInput("Qwen35 layout required")
        }
        guard inputDims > 0, numExperts > 0, groupLimit > 0, tokenWindow > 0, tokenWindow < .max,
            inputDims == catalog.inputDims, hiddenDims == catalog.hiddenDims,
            numExperts == catalog.numExperts,
            !x.shape.isEmpty, x.dim(-1) == inputDims,
            indices.shape.count == x.shape.count, indices.shape.dropLast() == x.shape.dropLast(),
            (indices.shape.last ?? 0) > 0, indices.shape == scores.shape,
            indices.dtype == .int32 || indices.dtype == .uint32,
            x.size > 0
        else { throw AffineStreamingExpertError.invalidInput("invalid shapes, dtype, or group limit") }
        let totalTokens = x.size / inputDims
        let kSlots = indices.dim(-1)
        let xFlat = x.reshaped([totalTokens, inputDims])
        let indicesFlat = indices.reshaped([totalTokens, kSlots])
        let scoresFlat = scores.reshaped([totalTokens, kSlots])
        // Select the kernel from the original full-bank call, not the window.
        let fullRoutes = indices.size
        let useRHS = fullRoutes >= 64 && fullRoutes >= 16 && fullRoutes / numExperts >= 4
        var outputs: [MLXArray] = []
        var tokenStart = 0
        while tokenStart < totalTokens {
            let tokenEnd = tokenStart + min(tokenWindow, totalTokens - tokenStart)
            func reduceWindow() throws -> MLXArray {
                let windowIndices = indicesFlat[tokenStart ..< tokenEnd, 0...]
                let ids = windowIndices.flattened().asArray(Int32.self)
                guard ids.allSatisfy({ $0 >= 0 && $0 < numExperts }) else {
                    throw AffineStreamingExpertError.invalidInput("routed expert ID out of range")
                }
                let windowX = xFlat[tokenStart ..< tokenEnd, 0...]
                let expanded = expandedDimensions(windowX, axes: [-2, -3])
                let (sortedX, sortedIndices, inverse) = gatherSort(x: expanded, indices: windowIndices)
                let sortedIDs = sortedIndices.asArray(Int32.self)
                let routes = sortedIDs.count
                var segments: [MLXArray] = []
                var paddedRows = 0
                var start = 0
                while start < routes {
                    let expertID = Int(sortedIDs[start])
                    var end = start + 1
                    while end < routes && sortedIDs[end] == sortedIDs[start] { end += 1 }
                    let mapped = try cachedExpert(expertID)
                    var offset = start
                    while offset < end {
                        let stop = offset + min(groupLimit, end - offset)
                        let count = stop - offset
                        let padded = useRHS ? max(count, 16) : count
                        paddedRows += padded
                        var group = sortedX[offset ..< stop, 0..., 0...]
                        if padded > count {
                            let copies = (0 ..< padded - count).map { _ in
                                sortedX[offset ..< offset + 1, 0..., 0...]
                            }
                            group = concatenated([group] + copies, axis: 0)
                        }
                        let localIDs = MLXArray([Int32](repeating: 0, count: padded)).asType(.uint32)
                        func project(_ projection: AffineStreamingExpertCatalog.ProjectionArrays,
                                     _ input: MLXArray) -> MLXArray {
                            let result = MLX.gatherQuantizedMM(input,
                                expandedDimensions(projection.weight, axis: 0),
                                scales: expandedDimensions(projection.scales, axis: 0),
                                biases: expandedDimensions(projection.biases, axis: 0),
                                rhsIndices: localIDs, transpose: true, groupSize: projection.groupSize,
                                bits: projection.bits, mode: .affine, sortedIndices: useRHS)
                            if !SwitchLayerRuntime.propagatePromotedOutput, result.dtype != input.dtype,
                                input.dtype == .bfloat16 || input.dtype == .float16 {
                                return result.asType(input.dtype)
                            }
                            return result
                        }
                        let gate = project(mapped.gate, group)
                        let up = project(mapped.up, group)
                        let down = project(mapped.down, silu(gate) * up)
                        let kept = down[0 ..< count, 0..., 0...]
                        MLX.eval(kept)
                        segments.append(kept)
                        offset = stop
                    }
                    start = end
                }
                let grouped = concatenated(segments, axis: 0)
                observeWindow?(grouped.dim(0), paddedRows)
                let ordered = scatterUnsort(x: grouped, invOrder: inverse, shape: windowIndices.shape)
                let routed = squeezed(ordered, axis: -2)
                let windowScores = scoresFlat[tokenStart ..< tokenEnd, 0...]
                let output = (routed * windowScores.asType(routed.dtype)[.ellipsis, .newAxis])
                    .sum(axis: -2)
                MLX.eval(output)
                return output
            }
            #if canImport(ObjectiveC)
            let output = try autoreleasepool { try reduceWindow() }
            #else
            let output = try reduceWindow()
            #endif
            outputs.append(output)
            tokenStart = tokenEnd
        }
        return concatenated(outputs, axis: 0).reshaped(x.shape)
    }

    private func expert(_ index: Int) -> AffineStreamingExpertCatalog.ExpertArrays {
        do { return try cachedExpert(index) }
        catch { fatalError("[Qwen4AffineExperts] \(error)") }
    }

    private func cachedExpert(_ index: Int) throws -> AffineStreamingExpertCatalog.ExpertArrays {
        lock.lock()
        accessClock &+= 1
        if var cached = cache[index] {
            cached.access = accessClock
            cache[index] = cached
            lock.unlock()
            return cached.arrays
        }
        lock.unlock()

        let loaded = try catalog.loadExpert(layer: layerIndex, expert: index)

        lock.lock()
        accessClock &+= 1
        if let raced = cache[index] {
            lock.unlock()
            return raced.arrays
        }
        cache[index] = CachedExpert(arrays: loaded, access: accessClock)
        while cache.count > cacheExpertLimit,
            let victim = cache.min(by: { $0.value.access < $1.value.access })?.key
        {
            cache.removeValue(forKey: victim)
        }
        let count = cache.count
        lock.unlock()

        if accessClock == 2 || accessClock % 256 == 0 {
            FileHandle.standardError.write(Data(
                "[Qwen4AffineExperts] layer=\(layerIndex) mapped_expert=\(index) cache=\(count)/\(cacheExpertLimit) backend=exact-mmap-region mode=affine\n".utf8))
        }
        return loaded
    }
}
