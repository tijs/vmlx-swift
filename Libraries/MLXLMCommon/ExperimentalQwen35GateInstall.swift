// Copyright © 2026 Apple Inc.
import Foundation
import MLX
import MLXNN

/// Unwired research-only Qwen35 gate transform. The caller must bind all
/// arguments to the independently pinned config/immutable checkpoint and own
/// the unpublished model; this helper cannot prove provenance or prevent
/// aliases. `load` must construct lazy arrays without submitting work. The
/// active-byte cap excludes allocator caches, padding, model and total RSS.
/// Installation is per-leaf and non-atomic across a model. Cancellation or
/// failure may follow a successful update; discard the entire private model
/// after verified stream drain and invalidation.
enum ExperimentalQwen35GateInstall {
    struct InstallError: Error, CustomStringConvertible {
        let description: String
        init(_ description: String) { self.description = description }
    }
    struct Triplet {
        let weight: MLXArray
        let scales: MLXArray
        let biases: MLXArray
    }
    private final class Owner {
        let model: Module
        let streams: [MLX.Stream]
        let maxBytes: UInt64
        var sources: [MLXArray] = []
        var graphs: [MLXArray] = []
        init(model: Module, streams: [MLX.Stream], maxBytes: UInt64) {
            self.model = model; self.streams = streams; self.maxBytes = maxBytes
        }
        func clear() { sources.removeAll(); graphs.removeAll() }
    }

    static func install(model: Module, path: String, layerCount: Int, numExperts: Int,
        hiddenSize: Int, bits: Int, groupSize: Int, arena: ExperimentalRequestArena,
        streams: [MLX.Stream], conversionStream: MLX.Stream, maxBytes: UInt64,
        shouldCancel: () -> Bool = { false }, load: () throws -> Triplet) throws {
        try ExperimentalOwnedStreamRequest.validateMappingStreams(arena: arena, streams: streams)
        guard streams.contains(conversionStream) else { throw InstallError("conversion stream not owned") }
        try perform(model: model, path: path, layerCount: layerCount, numExperts: numExperts,
            hiddenSize: hiddenSize, bits: bits, groupSize: groupSize, arena: arena,
            streams: streams, conversionStream: conversionStream, maxBytes: maxBytes,
            shouldCancel: shouldCancel, load: load,
            evaluate: { try MLX.checkedEval($0) }, checkpoint: {
                for stream in streams { try stream.experimentalSynchronize() }
            })
    }

    static func perform(model: Module, path: String, layerCount: Int, numExperts: Int,
        hiddenSize: Int, bits: Int, groupSize: Int, arena: ExperimentalRequestArena,
        streams: [MLX.Stream], conversionStream: MLX.Stream, maxBytes: UInt64,
        shouldCancel: () -> Bool = { false }, load: () throws -> Triplet,
        evaluate: ([MLXArray]) throws -> Void, checkpoint: () throws -> Void) throws {
        func cancel() throws {
            if shouldCancel() || Task<Never, Never>.isCancelled { throw CancellationError() }
        }
        try cancel()
        let parts = path.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 6, parts[0] == "language_model", parts[1] == "model",
            parts[2] == "layers", let layer = Int(parts[3]), layer >= 0, layer < layerCount,
            parts[4] == "mlp", parts[5] == "gate" || parts[5] == "shared_expert_gate",
            numExperts > 0, hiddenSize > 0, bits == 8, groupSize == 64, hiddenSize % 64 == 0,
            !streams.isEmpty, streams.contains(conversionStream), maxBytes > 0 else {
            throw InstallError("unsupported Qwen35 gate path or policy")
        }
        var leaves: [String: Module] = [:]
        for (key, leaf) in model.leafModules().flattened() {
            guard leaves.updateValue(leaf, forKey: key) == nil else { throw InstallError("duplicate model path") }
        }
        guard let leaf = leaves[path], type(of: leaf) == Linear.self, let linear = leaf as? Linear,
            linear.bias == nil, linear.weight.ndim == 2 else { throw InstallError("gate must be pristine bias-free Linear") }
        let output = parts[5] == "gate" ? numExperts : 1
        guard linear.shape.0 == output, linear.shape.1 == hiddenSize else { throw InstallError("gate geometry mismatch") }
        let elements = UInt64(output).multipliedReportingOverflow(by: UInt64(hiddenSize))
        let groups = UInt64(output).multipliedReportingOverflow(by: UInt64(hiddenSize / 64))
        guard !elements.overflow, !groups.overflow else { throw InstallError("geometry overflow") }
        // Source q8 bytes + BF16 scale/bias + BF16 dequant/F32/BF16 graphs.
        let graphs = elements.partialValue.multipliedReportingOverflow(by: 9)
        let metadata = groups.partialValue.multipliedReportingOverflow(by: 4)
        guard !graphs.overflow, !metadata.overflow else { throw InstallError("byte size overflow") }
        let total = graphs.partialValue.addingReportingOverflow(metadata.partialValue)
        guard !total.overflow, total.partialValue <= maxBytes else { throw InstallError("gate byte budget exceeded") }
        let owner: Owner
        if let existing = arena.registeredResource(ofType: Owner.self) {
            guard existing.model === model, existing.streams == streams, existing.maxBytes == maxBytes,
                existing.sources.isEmpty, existing.graphs.isEmpty else { throw InstallError("arena owner mismatch or occupied slot") }
            owner = existing
        } else {
            owner = Owner(model: model, streams: streams, maxBytes: maxBytes)
            try arena.retain(owner)
        }
        try cancel()
        let source = try load()
        owner.sources = [source.weight, source.scales, source.biases]
        try cancel()
        guard source.weight.dtype == .uint32, source.weight.shape == [output, hiddenSize / 4],
            source.scales.dtype == .bfloat16, source.biases.dtype == .bfloat16,
            source.scales.shape == [output, hiddenSize / 64], source.biases.shape == [output, hiddenSize / 64]
        else { throw InstallError("invalid q8 gate triplet") }
        try MLX.withError { error in
            let decoded = MLX.dequantized(source.weight, scales: source.scales, biases: source.biases,
                groupSize: 64, bits: 8, mode: .affine, stream: .stream(conversionStream))
            owner.graphs.append(decoded); try error.check(); try cancel()
            guard decoded.dtype == .bfloat16, decoded.shape == [output, hiddenSize] else {
                throw InstallError("unexpected dequantized gate dtype or shape")
            }
            let fp32 = decoded.asType(.float32, stream: .stream(conversionStream))
            owner.graphs.append(fp32); try error.check(); try cancel()
            guard fp32.dtype == .float32 else { throw InstallError("unexpected F32 intermediate") }
            let final = fp32.asType(.bfloat16, stream: .stream(conversionStream))
            owner.graphs.append(final); try error.check(); try cancel()
            guard final.dtype == .bfloat16 else { throw InstallError("unexpected final gate dtype") }
            try evaluate(owner.graphs); try error.check(); try cancel()
            try checkpoint(); try cancel()
            try model.update(parameters: ModuleParameters.unflattened([(path + ".weight", final)]),
                verify: [.noUnusedKeys, .shapeMismatch])
            try error.check()
        }
        owner.clear()
        try cancel()
    }
}
