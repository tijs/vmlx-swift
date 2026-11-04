// Copyright © 2026 Apple Inc.
import Foundation
import MLX
import MLXNN

/// Unwired research-only affine leaf installer. Caller must own a fresh,
/// unpublished model and discard it on any failure, validate checkpoint policy
/// and exact shard identity separately, and prevent escaping aliases. `load`
/// must construct lazy arrays without scheduling GPU work. The nominal byte
/// cap covers the active source triplet, not model/RSS/padding/caches. Streams
/// must cover every producer and must be quiescent before each checkpoint.
enum ExperimentalAffineLeafInstall {
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
        var replacement: Module?
        init(model: Module, streams: [MLX.Stream], maxBytes: UInt64) {
            self.model = model; self.streams = streams; self.maxBytes = maxBytes
        }
        func clear() { sources.removeAll(); replacement = nil }
    }

    static func install(model: Module, path: String, arena: ExperimentalRequestArena,
        streams: [MLX.Stream], maxBytes: UInt64, shouldCancel: () -> Bool = { false },
        load: () throws -> Triplet) throws {
        try ExperimentalOwnedStreamRequest.validateMappingStreams(arena: arena, streams: streams)
        try perform(model: model, path: path, arena: arena, streams: streams, maxBytes: maxBytes,
            shouldCancel: shouldCancel, load: load,
            evaluate: { try MLX.checkedEval($0) }, checkpoint: {
                for stream in streams { try stream.experimentalSynchronize() }
            })
    }

    /// Internal evaluator/checkpoint injection is test-only; callers use install.
    static func perform(model: Module, path: String, arena: ExperimentalRequestArena,
        streams: [MLX.Stream], maxBytes: UInt64, shouldCancel: () -> Bool = { false },
        load: () throws -> Triplet, evaluate: ([MLXArray]) throws -> Void,
        checkpoint: () throws -> Void) throws {
        func cancel() throws {
            if shouldCancel() || Task<Never, Never>.isCancelled { throw CancellationError() }
        }
        try cancel()
        let components = path.split(separator: ".", omittingEmptySubsequences: false)
        let isRouter = components.count >= 2 && components[components.count - 2] == "mlp"
            && (components.last == "gate" || components.last == "shared_expert_gate")
        guard !path.isEmpty, !streams.isEmpty, maxBytes > 0,
            !components.contains("switch_mlp"), !isRouter else {
            throw InstallError("invalid or routed leaf path")
        }
        var leaves: [String: Module] = [:]
        for (key, leaf) in model.leafModules().flattened() {
            guard leaves.updateValue(leaf, forKey: key) == nil else { throw InstallError("duplicate module path") }
        }
        guard let original = leaves[path] else { throw InstallError("missing module: \(path)") }
        let out: Int, input: Int
        if type(of: original) == Linear.self, let linear = original as? Linear, linear.bias == nil {
            guard linear.weight.ndim == 2 else { throw InstallError("invalid linear weight rank: \(path)") }
            (out, input) = linear.shape
        } else if type(of: original) == Embedding.self, let embedding = original as? Embedding {
            guard embedding.weight.ndim == 2 else { throw InstallError("invalid embedding weight rank: \(path)") }
            (out, input) = embedding.shape
        } else { throw InstallError("unsupported leaf or ordinary bias: \(path)") }
        guard out > 0, input > 0, input % 64 == 0 else { throw InstallError("invalid q4/group64 geometry") }
        let rows = UInt64(out), packedColumns = UInt64(input / 8), groups = UInt64(input / 64)
        let packedElements = rows.multipliedReportingOverflow(by: packedColumns)
        let groupElements = rows.multipliedReportingOverflow(by: groups)
        guard !packedElements.overflow, !groupElements.overflow else { throw InstallError("geometry overflow") }
        let packedBytes = packedElements.partialValue.multipliedReportingOverflow(by: 4)
        let companionBytes = groupElements.partialValue.multipliedReportingOverflow(by: 4)
        guard !packedBytes.overflow, !companionBytes.overflow else { throw InstallError("byte size overflow") }
        let total = packedBytes.partialValue.addingReportingOverflow(companionBytes.partialValue)
        guard !total.overflow, total.partialValue <= maxBytes else { throw InstallError("leaf byte budget exceeded") }
        let owner: Owner
        if let existing = arena.registeredResource(ofType: Owner.self) {
            guard existing.model === model, existing.streams == streams, existing.maxBytes == maxBytes,
                existing.sources.isEmpty, existing.replacement == nil else {
                throw InstallError("arena owner mismatch or occupied slot")
            }
            owner = existing
        } else {
            owner = Owner(model: model, streams: streams, maxBytes: maxBytes)
            try arena.retain(owner)
        }
        try cancel()
        let source = try load()
        owner.sources = [source.weight, source.scales, source.biases]
        try cancel()
        guard source.weight.dtype == .uint32, source.weight.shape == [out, input / 8],
            source.scales.dtype == .bfloat16, source.biases.dtype == .bfloat16,
            source.scales.shape == [out, input / 64], source.biases.shape == [out, input / 64] else {
            throw InstallError("invalid q4 affine triplet: \(path)")
        }
        try MLX.withError { error in
            try cancel()
            try evaluate(owner.sources)
            try error.check()
            try cancel()
            try checkpoint()
            try cancel()
            let replacement: Module
            if type(of: original) == Linear.self {
                replacement = QuantizedLinear(weight: source.weight, bias: nil, scales: source.scales,
                    biases: source.biases, groupSize: 64, bits: 4, mode: .affine)
            } else {
                replacement = QuantizedEmbedding(weight: source.weight, scales: source.scales,
                    biases: source.biases, groupSize: 64, bits: 4, mode: .affine)
            }
            owner.replacement = replacement
            try error.check()
            try cancel()
            try model.update(modules: ModuleChildren.unflattened([(path, replacement)]), verify: .noUnusedKeys)
            try error.check()
        }
        // BF16 companion arrays may remain file-backed via the installed
        // model; checkpoint ownership must outlive the model and its aliases.
        owner.clear()
        try cancel()
    }
}
