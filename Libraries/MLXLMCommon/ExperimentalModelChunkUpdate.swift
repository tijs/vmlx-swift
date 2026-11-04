// Copyright © 2026 Apple Inc.
import Foundation
import MLX
import MLXNN

/// Research-only explicit-parameter updater; not part of the generic loader.
/// The caller owns a fresh unpublished model and must discard it on any error.
/// `loadChunk` must only construct lazy arrays and must not submit GPU work;
/// it must not expose the model or aliases. Caller must stop/join producers,
/// retain/drain every actual stream, and hold source files immutable. The byte
/// cap is nominal for the active chunk only; it does not bound model weights,
/// external inputs, allocator caches, graph storage, page rounding or RSS.
/// BF16 passthrough preserves the source array and may remain mmap-backed; the
/// immutable snapshot must outlive the model and every alias of these arrays.
enum ExperimentalModelChunkUpdate {
    struct UpdateError: Error, CustomStringConvertible {
        let description: String
        init(_ description: String) { self.description = description }
    }
    struct Descriptor {
        let key: String
        let shape: [Int]
        let dtype: DType
        init(key: String, shape: [Int], dtype: DType) {
            self.key = key; self.shape = shape; self.dtype = dtype
        }
    }
    private final class Slot {
        var sources: [MLXArray] = []
        var outputs: [MLXArray] = []
        func clear() { sources.removeAll(); outputs.removeAll() }
    }

    static func update(model: Module, descriptors: [Descriptor], arena: ExperimentalRequestArena,
        streams: [MLX.Stream], conversionStream: MLX.Stream, maxChunkBytes: UInt64,
        shouldCancel: () -> Bool = { false }, loadChunk: ([String]) throws -> [String: MLXArray]) throws {
        try ExperimentalOwnedStreamRequest.validateMappingStreams(arena: arena, streams: streams)
        guard streams.contains(conversionStream) else { throw UpdateError("conversion stream not owned") }
        try perform(model: model, descriptors: descriptors, arena: arena, stream: conversionStream,
            maxChunkBytes: maxChunkBytes, shouldCancel: shouldCancel, loadChunk: loadChunk,
            evaluate: { try MLX.checkedEval($0) }, checkpoint: {
                for stream in streams { try stream.experimentalSynchronize() }
            })
    }

    /// Test-only injection of evaluator and checkpoint. Production research
    /// calls must use the typed owned-stream check in `update`.
    static func perform(model: Module, descriptors: [Descriptor], arena: ExperimentalRequestArena,
        stream: MLX.Stream, maxChunkBytes: UInt64, shouldCancel: () -> Bool = { false },
        loadChunk: ([String]) throws -> [String: MLXArray],
        evaluate: ([MLXArray]) throws -> Void, checkpoint: () throws -> Void) throws {
        func checkCancellation() throws {
            if shouldCancel() || Task<Never, Never>.isCancelled { throw CancellationError() }
        }
        try checkCancellation()
        guard maxChunkBytes > 0, !descriptors.isEmpty else { throw UpdateError("empty update or budget") }
        var expected: [String: [Int]] = [:]
        for (key, value) in model.parameters().flattened() {
            guard expected.updateValue(value.shape, forKey: key) == nil else {
                throw UpdateError("duplicate model parameter: \(key)")
            }
        }
        var seen = Set<String>()
        var entries: [(Descriptor, UInt64)] = []
        for descriptor in descriptors.sorted(by: { $0.key < $1.key }) {
            try checkCancellation()
            guard seen.insert(descriptor.key).inserted, expected[descriptor.key] == descriptor.shape,
                !AffineStreamingExpertCatalog.isRoutedTensorKey(descriptor.key, layout: .qwen35),
                [.bfloat16, .float16, .float32].contains(descriptor.dtype)
            else { throw UpdateError("invalid, duplicate or excluded parameter: \(descriptor.key)") }
            var count: UInt64 = 1
            for dimension in descriptor.shape {
                guard dimension > 0 else { throw UpdateError("invalid parameter shape: \(descriptor.key)") }
                let product = count.multipliedReportingOverflow(by: UInt64(dimension))
                guard !product.overflow else { throw UpdateError("parameter size overflow") }
                count = product.partialValue
            }
            let cost: UInt64 = descriptor.dtype == .bfloat16 ? 2 : (descriptor.dtype == .float16 ? 4 : 6)
            let bytes = count.multipliedReportingOverflow(by: cost)
            guard !bytes.overflow, bytes.partialValue <= maxChunkBytes else {
                throw UpdateError("parameter exceeds chunk budget: \(descriptor.key)")
            }
            entries.append((descriptor, bytes.partialValue))
        }
        // Register once before graph construction. A failed evaluation, drain
        // or update leaves model and current graphs in the quarantined arena.
        try arena.retain(model)
        let slot = Slot()
        try arena.retain(slot)
        var start = 0
        while start < entries.count {
            try checkCancellation()
            var end = start, used: UInt64 = 0
            while end < entries.count && entries[end].1 <= maxChunkBytes - used {
                used += entries[end].1; end += 1
            }
            let chunk = Array(entries[start..<end])
            let keys = chunk.map { $0.0.key }
            let loaded = try loadChunk(keys)
            try checkCancellation()
            guard Set(loaded.keys) == Set(keys) else { throw UpdateError("loaded chunk key mismatch") }
            slot.sources = keys.compactMap { loaded[$0] }
            for (descriptor, _) in chunk {
                guard let array = loaded[descriptor.key], array.shape == descriptor.shape,
                    array.dtype == descriptor.dtype else { throw UpdateError("loaded descriptor mismatch: \(descriptor.key)") }
            }
            try MLX.withError { error in
                var values: [(String, MLXArray)] = []
                for (descriptor, _) in chunk {
                    try checkCancellation()
                    let input = loaded[descriptor.key]!
                    let result = descriptor.dtype == .bfloat16 ? input : input.asType(.bfloat16, stream: .stream(stream))
                    slot.outputs.append(result)
                    try error.check()
                    values.append((descriptor.key, result))
                }
                try checkCancellation()
                try evaluate(slot.outputs)
                try error.check()
                try checkCancellation()
                // Retire only after every owned stream has synchronized.
                try checkpoint()
                try checkCancellation()
                // A single leaf per update makes the MLX error check run
                // before another parameter can be mutated after a C error.
                for (key, value) in values {
                    try checkCancellation()
                    try model.update(parameters: ModuleParameters.unflattened([(key, value)]),
                        verify: [.noUnusedKeys, .shapeMismatch])
                    try error.check()
                }
            }
            slot.clear()
            try checkCancellation()
            start = end
        }
        try checkCancellation()
    }
}
