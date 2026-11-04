// Copyright © 2026 Apple Inc.
import Foundation
import MLX

/// Standalone research helper; not used by the generic loader or serving.
/// Callers must own and quiesce all exact streams, stop all submitting workers
/// before returning, retain the enclosing arena through drain, and prevent
/// returned array aliases from outliving that ownership. Its byte limits cover
/// nominal source/output storage, not Metal caches, padding, graphs or RSS.
enum ExperimentalChunkConversion {
    struct ConversionError: Error, CustomStringConvertible {
        let description: String
        init(_ description: String) { self.description = description }
    }
    private final class Inputs {
        let arrays: [MLXArray]
        init(_ arrays: [MLXArray]) { self.arrays = arrays }
    }
    private final class ChunkGraphs {
        let inputs: [MLXArray]
        var outputs: [MLXArray] = []
        init(_ inputs: [MLXArray]) { self.inputs = inputs }
    }

    static func convert(_ input: [String: MLXArray], arena: ExperimentalRequestArena,
        streams: [MLX.Stream], conversionStream: MLX.Stream,
        maxChunkBytes: UInt64, maxTotalBytes: UInt64) throws -> [String: MLXArray] {
        try ExperimentalOwnedStreamRequest.validateMappingStreams(arena: arena, streams: streams)
        guard streams.contains(conversionStream) else { throw ConversionError("conversion stream not owned") }
        return try perform(input, arena: arena, stream: conversionStream,
            maxChunkBytes: maxChunkBytes, maxTotalBytes: maxTotalBytes,
            evaluate: { try MLX.checkedEval($0) })
    }

    // Injectable evaluator is internal research-test support, not an admission
    // path. Production research callers must use convert's owned-stream check.
    static func perform(_ input: [String: MLXArray], arena: ExperimentalRequestArena,
        stream: MLX.Stream, maxChunkBytes: UInt64, maxTotalBytes: UInt64,
        evaluate: ([MLXArray]) throws -> Void) throws -> [String: MLXArray] {
        guard maxChunkBytes > 0, maxTotalBytes > 0 else { throw ConversionError("invalid byte budget") }
        var entries: [(String, MLXArray, UInt64)] = []
        var total: UInt64 = 0
        for key in input.keys.sorted() {
            let array = input[key]!
            let width: UInt64
            if array.dtype == .float16 { width = 2 }
            else if array.dtype == .float32 { width = 4 }
            else { throw ConversionError("unsupported input dtype: \(key)") }
            var count: UInt64 = 1
            for dimension in array.shape {
                guard dimension > 0 else { throw ConversionError("unsupported empty dimension: \(key)") }
                let product = count.multipliedReportingOverflow(by: UInt64(dimension))
                guard !product.overflow else { throw ConversionError("element count overflow: \(key)") }
                count = product.partialValue
            }
            let bytes = count.multipliedReportingOverflow(by: width + 2)
            guard !bytes.overflow, bytes.partialValue <= maxChunkBytes,
                total <= maxTotalBytes, bytes.partialValue <= maxTotalBytes - total else {
                throw ConversionError("conversion byte budget exceeded: \(key)")
            }
            total += bytes.partialValue
            entries.append((key, array, bytes.partialValue))
        }
        try arena.retain(Inputs(entries.map { $0.1 }))
        var result: [String: MLXArray] = [:]
        var start = 0
        while start < entries.count {
            try Task<Never, Never>.checkCancellation()
            var end = start, chunkBytes: UInt64 = 0
            while end < entries.count && entries[end].2 <= maxChunkBytes - chunkBytes {
                chunkBytes += entries[end].2
                end += 1
            }
            let chunk = Array(entries[start..<end])
            let graphs = ChunkGraphs(chunk.map { $0.1 })
            // The holder is already arena-owned before graph construction; its
            // output references are installed before evaluation can submit work.
            try arena.retain(graphs)
            try MLX.withError { error in
                for (key, array, _) in chunk {
                    let converted = array.asType(.bfloat16, stream: .stream(stream))
                    graphs.outputs.append(converted)
                    try error.check()
                    result[key] = converted
                }
                try evaluate(graphs.outputs)
                try error.check()
            }
            try Task<Never, Never>.checkCancellation()
            start = end
        }
        return result
    }
}
