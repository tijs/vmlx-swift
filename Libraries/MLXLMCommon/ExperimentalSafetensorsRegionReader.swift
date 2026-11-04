// Copyright © 2026 Apple Inc.
import Cmlx
import Foundation
import MLX
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

/// Exact-region resident tensor mapping for research, not a serving API.
/// Inspection must come from the same immutable snapshot and the caller must
/// retain that snapshot, prevent path replacement/truncation, own all streams,
/// and prevent returned aliases escaping the enclosing drained arena. The C
/// operation creates a file-backed Metal buffer but does not submit GPU work.
/// The cap covers only this region's page-rounded mapping, not total memory.
enum ExperimentalSafetensorsRegionReader {
    struct ReadError: Error, CustomStringConvertible {
        let description: String
        init(_ description: String) { self.description = description }
    }

    static func read(key: String, inspection: Qwen35AffineResearchManifest.Inspection,
        arena: ExperimentalRequestArena, maxMappedBytes: UInt64) throws -> MLXArray {
        try Task<Never, Never>.checkCancellation()
        let parts = key.split(separator: ".", omittingEmptySubsequences: false)
        guard !parts.contains("switch_mlp"), !parts.contains("experts"),
            let descriptor = inspection.tensors[key] else { throw ReadError("excluded or unknown resident tensor: \(key)") }
        let dtype: DType
        switch descriptor.dtype {
        case "U32": dtype = .uint32
        case "BF16": dtype = .bfloat16
        case "F16": dtype = .float16
        case "F32": dtype = .float32
        default: throw ReadError("unsupported tensor dtype: \(key)")
        }
        guard descriptor.byteLength > 0, descriptor.byteLength <= UInt64(Int.max),
            descriptor.shape.count <= Int(Int32.max),
            descriptor.shape.allSatisfy({ $0 >= 0 && $0 <= Int(Int32.max) }),
            descriptor.byteOffset % UInt64(dtype.size) == 0 else { throw ReadError("invalid region geometry: \(key)") }
        var count: UInt64 = 1
        for dimension in descriptor.shape {
            let product = count.multipliedReportingOverflow(by: UInt64(dimension))
            guard !product.overflow else { throw ReadError("tensor shape overflow: \(key)") }
            count = product.partialValue
        }
        let bytes = count.multipliedReportingOverflow(by: UInt64(dtype.size))
        guard !bytes.overflow, bytes.partialValue == descriptor.byteLength else { throw ReadError("tensor byte mismatch: \(key)") }
        let page = UInt64(getpagesize())
        let span = (descriptor.byteOffset % page).addingReportingOverflow(descriptor.byteLength)
        guard !span.overflow else { throw ReadError("mapping span overflow: \(key)") }
        let padding = (page - span.partialValue % page) % page
        let mapped = span.partialValue.addingReportingOverflow(padding)
        guard !mapped.overflow, mapped.partialValue <= maxMappedBytes else { throw ReadError("mapping budget exceeded: \(key)") }
        let root = inspection.modelDirectory.resolvingSymlinksInPath().standardizedFileURL
        let file = root.appendingPathComponent(descriptor.shard).resolvingSymlinksInPath().standardizedFileURL
        guard file.path.hasPrefix(root.path + "/") else { throw ReadError("shard escaped checkpoint root") }
        var shape = descriptor.shape.map(Int32.init)
        var result = mlx_array_new()
        let status: Int32
        do {
            status = try MLX.withError {
                file.withUnsafeFileSystemRepresentation { path -> Int32 in
                    guard let path else { return 1 }
                    return mlx_array_new_mmap_file_region(&result, path, descriptor.byteOffset,
                        Int(descriptor.byteLength), &shape, Int32(shape.count), dtype.cmlxDtype)
                }
            }
        } catch {
            mlx_array_free(result)
            throw error
        }
        guard status == 0 else { mlx_array_free(result); throw ReadError("mmap failed with status \(status): \(key)") }
        let array = MLXArray(result)
        // No graph or evaluation has been submitted; retain before returning.
        try arena.retain(array)
        return array
    }
}
