// Copyright © 2026 Apple Inc.
import Foundation

/// Bounded CPU-only norm vote for strictly inspected, immutable safetensors.
/// The caller must keep the same trusted files immutable between inspection and
/// this read; path checks do not defend against concurrent replacement/writes.
/// Float32 reduction order is not guaranteed across MLX backends: ambiguous
/// near-threshold cases fail closed rather than asserting numerical parity.
public enum Qwen35NormHeaderCheck {
    public struct CheckError: Error, CustomStringConvertible {
        public let description: String
        init(_ description: String) { self.description = description }
    }
    public static func requiresShift(
        inspection: Qwen35AffineResearchManifest.Inspection, modelDirectory: URL,
        metadataConvention: String? = nil, configConvention: String? = nil,
        declaredConvention: String? = nil, fallbackWhenNoProbe: Bool = false,
        maxProbeBytes: Int = 4 * 1024 * 1024, maxTotalBytes: Int = 32 * 1024 * 1024,
        shouldCancel: () -> Bool = { false }
    ) throws -> Bool {
        func checkCancellation() throws {
            if shouldCancel() || Task<Never, Never>.isCancelled { throw CancellationError() }
        }
        try checkCancellation()
        guard maxProbeBytes > 0, maxTotalBytes > 0 else { throw CheckError("invalid read limits") }
        if NormConventionResolver.usesPlusOne(metadataConvention)
            || NormConventionResolver.usesPlusOne(configConvention)
            || NormConventionResolver.usesPlusOne(declaredConvention) { return true }
        var below = 0, above = 0, total = 0
        let suffixes = [".input_layernorm.weight", ".post_attention_layernorm.weight"]
        for (key, desc) in inspection.tensors.sorted(by: { $0.key < $1.key }) {
            try checkCancellation()
            guard desc.shape.count == 1, suffixes.contains(where: { key.hasSuffix($0) }) else { continue }
            let width: Int
            switch desc.dtype { case "BF16", "F16": width = 2; case "F32": width = 4
            default: throw CheckError("unsupported norm dtype: \(key)") }
            guard desc.shape[0] > 0, desc.byteLength <= UInt64(maxProbeBytes),
                desc.byteLength <= UInt64(Int.max), desc.byteLength % UInt64(width) == 0,
                desc.shape[0] == Int(desc.byteLength) / width,
                Int(desc.byteLength) <= maxTotalBytes - total else { throw CheckError("invalid or oversized norm: \(key)") }
            total += Int(desc.byteLength)
            let url = modelDirectory.appendingPathComponent(desc.shard)
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            let fileSize = try handle.seekToEnd()
            guard desc.byteOffset <= fileSize, desc.byteLength <= fileSize - desc.byteOffset else {
                throw CheckError("truncated norm: \(key)")
            }
            try handle.seek(toOffset: desc.byteOffset)
            guard let bytes = try handle.read(upToCount: Int(desc.byteLength)), bytes.count == Int(desc.byteLength) else {
                throw CheckError("short norm read: \(key)")
            }
            var sum = 0.0, maxMagnitude = 0.0
            var allHalf = true
            let raw = [UInt8](bytes)
            for index in 0..<desc.shape[0] {
                if index % 4096 == 0 { try checkCancellation() }
                let i = index * width
                let value: Float
                if desc.dtype == "F32" {
                    let bits = UInt32(raw[i]) | (UInt32(raw[i + 1]) << 8) | (UInt32(raw[i + 2]) << 16) | (UInt32(raw[i + 3]) << 24)
                    value = Float(bitPattern: bits)
                } else {
                    let bits = UInt16(raw[i]) | (UInt16(raw[i + 1]) << 8)
                    value = desc.dtype == "BF16" ? Float(bitPattern: UInt32(bits) << 16) : Float(Float16(bitPattern: bits))
                }
                guard value.isFinite else { throw CheckError("non-finite norm: \(key)") }
                allHalf = allHalf && value == 0.5
                sum += Double(value)
                maxMagnitude = max(maxMagnitude, abs(Double(value)))
            }
            if allHalf { continue }
            let n = Double(desc.shape[0])
            let u = Double(Float.ulpOfOne) / 2
            guard n * u < 0.5, n * maxMagnitude < Double(Float.greatestFiniteMagnitude) else {
                throw CheckError("unsafe reduction range: \(key)")
            }
            let mean = sum / n
            // Bound arbitrary float32 addition order and final division, with
            // extra room for subnormal rounding. Do not guess near 0.5.
            let error = maxMagnitude * (n * u / (1 - n * u)) + abs(mean) * 2 * u
                + n * Double(Float.leastNonzeroMagnitude)
            guard abs(mean - 0.5) > error else { throw CheckError("ambiguous norm threshold: \(key)") }
            if mean < 0.5 { below += 1 } else { above += 1 }
        }
        try checkCancellation()
        return below + above == 0 ? fallbackWhenNoProbe : below >= above
    }
}
