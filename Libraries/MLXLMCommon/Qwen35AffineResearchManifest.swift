// Copyright © 2026 Apple Inc.

import Foundation
import CoreFoundation

/// Strict, research-only source validation. It does not load a model or make
/// mutable checkpoint files safe to map; callers must keep the files immutable.
public enum Qwen35AffineResearchManifest {
    public struct ValidationError: Error, CustomStringConvertible {
        public let description: String
        init(_ description: String) { self.description = description }
    }

    /// Validates the exact selected shards and returns a separately configured
    /// catalog. No caller-owned catalog is mutated on failure.
    public static func validate(
        modelDirectory: URL, selectedShards: [URL], layerCount: Int,
        numExperts: Int, inputDims: Int, hiddenDims: Int
    ) throws -> AffineStreamingExpertCatalog {
        guard layerCount > 0, numExperts > 0, inputDims > 0, hiddenDims > 0 else {
            throw ValidationError("invalid geometry")
        }
        let root = modelDirectory.resolvingSymlinksInPath().standardizedFileURL
        let indexURL = root.appendingPathComponent("model.safetensors.index.json")
            .resolvingSymlinksInPath().standardizedFileURL
        guard indexURL.path.hasPrefix(root.path + "/") else {
            throw ValidationError("index escapes model directory")
        }
        let indexValues = try indexURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard indexValues.isRegularFile == true, let size = indexValues.fileSize,
            size >= 0, size <= 64 * 1024 * 1024 else {
            throw ValidationError("index must be a bounded regular file")
        }
        let indexData = try Data(contentsOf: indexURL)
        guard indexData.count <= 64 * 1024 * 1024 else {
            throw ValidationError("index exceeds size limit")
        }
        let index = try object(indexData, label: "index")
        guard let weightMap = index["weight_map"] as? [String: String] else {
            throw ValidationError("invalid weight_map")
        }
        var selected: [String: URL] = [:]
        for shard in selectedShards {
            let url = shard.resolvingSymlinksInPath().standardizedFileURL
            guard url.path.hasPrefix(root.path + "/") else {
                throw ValidationError("selected shard escapes model directory")
            }
            guard try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else {
                throw ValidationError("selected shard is not a regular file")
            }
            let name = String(url.path.dropFirst(root.path.count + 1))
            guard selected.updateValue(url, forKey: name) == nil else {
                throw ValidationError("duplicate selected shard: \(name)")
            }
        }
        guard !selected.isEmpty, Set(weightMap.values) == Set(selected.keys) else {
            throw ValidationError("selected shards do not match index")
        }
        var actual: [String: String] = [:]
        for (name, url) in selected {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
                throw ValidationError("truncated shard header: \(name)")
            }
            var length: UInt64 = 0
            for (i, byte) in prefix.enumerated() { length |= UInt64(byte) << UInt64(8 * i) }
            guard length > 0, length <= 64 * 1024 * 1024,
                let bytes = try handle.read(upToCount: Int(length)), bytes.count == Int(length)
            else { throw ValidationError("invalid shard header length: \(name)") }
            let header = try object(bytes, label: name)
            let fileSize = try handle.seekToEnd()
            let dataStart = UInt64(8) + length // length is bounded above
            guard fileSize >= dataStart else { throw ValidationError("truncated shard: \(name)") }
            var ranges: [(UInt64, UInt64)] = []
            for (key, value) in header where key != "__metadata__" {
                guard let descriptor = value as? [String: Any],
                    descriptor["dtype"] is String,
                    let shape = descriptor["shape"] as? [Any],
                    shape.allSatisfy({ unsignedInteger($0).map { $0 <= UInt64(Int.max) } ?? false }),
                    let offsets = descriptor["data_offsets"] as? [Any], offsets.count == 2,
                    let start = unsignedInteger(offsets[0]), let end = unsignedInteger(offsets[1]),
                    start <= end, end <= fileSize - dataStart
                else { throw ValidationError("invalid tensor descriptor: \(name): \(key)") }
                ranges.append((start, end))
                guard actual.updateValue(name, forKey: key) == nil else {
                    throw ValidationError("duplicate tensor across shards: \(key)")
                }
            }
            ranges.sort { $0.0 == $1.0 ? $0.1 < $1.1 : $0.0 < $1.0 }
            var cursor: UInt64 = 0
            for (start, end) in ranges {
                guard start == cursor else {
                    throw ValidationError("overlapping or gapped tensor data: \(name)")
                }
                cursor = end
            }
            guard cursor == fileSize - dataStart else {
                throw ValidationError("unclaimed shard payload: \(name)")
            }
        }
        guard actual == weightMap else {
            throw ValidationError("shard headers and index disagree")
        }
        var expected = Set<String>()
        for layer in 0..<layerCount {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    expected.insert("language_model.model.layers.\(layer).mlp.switch_mlp.\(projection).\(suffix)")
                }
            }
        }
        // Reject unsupported prefixes, layers, suffixes, and aliases, rather
        // than guessing how they would be renamed by the model sanitizer.
        let routed = Set(actual.keys.filter { $0.split(separator: ".").contains("switch_mlp") })
        guard routed == expected else { throw ValidationError("routed tensor coverage or naming mismatch") }
        let catalog = AffineStreamingExpertCatalog(
            numExperts: numExperts, inputDims: inputDims, hiddenDims: hiddenDims, layout: .qwen35)
        try catalog.configure(modelDirectory: root, layerCount: layerCount)
        return catalog
    }

    private static func unsignedInteger(_ value: Any) -> UInt64? {
        guard let number = value as? NSNumber,
            CFGetTypeID(number) != CFBooleanGetTypeID(),
            !["f", "d"].contains(String(cString: number.objCType))
        else { return nil }
        return UInt64(number.stringValue)
    }

    // JSONSerialization collapses duplicate keys. Validate duplicate object
    // members first, including escaped-equivalent keys, at every nesting level.
    private static func object(_ data: Data, label: String) throws -> [String: Any] {
        guard let value = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ValidationError("invalid JSON object: \(label)")
        }
        let bytes = Array(data)
        var objectKeys: [Set<String>?] = []
        var i = 0
        while i < bytes.count {
            switch bytes[i] {
            case 123: objectKeys.append([]); i += 1
            case 91: objectKeys.append(nil); i += 1
            case 125, 93: _ = objectKeys.popLast(); i += 1
            case 34:
                let start = i
                i += 1
                while i < bytes.count {
                    if bytes[i] == 92 { i += 2; continue }
                    if bytes[i] == 34 { i += 1; break }
                    i += 1
                }
                var next = i
                while next < bytes.count && [9, 10, 13, 32].contains(bytes[next]) { next += 1 }
                if next < bytes.count, bytes[next] == 58, !objectKeys.isEmpty,
                    objectKeys[objectKeys.count - 1] != nil {
                    let raw = Data(bytes[start..<i])
                    var wrapped = Data([91]); wrapped.append(raw); wrapped.append(93)
                    guard let key = (try JSONSerialization.jsonObject(with: wrapped) as? [String])?.first else {
                        throw ValidationError("invalid object key: \(label)")
                    }
                    if !objectKeys[objectKeys.count - 1]!.insert(key).inserted {
                        throw ValidationError("duplicate object key in \(label): \(key)")
                    }
                }
            default: i += 1
            }
        }
        return value
    }
}
