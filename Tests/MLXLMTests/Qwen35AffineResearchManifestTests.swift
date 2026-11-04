// Copyright © 2026 Apple Inc.

import Foundation
import Testing
@testable import MLXLMCommon

struct Qwen35AffineResearchManifestTests {
    private func fixture() throws -> (URL, URL, [String: String]) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let shard = root.appendingPathComponent("model.safetensors")
        var header: [String: Any] = [:]
        var index: [String: String] = [:]
        var offset = 0
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            for suffix in ["weight", "scales", "biases"] {
                let key = "language_model.model.layers.0.mlp.switch_mlp.\(projection).\(suffix)"
                let size = suffix == "weight" ? 2048 : 128
                header[key] = ["dtype": suffix == "weight" ? "U32" : "BF16",
                               "shape": suffix == "weight" ? [1, 64, 8] : [1, 64, 1],
                               "data_offsets": [offset, offset + size]] as [String: Any]
                index[key] = shard.lastPathComponent
                offset += size
            }
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        json.append(Data(repeating: 32, count: (8 - (json.count + 8) % 8) % 8))
        var length = UInt64(json.count).littleEndian
        var data = Data(bytes: &length, count: 8)
        data.append(json)
        data.append(Data(repeating: 0, count: offset))
        try data.write(to: shard)
        try JSONSerialization.data(withJSONObject: ["weight_map": index])
            .write(to: root.appendingPathComponent("model.safetensors.index.json"))
        return (root, shard, index)
    }

    private func validate(_ root: URL, _ shards: [URL]) throws {
        _ = try Qwen35AffineResearchManifest.validate(modelDirectory: root, selectedShards: shards,
            layerCount: 1, numExperts: 1, inputDims: 64, hiddenDims: 64)
    }

    @Test("complete indexed catalog validates; duplicate selection is rejected")
    func complete() throws {
        let (root, shard, _) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        try validate(root, [shard])
        #expect(throws: (any Error).self) { try validate(root, [shard, shard]) }
    }

    @Test("index disagreement and duplicate JSON keys fail closed")
    func invalidIndex() throws {
        let (root, shard, index) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        var changed = index
        changed.removeValue(forKey: changed.keys.sorted()[0])
        let url = root.appendingPathComponent("model.safetensors.index.json")
        try JSONSerialization.data(withJSONObject: ["weight_map": changed]).write(to: url)
        #expect(throws: (any Error).self) { try validate(root, [shard]) }
        try Data("{\"weight_map\":{},\"weight_map\":{}}".utf8).write(to: url)
        #expect(throws: (any Error).self) { try validate(root, [shard]) }
    }

    @Test("unselected shard and escaped duplicate index member are rejected")
    func selectedAndEscapedDuplicate() throws {
        let (root, shard, _) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(throws: (any Error).self) { try validate(root, []) }
        let url = root.appendingPathComponent("model.safetensors.index.json")
        try Data("{\"weight_map\":{},\"weight_\\u006dap\":{}}".utf8).write(to: url)
        #expect(throws: (any Error).self) { try validate(root, [shard]) }
    }

    @Test("all routed and non-routed descriptors reject unsafe offsets")
    func unsafeDescriptors() throws {
        for routed in [true, false] {
            for offsets in ([[-1, 0], [2, 1], [0, UInt64.max], [0.5, 1], [false, 1]] as [[Any]]) {
                let (root, shard, index) = try fixture()
                defer { try? FileManager.default.removeItem(at: root) }
                let original = try Data(contentsOf: shard)
                let headerLength = original.prefix(8).withUnsafeBytes {
                    $0.loadUnaligned(as: UInt64.self).littleEndian
                }
                let headerEnd = 8 + Int(headerLength)
                var header = try JSONSerialization.jsonObject(with: original.subdata(in: 8..<headerEnd)) as! [String: Any]
                var updatedIndex = index
                let key = routed ? index.keys.sorted()[0] : "language_model.model.embed_tokens.weight"
                var descriptor = (header[key] as? [String: Any]) ?? ["dtype": "U32", "shape": [1]]
                descriptor["data_offsets"] = offsets
                header[key] = descriptor
                updatedIndex[key] = shard.lastPathComponent
                var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
                json.append(Data(repeating: 32, count: (8 - (json.count + 8) % 8) % 8))
                var length = UInt64(json.count).littleEndian
                var rewritten = Data(bytes: &length, count: 8)
                rewritten.append(json)
                rewritten.append(original.suffix(from: headerEnd))
                try rewritten.write(to: shard)
                try JSONSerialization.data(withJSONObject: ["weight_map": updatedIndex])
                    .write(to: root.appendingPathComponent("model.safetensors.index.json"))
                #expect(throws: (any Error).self) { try validate(root, [shard]) }
            }
        }
    }

    @Test("index symlinks may not escape the checkpoint")
    func outsideIndex() throws {
        let (root, shard, _) = try fixture()
        let outside = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: outside)
        }
        try Data("{}".utf8).write(to: outside)
        let index = root.appendingPathComponent("model.safetensors.index.json")
        try FileManager.default.removeItem(at: index)
        try FileManager.default.createSymbolicLink(at: index, withDestinationURL: outside)
        #expect(throws: (any Error).self) { try validate(root, [shard]) }
    }

    @Test("tensor payload coverage rejects aliases, gaps and trailing bytes")
    func payloadCoverage() throws {
        for mode in ["routedAlias", "otherAlias", "gap", "trailing", "empty"] {
            let (root, shard, index) = try fixture()
            defer { try? FileManager.default.removeItem(at: root) }
            let data = try Data(contentsOf: shard)
            let length = data.prefix(8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
            let end = 8 + Int(length)
            var header = try JSONSerialization.jsonObject(with: data.subdata(in: 8..<end)) as! [String: Any]
            var updatedIndex = index
            let ordered = header.keys.sorted {
                let a = (header[$0] as! [String: Any])["data_offsets"] as! [Int]
                let b = (header[$1] as! [String: Any])["data_offsets"] as! [Int]
                return a[0] < b[0]
            }
            let first = header[ordered[0]] as! [String: Any]
            if mode == "routedAlias" {
                var second = header[ordered[1]] as! [String: Any]
                second["data_offsets"] = first["data_offsets"]
                header[ordered[1]] = second
            } else if mode == "otherAlias" || mode == "empty" {
                let key = "language_model.model.embed_tokens.weight"
                header[key] = ["dtype": "U32", "shape": [0],
                               "data_offsets": mode == "empty" ? [0, 0] as Any : first["data_offsets"]!] as [String: Any]
                updatedIndex[key] = shard.lastPathComponent
            } else if mode == "gap" {
                var second = header[ordered[1]] as! [String: Any]
                let old = second["data_offsets"] as! [Int]
                second["data_offsets"] = [old[0] + 1, old[1]]
                header[ordered[1]] = second
            }
            var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
            json.append(Data(repeating: 32, count: (8 - (json.count + 8) % 8) % 8))
            var headerLength = UInt64(json.count).littleEndian
            var rewritten = Data(bytes: &headerLength, count: 8)
            rewritten.append(json)
            rewritten.append(data.suffix(from: end))
            if mode == "trailing" { rewritten.append(0) }
            try rewritten.write(to: shard)
            try JSONSerialization.data(withJSONObject: ["weight_map": updatedIndex])
                .write(to: root.appendingPathComponent("model.safetensors.index.json"))
            if mode == "empty" {
                try validate(root, [shard])
            } else {
                #expect(throws: (any Error).self) { try validate(root, [shard]) }
            }
        }
    }

    @Test("non-routed descriptors require supported dtype and exact shape bytes")
    func nonRoutedShapeBytes() throws {
        for (dtype, shape, shouldPass) in [
            ("F32", [0], true), ("F32", [1], false),
            ("UNKNOWN", [0], false), ("U64", [Int.max, 2], false),
        ] {
            let (root, shard, index) = try fixture()
            defer { try? FileManager.default.removeItem(at: root) }
            let data = try Data(contentsOf: shard)
            let length = data.prefix(8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
            let end = 8 + Int(length)
            var header = try #require(JSONSerialization.jsonObject(with: data.subdata(in: 8..<end)) as? [String: Any])
            let key = "language_model.model.embed_tokens.weight"
            header[key] = ["dtype": dtype, "shape": shape, "data_offsets": [0, 0]] as [String: Any]
            var updatedIndex = index
            updatedIndex[key] = shard.lastPathComponent
            var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
            json.append(Data(repeating: 32, count: (8 - (json.count + 8) % 8) % 8))
            var headerLength = UInt64(json.count).littleEndian
            var rewritten = Data(bytes: &headerLength, count: 8)
            rewritten.append(json)
            rewritten.append(data.suffix(from: end))
            try rewritten.write(to: shard)
            try JSONSerialization.data(withJSONObject: ["weight_map": updatedIndex])
                .write(to: root.appendingPathComponent("model.safetensors.index.json"))
            if shouldPass { try validate(root, [shard]) }
            else { #expect(throws: (any Error).self) { try validate(root, [shard]) } }
        }
    }
}
