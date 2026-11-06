import Foundation
import XCTest

@testable import MLXLMCommon

final class JANGHHeaderAdapterTests: XCTestCase {
    private let shard = "model.safetensors"
    private let index = "model.safetensors.index.json"
    private let header = #"{"weight":{"dtype":"U32","shape":[2],"data_offsets":[0,8]},"__metadata__":{"note":"fixture"}}"#

    private func withDirectory(_ body: (URL) throws -> Void) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try body(directory)
    }

    private func write(_ directory: URL, header: String? = nil, indexText: String? = nil) throws {
        let raw = Data((header ?? self.header).utf8)
        var length = UInt64(raw.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(raw)
        data.append(Data(repeating: 0xAB, count: 8))
        try data.write(to: directory.appendingPathComponent(shard), options: .atomic)
        let text = indexText ?? #"{"weight_map":{"weight":"model.safetensors"}}"#
        try Data(text.utf8).write(to: directory.appendingPathComponent(index), options: .atomic)
    }

    func testReadsOnlyMetadataAndRecordsReplacementIdentity() throws {
        try withDirectory { directory in
            try write(directory)
            let snapshot = try JANGHHeaderAdapter.read(directory: directory, indexName: index)
            XCTAssertEqual(snapshot.weightMap, ["weight": shard])
            XCTAssertEqual(snapshot.headerData[shard], Data(header.utf8))
            XCTAssertEqual(snapshot.shards[shard]?.tensors["weight"]?.shape, [2])
            XCTAssertEqual(snapshot.shards[shard]?.tensors.count, 1)
            XCTAssertEqual(snapshot.identities[shard]?.bytes, 8 + header.utf8.count + 8)
            // Atomic replacement with identical contents has a distinct open-file identity.
            try write(directory)
            let replacement = try JANGHHeaderAdapter.read(directory: directory, indexName: index)
            XCTAssertNotEqual(snapshot.identities[shard], replacement.identities[shard])
            XCTAssertEqual(snapshot.headerData, replacement.headerData)
        }
    }

    func testRejectsDuplicateAndMalformedMetadata() throws {
        try withDirectory { directory in
            for invalid in [
                #"{"weight":{"dtype":"U32","shape":[2],"data_offsets":[0,8]},"weight":{}}"#,
                #"{"weight":{"dtype":"U32","dtype":"F16","shape":[2],"data_offsets":[0,8]}}"#,
                #"{"weight":{"dtype":"U32","shape":[2],"data_offsets":[0]}}"#,
                #"{"weight":{"dtype":"U32","shape":[true],"data_offsets":[0,8]}}"#,
                #"{"weight":{"dtype":"U32","shape":[2.5],"data_offsets":[0,8]}}"#,
                #"{"weight":{},"\u0077eight":{}}"#,
            ] {
                try write(directory, header: invalid)
                XCTAssertThrowsError(try JANGHHeaderAdapter.read(directory: directory, indexName: index))
            }
            try write(directory, indexText: #"{"weight_map":{"weight":"model.safetensors","weight":"other.safetensors"}}"#)
            XCTAssertThrowsError(try JANGHHeaderAdapter.read(directory: directory, indexName: index))
        }
    }

    func testRejectsMetadataLengthAndAggregateBounds() throws {
        try withDirectory { directory in
            try write(directory)
            XCTAssertThrowsError(try JANGHHeaderAdapter.read(directory: directory, indexName: index,
                                                            maximumMetadataBytes: 8))
            let size = max(header.utf8.count, 64)
            XCTAssertThrowsError(try JANGHHeaderAdapter.read(directory: directory, indexName: index,
                                                            maximumMetadataBytes: size,
                                                            maximumTotalMetadataBytes: size))
            var oversized = UInt64.max.littleEndian
            let data = withUnsafeBytes(of: &oversized) { Data($0) }
            try data.write(to: directory.appendingPathComponent(shard))
            XCTAssertThrowsError(try JANGHHeaderAdapter.read(directory: directory, indexName: index))
            try Data([0, 0, 0]).write(to: directory.appendingPathComponent(shard))
            XCTAssertThrowsError(try JANGHHeaderAdapter.read(directory: directory, indexName: index))
        }
    }

    func testRejectsSymlinkMissingAndTraversalFiles() throws {
        try withDirectory { directory in
            try write(directory)
            try FileManager.default.moveItem(at: directory.appendingPathComponent(shard),
                                             to: directory.appendingPathComponent("target.safetensors"))
            try FileManager.default.createSymbolicLink(atPath: directory.appendingPathComponent(shard).path,
                                                       withDestinationPath: "target.safetensors")
            XCTAssertThrowsError(try JANGHHeaderAdapter.read(directory: directory, indexName: index))
            try FileManager.default.removeItem(at: directory.appendingPathComponent(shard))
            XCTAssertThrowsError(try JANGHHeaderAdapter.read(directory: directory, indexName: index))
            for name in ["../model.safetensors.index.json", "/model.safetensors.index.json", "bad\\index"] {
                XCTAssertThrowsError(try JANGHHeaderAdapter.read(directory: directory, indexName: name))
            }
            try write(directory, indexText: #"{"weight_map":{"weight":"../outside.safetensors"}}"#)
            XCTAssertThrowsError(try JANGHHeaderAdapter.read(directory: directory, indexName: index))
        }
    }

    func testAllowsSymlinkBundleRootButNotSymlinkIndex() throws {
        try withDirectory { directory in
            let actual = directory.appendingPathComponent("actual")
            try FileManager.default.createDirectory(at: actual, withIntermediateDirectories: true)
            try write(actual)
            let link = directory.appendingPathComponent("bundle")
            try FileManager.default.createSymbolicLink(at: link, withDestinationURL: actual)
            let snapshot = try JANGHHeaderAdapter.read(directory: link, indexName: index)
            XCTAssertEqual(snapshot.weightMap.count, 1)
            try FileManager.default.moveItem(at: actual.appendingPathComponent(index),
                                             to: actual.appendingPathComponent("index-target.json"))
            try FileManager.default.createSymbolicLink(atPath: actual.appendingPathComponent(index).path,
                                                       withDestinationPath: "index-target.json")
            XCTAssertThrowsError(try JANGHHeaderAdapter.read(directory: link, indexName: index))
        }
    }
}
