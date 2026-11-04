// Copyright © 2026 Apple Inc.

import Foundation
import Testing
@testable import MLXLMCommon

struct JangPressShardBoundsTests {
    @Test("malformed header sizes and tensor offsets throw rather than overflow")
    func bounds() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let shard = directory.appendingPathComponent("model.safetensors")
        var huge = UInt64.max.littleEndian
        try Data(bytes: &huge, count: 8).write(to: shard)
        #expect(throws: JangPressShardError.self) { _ = try JangPressShard(path: shard) }
        for offsets in ["[4, 2]", "[0, 18446744073709551615]", "[18446744073709551615, 18446744073709551615]"] {
            let header = Data("{\"tensor\":{\"dtype\":\"U32\",\"shape\":[1],\"data_offsets\":\(offsets)}}".utf8)
            var length = UInt64(header.count).littleEndian
            var file = Data(bytes: &length, count: 8)
            file.append(header)
            file.append(Data(repeating: 0, count: 4))
            try file.write(to: shard)
            #expect(throws: JangPressShardError.self) { _ = try JangPressShard(path: shard) }
        }
        for offsets in ["[0.5, 4]", "[false, 4]", "[-1, 4]"] {
            let header = Data("{\"tensor\":{\"dtype\":\"U32\",\"shape\":[1],\"data_offsets\":\(offsets)}}".utf8)
            var length = UInt64(header.count).littleEndian
            var file = Data(bytes: &length, count: 8)
            file.append(header)
            file.append(Data(repeating: 0, count: 4))
            try file.write(to: shard)
            let parsed = try JangPressShard(path: shard)
            #expect(parsed.descriptor(for: "tensor") == nil)
        }
    }
}
