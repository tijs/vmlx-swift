// Copyright © 2026 Apple Inc.

import Foundation
import Testing
@testable import MLXLMCommon

struct LoadExclusionHeaderTests {
    @Test("required pre-map exclusion rejects unreadable and malformed headers")
    func failsClosed() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("model.safetensors")
        #expect(throws: (any Error).self) { try baseLoadHeaderNames(url, requiresExclusion: true) }
        for data in [Data(), Data(repeating: 0, count: 8), Data([4, 0, 0, 0, 0, 0, 0, 0, 123])] {
            try data.write(to: url)
            #expect(throws: (any Error).self) { try baseLoadHeaderNames(url, requiresExclusion: true) }
            #expect(try baseLoadHeaderNames(url, requiresExclusion: false).isEmpty)
        }
        func writeHeader(_ json: String) throws {
            let header = Data(json.utf8)
            var length = UInt64(header.count).littleEndian
            var data = Data(bytes: &length, count: 8)
            data.append(header)
            try data.write(to: url)
        }
        for empty in ["{}", "{\"__metadata__\":{\"format\":\"mlx\"}}"] {
            try writeHeader(empty)
            #expect(try baseLoadHeaderNames(url, requiresExclusion: true).isEmpty)
        }
        try writeHeader("[]")
        #expect(throws: (any Error).self) { try baseLoadHeaderNames(url, requiresExclusion: true) }
        // This is a name reader, not tensor descriptor/schema validation.
        try writeHeader("{\"layer.weight\":{},\"__metadata__\":{}}")
        #expect(try baseLoadHeaderNames(url, requiresExclusion: true) == ["layer.weight"])
    }
}
