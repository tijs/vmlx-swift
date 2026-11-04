// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXLMCommon

struct LoadExclusionHeaderTests {
    final class ExcludingModel: Module, LanguageModel, SafetensorsLoadKeyExcluding {
        var observedKeys: [String] = []
        var kvHeads: [Int] { [1] }
        func excludeFromGenericSafetensorsLoad(key: String) -> Bool {
            observedKeys.append(key)
            return key == "auxiliary"
        }
        func newCache(parameters: GenerateParameters?) -> [KVCache] { [KVCacheSimple()] }
        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            .tokens(input.text)
        }
        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray { inputs }
    }

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

    @Test("actual loader rejects malformed headers and excludes valid auxiliary shards")
    func loaderExclusion() throws {
        try MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let shard = directory.appendingPathComponent("model.safetensors")
            try Data([4, 0, 0, 0, 0, 0, 0, 0, 123]).write(to: shard)
            let malformed = ExcludingModel()
            #expect(throws: (any Error).self) { try loadWeights(modelDirectory: directory, model: malformed) }
            #expect(malformed.observedKeys.isEmpty)
            try MLX.save(arrays: ["auxiliary": MLXArray([Float(1)])], url: shard)
            let valid = ExcludingModel()
            try loadWeights(modelDirectory: directory, model: valid)
            #expect(valid.observedKeys.contains("auxiliary"))
            #expect(valid.parameters().flattened().isEmpty)
        }
    }
}
