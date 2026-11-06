import Foundation
import MLX
import MLXLLM
import MLXVLM
import XCTest

@testable import MLXLMCommon

final class Gemma4CacheStorageDTypeTests: XCTestCase {
    private static func models() throws -> [any LanguageModel] {
        let json = """
            {"model_type":"gemma4_text","hidden_size":64,"intermediate_size":64,
             "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
             "head_dim":32,"global_head_dim":32,"vocab_size":64,"sliding_window":4,
             "layer_types":["sliding_attention","full_attention"]}
            """
        let decoder = JSONDecoder()
        let text = Gemma4TextModel(
            try decoder.decode(
                Gemma4TextConfiguration.self, from: Data(json.utf8)))
        let vlmJSON = "{\"model_type\":\"gemma4\",\"text_config\":\(json)}"
        let vlm = Gemma4(
            try decoder.decode(
                Gemma4Configuration.self, from: Data(vlmJSON.utf8)))
        return [text, vlm]
    }

    private static func container(_ model: any LanguageModel) -> ModelContainer {
        let processor = TestInputProcessor()
        return ModelContainer(
            context: ModelContext(
                configuration: ModelConfiguration(id: "dtype-fixture"), model: model,
                processor: processor, tokenizer: processor.tokenizer))
    }

    func testBothGemmaEntrypointsScopeSyncAndAsyncCachePolicies() async throws {
        let containers = try await MLXMetalTestLock.withLock {
            try Self.models().map { Self.container($0) }
        }
        for container in containers {
            let config = CacheCoordinatorConfig(
                usePagedCache: false, enableDiskCache: false, modelKey: "same-weights")
            container.enableCaching(config: config)
            XCTAssertEqual(container.cacheCoordinator?.config.preserveStandardKVStorageDType, true)
            XCTAssertEqual(
                container.cacheCoordinator?.config.modelKey,
                "same-weights|gemma4-native-kv-dtype-v1")
            await container.enableCachingAsync(config: config)
            XCTAssertEqual(container.cacheCoordinator?.config.preserveStandardKVStorageDType, true)
            XCTAssertEqual(
                container.cacheCoordinator?.config.modelKey,
                "same-weights|gemma4-native-kv-dtype-v1")
        }
    }

    func testDiskReopenAndAppendPreserveMixedSlidingAndFullCacheDTypes() throws {
        try MLXMetalTestLock.withLock {
            let file = FileManager.default.temporaryDirectory
                .appendingPathComponent("gemma-kv-dtype-\(UUID().uuidString).safetensors")
            defer { try? FileManager.default.removeItem(at: file) }
            for model in try Self.models() {
                let container = Self.container(model)
                container.enableCaching(
                    config: CacheCoordinatorConfig(
                        usePagedCache: false, enableDiskCache: false))
                let preserve = try XCTUnwrap(
                    container.cacheCoordinator?.config.preserveStandardKVStorageDType)
                for dtype: DType in [.float16, .bfloat16, .float32] {
                    let original = model.newCache(parameters: nil)
                    // Cross the sliding window; the full cache must retain all rows.
                    for position in 0 ..< 7 {
                        let row = MLXArray.full(
                            [1, 1, 1, 32], values: MLXArray(Float(position) / 8)
                        )
                        .asType(dtype)
                        for layer in original { _ = layer.update(keys: row, values: -row) }
                    }
                    eval(original)
                    try MLX.save(
                        arrays: TQDiskSerializer.serialize(
                            cache: original, preserveStandardKVStorageDType: preserve), url: file)
                    var restored = model.newCache(parameters: nil)
                    XCTAssertEqual(
                        restoreFromDiskArrays(
                            try MLX.loadArrays(url: file), into: &restored), 7)
                    for position in 7 ..< 10 {
                        let row = MLXArray.full(
                            [1, 1, 1, 32], values: MLXArray(Float(position) / 8)
                        )
                        .asType(dtype)
                        for (cold, warm) in zip(original, restored) {
                            let expected = cold.update(keys: row, values: -row)
                            let actual = warm.update(keys: row, values: -row)
                            eval(expected.0, expected.1, actual.0, actual.1)
                            XCTAssertEqual(actual.0.dtype, dtype)
                            XCTAssertEqual(actual.1.dtype, dtype)
                            XCTAssertEqual(cold.offset, warm.offset)
                            XCTAssertTrue(MLX.all(expected.0 .== actual.0).item(Bool.self))
                            XCTAssertTrue(MLX.all(expected.1 .== actual.1).item(Bool.self))
                        }
                    }
                }
            }
        }
    }
}
