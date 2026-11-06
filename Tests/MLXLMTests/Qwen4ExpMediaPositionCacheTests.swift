import Foundation
import MLX
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

final class Qwen4ExpMediaPositionCacheTests: XCTestCase {
    private func mediaPrefix() throws -> (Qwen4Exp, [KVCache]) {
        let config: [String: Any] = [
            "model_type": "qwen4_exp", "image_token_id": 125,
            "video_token_id": 126, "vision_start_token_id": 124,
            "text_config": [
                "model_type": "qwen4_exp_text", "dtype": "bfloat16",
                "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 64,
                "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 16,
                "linear_num_value_heads": 4, "linear_num_key_heads": 1,
                "linear_key_head_dim": 16, "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4, "vocab_size": 128,
                "num_experts": 4, "num_experts_per_tok": 2,
                "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
                "layer_types": ["linear_attention", "full_attention"],
                "full_attention_interval": 2,
                "hc_count": 4, "hc_lowrank": 8, "ple_layer_ids": [],
                "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 8,
                "indexer_budget": 32, "indexer_compress_ratio": 4,
                "mrope_section": [1, 1, 0],
            ],
            "vision_config": [
                "model_type": "qwen3_vl", "depth": 2, "hidden_size": 64,
                "intermediate_size": 128, "out_hidden_size": 64, "num_heads": 4,
                "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2,
                "num_position_embeddings": 64,
            ],
        ]
        let decoded = try JSONDecoder().decode(Qwen4ExpConfiguration.self,
            from: JSONSerialization.data(withJSONObject: config))
        let model = try Qwen4Exp(decoded, requesting: [.text, .vision])
        let cache = model.newCache(parameters: nil)
        let ids = MLXArray([Int32(1), 124, 125, 125, 125, 125, 2]).reshaped(1, 7)
        let frames = [THW(1, 4, 4)]
        let (_, delta) = Qwen3VLLanguage.getRopeIndex(
            inputIds: ids, imageGridTHW: frames, videoGridTHW: nil,
            spatialMergeSize: 2, imageTokenId: 125, videoTokenId: 126,
            visionStartTokenId: 124, attentionMask: nil)
        XCTAssertNotEqual(delta.item(Int.self), 0, "Fixture must require a media position offset")
        let pixels = MLXArray((0 ..< 16 * 1176).map { Float($0 % 31) / 31 })
            .reshaped(16, 1176)
        let input = LMInput(text: .init(tokens: ids),
            image: .init(pixels: pixels, frames: frames), mediaTokenIds: [125, 126])
        _ = try model.prepare(input, cache: cache, windowSize: 512)
        MLX.eval(cache)
        XCTAssertTrue(cache.allSatisfy { $0.offset == 7 })
        return (model, cache)
    }

    private func assertEqualLogits(_ lhs: MLXArray, _ rhs: MLXArray) {
        let difference = abs(lhs - rhs).max().item(Float.self)
        print("QWEN4_MEDIA_CACHE max_logit_difference=\(difference)")
        XCTAssertLessThanOrEqual(difference, 1e-5)
    }

    func testCopiedMediaCachePreservesDecodePositions() throws {
        let (model, cache) = try mediaPrefix()
        let copied = cache.map { $0.copy() }
        MLX.eval(copied)
        let token = MLXArray([Int32(3)]).reshaped(1, 1)
        assertEqualLogits(model(token, cache: cache), model(token, cache: copied))
    }

    func testDiskMediaCachePreservesDecodePositions() throws {
        let (model, cache) = try mediaPrefix()
        let arrays = TQDiskSerializer.serialize(cache: cache, preserveStandardKVStorageDType: true)
        MLX.eval(Array(arrays.values))
        var restored = model.newCache(parameters: nil)
        XCTAssertEqual(restoreFromDiskArrays(arrays, into: &restored, requirePromptBoundary: true), 7)
        let token = MLXArray([Int32(3)]).reshaped(1, 1)
        assertEqualLogits(model(token, cache: cache), model(token, cache: restored))
    }

    func testTextSuffixPreparePreservesMediaDecodePositions() throws {
        let (model, cache) = try mediaPrefix()
        let copied = cache.map { $0.copy() }
        MLX.eval(copied)
        let token = MLXArray([Int32(3)]).reshaped(1, 1)
        let reference = model(token, cache: cache)
        MLX.eval(reference)
        let result = try model.prepare(LMInput(tokens: token), cache: copied, windowSize: 512)
        guard case .logits(let output) = result else { return XCTFail("Expected suffix logits") }
        assertEqualLogits(reference, output.logits)
    }
    func testLegacyDiskRecordWithoutPositionRefusesBeforeMutatingSibling() throws {
        let (model, cache) = try mediaPrefix()
        var arrays = TQDiskSerializer.serialize(cache: cache, preserveStandardKVStorageDType: true)
        arrays.removeValue(forKey: "__qsa_1_media_position_offset__")
        var restored = model.newCache(parameters: nil)
        XCTAssertEqual(restoreFromDiskArrays(arrays, into: &restored, requirePromptBoundary: true), 0)
        XCTAssertTrue(restored.allSatisfy { $0.offset == 0 && $0.state.isEmpty })
    }

    func testMalformedPositionMetadataRefusesWholeRecord() throws {
        let (model, cache) = try mediaPrefix()
        for damaged in [MLXArray([Int32(1), 2]), MLXArray(Float(1))] {
            var arrays = TQDiskSerializer.serialize(cache: cache, preserveStandardKVStorageDType: true)
            arrays["__qsa_1_media_position_offset__"] = damaged
            var restored = model.newCache(parameters: nil)
            XCTAssertEqual(restoreFromDiskArrays(arrays, into: &restored, requirePromptBoundary: true), 0)
            XCTAssertTrue(restored.allSatisfy { $0.offset == 0 && $0.state.isEmpty })
        }
    }

    func testNativeBackbonePathsUseSameRestoredMediaPositions() throws {
        let (model, cache) = try mediaPrefix()
        let token = MLXArray([Int32(3)]).reshaped(1, 1)
        let reference = model(token, cache: cache.map { $0.copy() })
        assertEqualLogits(reference, model.nativeBackboneForward(token, cache: cache.map { $0.copy() }).logits)
        assertEqualLogits(reference, model.nativeAutoregressiveBackboneForward(token, cache: cache.map { $0.copy() }).logits)
        assertEqualLogits(reference, model.nativeBackboneMTPVerifyForward(token, cache: cache.map { $0.copy() }).logits)
    }

    func testActualSafetensorsMediaCachePreservesDecodePositions() throws {
        let (model, cache) = try mediaPrefix()
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen4-media-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: file) }
        try MLX.save(arrays: TQDiskSerializer.serialize(
            cache: cache, preserveStandardKVStorageDType: true), url: file)
        let arrays = try MLX.loadArrays(url: file)
        var restored = model.newCache(parameters: nil)
        XCTAssertEqual(restoreFromDiskArrays(arrays, into: &restored, requirePromptBoundary: true), 7)
        let token = MLXArray([Int32(3)]).reshaped(1, 1)
        assertEqualLogits(model(token, cache: cache), model(token, cache: restored))
    }

    func testIndependentRestoresFromOnePayloadDoNotAdvanceEachOther() throws {
        let (model, cache) = try mediaPrefix()
        let arrays = TQDiskSerializer.serialize(cache: cache, preserveStandardKVStorageDType: true)
        MLX.eval(Array(arrays.values))
        var first = model.newCache(parameters: nil)
        var second = model.newCache(parameters: nil)
        XCTAssertEqual(restoreFromDiskArrays(arrays, into: &first, requirePromptBoundary: true), 7)
        XCTAssertEqual(restoreFromDiskArrays(arrays, into: &second, requirePromptBoundary: true), 7)
        let token = MLXArray([Int32(3)]).reshaped(1, 1)
        let reference = model(token, cache: cache.map { $0.copy() })
        MLX.eval(reference)
        let firstLogits = model(token, cache: first)
        MLX.eval(firstLogits)
        assertEqualLogits(reference, firstLogits)
        assertEqualLogits(reference, model(token, cache: second))
    }

}
