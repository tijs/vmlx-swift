import Foundation
import MLX
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

final class Qwen4ExpMediaSuffixCacheTests: XCTestCase {
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
                "layer_types": ["full_attention", "full_attention"],
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

    private func input(_ ids: [Int32], imageCount: Int = 0, video: Bool = false) -> LMInput {
        let tokens = MLXArray(ids).reshaped(1, ids.count)
        guard imageCount > 0 else { return LMInput(tokens: tokens) }
        let framePixels = (0 ..< 16 * 1176).map { Float($0 % 31) / 31 }
        let pixels = MLXArray(Array(repeating: framePixels, count: imageCount).flatMap { $0 })
            .reshaped(imageCount * 16, 1176)
        let frames = Array(repeating: THW(1, 4, 4), count: imageCount)
        return LMInput(text: .init(tokens: tokens),
            image: video ? nil : .init(pixels: pixels, frames: frames),
            video: video ? .init(pixels: pixels, frames: frames) : nil,
            mediaTokenIds: [125, 126])
    }

    private func compareSplit(prefix: LMInput, suffix: LMInput, full: LMInput) throws {
        let (model, _) = try mediaPrefix()
        let reference = model.newCache(parameters: nil)
        let split = model.newCache(parameters: nil)
        _ = try model.prepare(full, cache: reference, windowSize: 512)
        MLX.eval(reference)
        _ = try model.prepare(prefix, cache: split, windowSize: 512)
        MLX.eval(split)
        _ = try model.prepare(suffix, cache: split, windowSize: 512)
        MLX.eval(split)
        XCTAssertEqual(reference.first?.offset, split.first?.offset)
        let token = MLXArray([Int32(3)]).reshaped(1, 1)
        assertEqualLogits(model(token, cache: reference), model(token, cache: split))
    }

    func testTextSplitControlMatchesFreshPrefill() throws {
        let a: [Int32] = [5, 6]
        let b: [Int32] = [1, 7, 8, 9, 10, 11, 2]
        try compareSplit(prefix: input(a), suffix: input(b), full: input(a + b))
    }

    func testNewImageAfterCachedTextMatchesFreshPrefill() throws {
        let a: [Int32] = [5, 6]
        let b: [Int32] = [1, 124, 125, 125, 125, 125, 2]
        try compareSplit(prefix: input(a), suffix: input(b, imageCount: 1),
                         full: input(a + b, imageCount: 1))
    }

    func testNewVideoAfterCachedTextMatchesFreshPrefill() throws {
        let a: [Int32] = [5, 6]
        let b: [Int32] = [1, 124, 126, 126, 126, 126, 2]
        try compareSplit(prefix: input(a), suffix: input(b, imageCount: 1, video: true),
                         full: input(a + b, imageCount: 1, video: true))
    }

    func testSecondImageAccumulatesExistingMediaPositionOffset() throws {
        let a: [Int32] = [1, 124, 125, 125, 125, 125, 2, 5, 6]
        let b: [Int32] = [1, 124, 125, 125, 125, 125, 2]
        try compareSplit(prefix: input(a, imageCount: 1), suffix: input(b, imageCount: 1),
                         full: input(a + b, imageCount: 2))
    }
}
