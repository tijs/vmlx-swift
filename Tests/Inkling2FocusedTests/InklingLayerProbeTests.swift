import Foundation
import MLX
import MLXNN
import XCTest
@testable import MLXLLM

final class InklingLayerProbeTests: XCTestCase {
    func testActualLayerTwoStages() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["VMLX_INKLING_LAYER_PROBE"] == "1" else { throw XCTSkip("Explicit supervised checkpoint projection diagnostic") }
        let directory = URL(fileURLWithPath: try XCTUnwrap(env["VMLX_INKLING_PROBE_MODEL"]))
        let output = URL(fileURLWithPath: try XCTUnwrap(env["VMLX_INKLING_PROBE_OUTPUT"]))
        let fixture = URL(fileURLWithPath: try XCTUnwrap(env["VMLX_INKLING_LAYER_FIXTURE"]))
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let config = try JSONDecoder().decode(Inkling2LoopConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        try Device.withDefaultDevice(.gpu) {
            Memory.cacheLimit = 536_870_912
            var arrays = try loadArrays(url: fixture)
            let x = try XCTUnwrap(arrays.removeValue(forKey: "fixture.input"))
            let block = try XCTUnwrap(arrays.removeValue(forKey: "fixture.block"))
            let layer = InklingDecoderLayer(config, layer: 2)
            try layer.update(parameters: ModuleParameters.unflattened(arrays), verify: .all)
            eval(layer)
            var stages: [String: MLXArray] = [:]
            let mixed = layer.attentionInput!([block], x)
            stages["01-mixed"] = mixed
            let normalized = layer.localNorm!(mixed[.ellipsis, 0, 0...]).asType(x.dtype)
            stages["02-normalized"] = normalized
            let attn = try layer.attention(normalized, keyInput: normalized, valueInput: normalized, cache: nil)
            stages["03-attention"] = attn
            let conv = layer.attnConv(attn, nil)
            stages["04-attn-convolution"] = conv
            let residual = conv.asType(.float32) + x.asType(.float32)
            stages["05-residual"] = residual
            let hidden = layer.mlpInput([block], residual)[.ellipsis, 0, 0...]
            stages["06-mlp-input"] = hidden
            let post = layer.postNorm(hidden).asType(x.dtype)
            stages["07-post-norm"] = post
            let scores = sigmoid(matmul(post.asType(.float32), layer.mlp.router.asType(.float32).T))
            stages["08-route-scores"] = scores
            let indices = argPartition(-(scores + layer.mlp.correction), kth: config.text.topK-1, axis: -1)[.ellipsis, 0..<config.text.topK]
            stages["09-route-indices"] = indices.asType(.float32)
            var weights = takeAlong(scores, indices, axis: -1)
            weights = weights / (sum(weights, axis: -1, keepDims: true) + 1e-20) * config.text.routeScale
            stages["10-route-weights"] = weights
            let routed = layer.mlp.experts(post, indices)
            stages["11-expert-output"] = routed
            let mlp = sum(routed.asType(.float32) * weights.expandedDimensions(axis: -1), axis: -2).asType(routed.dtype)
            stages["12-mlp-combined"] = mlp
            let mlpConv = layer.mlpConv(mlp, nil)
            stages["13-mlp-convolution"] = mlpConv
            let result = (residual + mlpConv.asType(.float32)).asType(x.dtype)
            stages["14-result"] = result
            var blocks = [block]
            let ordinary = try layer(x, blocks: &blocks, cache: nil, messages: nil)
            eval(result, ordinary)
            XCTAssertEqual(max(abs(result.asType(.float32)-ordinary.asType(.float32))).item(Float.self), 0)
            try save(arrays: stages, url: output.appendingPathComponent("layer-stages.safetensors"))
            try JSONSerialization.data(withJSONObject: ["status": "completed", "replay_matches": true], options: [.prettyPrinted])
                .write(to: output.appendingPathComponent("record.json"))
        }
    }
}
