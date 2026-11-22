// Copyright © 2026. Apache-2.0.
// Stateless @mx.compile boundaries from pinned Inkling2 language.py.
// See docs/INKLING2_SOURCE.md. No weights or visit caches are captured.
import Foundation
import MLX

enum Inkling2CompiledPrimitives {
    static let enabled = ProcessInfo.processInfo.environment["VMLX_INKLING_COMPILE_PRIMITIVES"] == "1"
    static var useCompiled: Bool { enabled && Device.defaultDevice().deviceType == .gpu }

    static func clipEager(_ x: MLXArray, threshold: MLXArray, scale: MLXArray?) -> MLXArray {
        let y = x.asType(.float32) * (scale?.asType(.float32) ?? MLXArray(Float(1)))
        let norm = sqrt(sum(y * y, axis: -1, keepDims: true))
        return (y * minimum(threshold / (norm + 1e-6), 1)).asType(x.dtype)
    }
    private static let clipPlain = vmlxTrustedCompile { args in
        [clipEager(args[0], threshold: args[1], scale: nil)]
    }
    private static let clipScaled = vmlxTrustedCompile { args in
        [clipEager(args[0], threshold: args[1], scale: args[2])]
    }
    static func clip(_ x: MLXArray, threshold: Float, scale: MLXArray?) -> MLXArray {
        let limit = MLXArray(threshold)
        if useCompiled {
            if let scale { return clipScaled([x, limit, scale])[0] }
            return clipPlain([x, limit])[0]
        }
        return clipEager(x, threshold: limit, scale: scale)
    }

    static func activationEager(_ gate: MLXArray, _ up: MLXArray,
                                gateCap: MLXArray, upCap: MLXArray) -> MLXArray {
        let g = gate.asType(.float32), u = up.asType(.float32)
        return (tanh(g / gateCap) * gateCap * sigmoid(g) * tanh(u / upCap) * upCap).asType(gate.dtype)
    }
    private static let compiledActivation = vmlxTrustedCompile { args in
        [activationEager(args[0], args[1], gateCap: args[2], upCap: args[3])]
    }
    static func activation(_ gate: MLXArray, _ up: MLXArray,
                           gateCap: Float, upCap: Float) -> MLXArray {
        let g = MLXArray(gateCap), u = MLXArray(upCap)
        if useCompiled { return compiledActivation([gate, up, g, u])[0] }
        return activationEager(gate, up, gateCap: g, upCap: u)
    }

    // topK is model configuration, never request state. Router storage stays
    // BF16 where authored; only the scoring operation promotes it to FP32.
    static func routeEager(_ args: [MLXArray], topK: Int) -> [MLXArray] {
        let scores = sigmoid(matmul(args[0].asType(.float32), args[1].asType(.float32).T))
        let indices = argPartition(-(scores + args[2]), kth: topK - 1, axis: -1)[.ellipsis, 0..<topK]
        var weights = takeAlong(scores, indices, axis: -1)
        weights = weights / (sum(weights, axis: -1, keepDims: true) + 1e-20) * args[3]
        return [indices, weights]
    }
    static func makeRoute(topK: Int) -> @Sendable ([MLXArray]) -> [MLXArray] {
        vmlxTrustedCompile { args in routeEager(args, topK: topK) }
    }

    static func combineEager(_ out: MLXArray, _ weights: MLXArray) -> MLXArray {
        sum(out.asType(.float32) * weights.expandedDimensions(axis: -1), axis: -2).asType(out.dtype)
    }
    private static let compiledCombine = vmlxTrustedCompile { args in
        [combineEager(args[0], args[1])]
    }
    static func combine(_ out: MLXArray, _ weights: MLXArray) -> MLXArray {
        if useCompiled { return compiledCombine([out, weights])[0] }
        return combineEager(out, weights)
    }
}
