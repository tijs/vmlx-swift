// Copyright © 2026 Apple Inc.

import MLX

/// Position-only common subexpressions for ONE Qwen4 AR forward. The owner
/// creates a new context for every token; nothing survives in a KV/SSM cache,
/// a compiled closure, or a global model table. Different layer instances can
/// share factors only when their complete rotary configuration matches.
///
/// The original FP32 position product, cos/sin operations, and output cast
/// still execute. This changes graph construction/reuse, not rotary arithmetic.
/// Explicit media position arrays bypass this helper at the attention caller.
final class Qwen4ExpRotaryContext {
    private struct Key: Hashable {
        let signature: Qwen35Language.RotaryEmbedding.FactorSignature
        let dtype: DType
        let start: Int
        let end: Int
        let step: Int
    }

    private var entries: [Key: (MLXArray, MLXArray)] = [:]
    private(set) var reuseCount = 0
    var factorCount: Int { entries.count }

    func factors(
        rotary: Qwen35Language.RotaryEmbedding, like x: MLXArray,
        start: Int, end: Int, step: Int = 1
    ) -> (MLXArray, MLXArray) {
        precondition(step > 0)
        let key = Key(
            signature: rotary.factorSignature, dtype: x.dtype,
            start: start, end: end, step: step)
        if let found = entries[key] {
            reuseCount += 1
            return found
        }
        let ids = MLXArray(stride(from: start, to: end, by: step))
            .asType(.int32).reshaped(1, -1)
        let result = rotary(x: x, positionIds: ids)
        entries[key] = result
        return result
    }
}
