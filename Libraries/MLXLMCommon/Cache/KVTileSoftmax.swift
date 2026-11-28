import MLX

/// Accumulates attention over separately staged KV tiles. The caller computes
/// tile scores with its own positional bias and masking. Query rows and heads
/// are retained between tiles, while each tile's score and value arrays are
/// released after evaluation.
package struct KVTileSoftmax {
    private var maximumScore: MLXArray
    private var denominator: MLXArray
    private var numerator: MLXArray

    /// A sink contributes to the denominator exactly once and has no value
    /// vector. Its shape is the score shape with the key axis replaced by 1.
    package init(rowShape: [Int], valueWidth: Int, sink: MLXArray? = nil) {
        let scalarShape = rowShape + [1]
        maximumScore = sink?.asType(.float32)
            ?? MLXArray.full(scalarShape, values: MLXArray(-Float.infinity))
        denominator = sink.map { isFinite($0).asType(.float32) }
            ?? MLXArray.zeros(scalarShape, dtype: .float32)
        numerator = MLXArray.zeros(rowShape + [valueWidth], dtype: .float32)
    }

    /// Masked scores use negative infinity. Values must broadcast against
    /// scores for matmul, including any grouped-query head dimension.
    package mutating func accumulate(scores: MLXArray, values: MLXArray) {
        let scores = scores.asType(.float32)
        let nextMaximum = maximum(maximumScore, scores.max(axis: -1, keepDims: true))
        // When the first tile is entirely masked, subtracting -infinity
        // would yield NaNs. A finite origin makes both contributions zero.
        let origin = MLX.where(nextMaximum .== -Float.infinity, MLXArray(Float(0)), nextMaximum)
        let priorScale = exp(maximumScore - origin)
        let weights = exp(scores - origin)
        numerator = numerator * priorScale + matmul(weights, values.asType(.float32))
        denominator = denominator * priorScale + weights.sum(axis: -1, keepDims: true)
        maximumScore = nextMaximum
    }

    /// Evaluate the reduced state before the caller releases its staged page.
    package func materialize() {
        eval(maximumScore, denominator, numerator)
    }

    package func output(dtype: DType) -> MLXArray {
        // An unmasked row always has a term equal to one. An entirely masked
        // row returns zero rather than dividing by zero.
        (numerator / maximum(denominator, MLXArray(Float(1)))).asType(dtype)
    }
}
