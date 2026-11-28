#if canImport(Darwin)
import Foundation
import MLX
import MLXLMCommon

/// Evaluate global Inkling attention without materializing full-history KV.
/// Convolution state and local visits are owned by the original visit caches.
func inklingSpilledAttention(
    queries: MLXArray, keys: MLXArray, values: MLXArray, relative: MLXArray,
    projection: MLXArray, sinks: MLXArray, history: ActiveKVSpillHistory, extent: Int
) throws -> MLXArray {
    let start = history.count
    try history.append(keys: keys, values: values)
    return try withError { error in
        let batch = queries.dim(0), heads = queries.dim(1), rows = queries.dim(2), width = queries.dim(3)
        let kvHeads = keys.dim(1), group = heads / kvHeads
        let rowShape = [batch, kvHeads, group, rows]
        let query = queries.asType(.float32).reshaped(rowShape + [width])
        let projected = matmul(relative, projection).transposed(0, 2, 1, 3)
        if Inkling2SpilledPrefill.supports(
            queries: queries, keys: keys, values: values, pageTokens: history.pageTokens
        ) {
            return try Inkling2SpilledPrefill.attention(
                queries: queries, projected: projected, sinks: sinks, history: history, extent: extent)
        }
        if Inkling2SpilledVector.supports(queries: queries, keys: keys, values: values, count: history.count) {
            return try Inkling2SpilledVector.attention(
                queries: queries, projectedBias: projected[.ellipsis, ..<extent].asType(queries.dtype),
                sinks: sinks, history: history)
        }
        let positions = MLXArray(Int32(start)..<Int32(start + rows)).expandedDimensions(axis: 1)
        var attention = KVTileSoftmax(
            rowShape: rowShape, valueWidth: width, sink: sinks.reshaped(rowShape + [1]))
        func accumulate(_ range: Range<Int>, tileKeys: MLXArray, tileValues: MLXArray) {
            let logits = matmul(
                query, tileKeys.asType(.float32).expandedDimensions(axis: 2).swappedAxes(-1, -2))
                * (1 / sqrt(Float(width)))
            let scores: MLXArray
            if start - (range.upperBound - 1) >= extent {
                // Every query can see this tile and every key is beyond the relative bias range.
                scores = logits
            } else {
                let positionsInTile = MLXArray(Int32(range.lowerBound)..<Int32(range.upperBound))
                let distance = positions - positionsInTile
                let maskShape = [1, 1, rows, range.count]
                let index = broadcast(
                    clip(distance, min: 0, max: extent - 1).reshaped(maskShape),
                    to: [batch, heads, rows, range.count])
                let bias = takeAlong(projected, index, axis: -1)
                let relativeBias = MLX.where(distance.reshaped(maskShape) .>= extent, MLXArray(Float(0)), bias)
                let causal = (distance .>= 0).reshaped([1, 1, 1, rows, range.count])
                scores = MLX.where(
                    causal, logits + relativeBias.asType(.float32).reshaped(rowShape + [range.count]),
                    MLXArray(-Float.infinity))
            }
            attention.accumulate(scores: scores, values: tileValues.expandedDimensions(axis: 2))
        }
        try history.forEachTile { page, keys, values in
            // A larger SSD read can amortize transfers and synchronization.
            // Keep the original reduction width and order for greedy parity.
            let reductionWidth = 128
            if page.count <= reductionWidth {
                accumulate(page, tileKeys: keys, tileValues: values)
            } else {
                for startInPage in stride(from: 0, to: page.count, by: reductionWidth) {
                    let endInPage = min(startInPage + reductionWidth, page.count)
                    let slice = startInPage..<endInPage
                    accumulate(
                        (page.lowerBound + startInPage)..<(page.lowerBound + endInPage),
                        tileKeys: contiguous(keys[.ellipsis, slice, 0...]),
                        tileValues: contiguous(values[.ellipsis, slice, 0...]))
                }
            }
            attention.materialize()
            try error.check()
        }
        return attention.output(dtype: queries.dtype).reshaped([batch, heads, rows, width])
    }
}
#endif
