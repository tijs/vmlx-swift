// Copyright © 2026. Apache-2.0.
import MLX
import MLXNN

/// Tokens in [lower, upper) use an affine table. The original BF16 rows
/// for both outer ranges are concatenated in their original order.
struct InklingVocabularyRange: Decodable, Sendable, Equatable {
    let lower: Int
    let upper: Int

    func validate(vocabularySize: Int) -> Bool {
        lower > 0 && lower < upper && upper < vocabularySize
            && lower.isMultiple(of: 32) && upper.isMultiple(of: 32)
    }
}

final class InklingTieredEmbedding: QuantizedEmbedding {
    @ParameterInfo(key: "bf16_edges") var edges: MLXArray
    let range: InklingVocabularyRange

    init(vocab: Int, hidden: Int, range: InklingVocabularyRange, group: Int, bits: Int) {
        self.range = range
        let rows = range.upper - range.lower
        _edges.wrappedValue = .zeros([vocab - rows, hidden], dtype: .bfloat16)
        super.init(weight: .zeros([rows, hidden * bits / 32], dtype: .uint32),
                   scales: .zeros([rows, hidden / group], dtype: .bfloat16),
                   biases: .zeros([rows, hidden / group], dtype: .bfloat16),
                   groupSize: group, bits: bits)
    }

    override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let ids = input.asType(.int32)
        let low = ids .< range.lower
        let middle = (ids .>= range.lower) .&& (ids .< range.upper)
        let packedIDs = clip(ids - range.lower, min: 0, max: range.upper - range.lower - 1)
        let edgeIDs = which(low, ids, maximum(ids - range.upper, 0) + range.lower)
        return which(middle.expandedDimensions(axis: -1), super.callAsFunction(packedIDs), edges[edgeIDs])
    }
}

final class InklingTieredHead: QuantizedLinear {
    @ParameterInfo(key: "bf16_edges") var edges: MLXArray
    let range: InklingVocabularyRange

    init(vocab: Int, hidden: Int, range: InklingVocabularyRange, group: Int, bits: Int) {
        self.range = range
        let rows = range.upper - range.lower
        _edges.wrappedValue = .zeros([vocab - rows, hidden], dtype: .bfloat16)
        super.init(weight: .zeros([rows, hidden * bits / 32], dtype: .uint32),
                   scales: .zeros([rows, hidden / group], dtype: .bfloat16),
                   biases: .zeros([rows, hidden / group], dtype: .bfloat16),
                   groupSize: group, bits: bits)
    }

    override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let edgeLogits: MLXArray
        if Inkling2HeadProjection.enabled && Inkling2HeadProjection.supports(input, weight: edges) {
            edgeLogits = Inkling2HeadProjection.project(input, weight: edges)
        } else {
            edgeLogits = matmul(input, edges.asType(.float32).T)
        }
        return concatenated([edgeLogits[.ellipsis, ..<range.lower], super.callAsFunction(input),
                             edgeLogits[.ellipsis, range.lower...]], axis: -1)
    }
}
