import MLX
import MLXNN
import XCTest
@testable import MLXLLM

final class InklingTieredVocabularyTests: XCTestCase {
    func testRowsAtBothBoundariesAndAllLogits() throws {
        try Device.withDefaultDevice(.cpu) {
            let vocab = 160, width = 128
            let range = InklingVocabularyRange(lower: 32, upper: 128)
            let embedding = InklingTieredEmbedding(vocab: vocab, hidden: width, range: range,
                                                  group: 64, bits: 4)
            let head = InklingTieredHead(vocab: vocab, hidden: width, range: range,
                                         group: 64, bits: 4)
            let packedRows = range.upper - range.lower
            let q = (0..<packedRows).flatMap { i in
                Array(repeating: UInt32(i % 16) * 0x11111111, count: width / 8)
            }
            let rowValue = (0..<vocab).map {
                Float((range.lower..<range.upper).contains($0) ? ($0 - range.lower) % 16 : -$0 - 1)
            }
            let edges = (Array(0..<range.lower) + Array(range.upper..<vocab)).flatMap {
                Array(repeating: rowValue[$0], count: width)
            }
            let parameters = ModuleParameters.unflattened([
                "weight": MLXArray(q, [packedRows, width / 8]),
                "scales": MLXArray.ones([packedRows, width / 64], dtype: .bfloat16),
                "biases": MLXArray.zeros([packedRows, width / 64], dtype: .bfloat16),
                "bf16_edges": MLXArray(edges, [vocab-packedRows,width]).asType(.bfloat16),
            ])
            try embedding.update(parameters: parameters, verify: .all)
            try head.update(parameters: parameters, verify: .all)
            let ids: [Int32] = [0, 31, 32, 33, 127, 128, 159]
            let values = embedding(MLXArray(ids, [1, ids.count]))
            XCTAssertEqual(values.shape, [1, ids.count, width])
            XCTAssertEqual(values[0, 0..., 0].asType(.float32).asArray(Float.self), ids.map { rowValue[Int($0)] })
            let logits = head(.ones([1, 2, width], dtype: .float32))
            XCTAssertEqual(logits.shape, [1, 2, vocab])
            XCTAssertEqual(logits[0, 1].asArray(Float.self), rowValue.map { $0 * Float(width) })
        }
    }
}
