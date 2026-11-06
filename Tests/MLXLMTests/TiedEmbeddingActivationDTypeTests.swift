import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

final class TiedEmbeddingActivationDTypeTests: XCTestCase {
    private final class Fixture: Module {
        @ModuleInfo var embedding: Embedding

        init(_ embedding: Embedding) {
            self._embedding.wrappedValue = embedding
        }
    }

    func testHeadConversionPreservesSourceActivationPrecision() throws {
        try MLXMetalTestLock.withLock {
            for dtype: DType in [.float16, .bfloat16, .float32] {
                let weights = MLXArray(
                    (0 ..< 8 * 64).map {
                        Float($0 % 31 - 15) / 32
                    }, [8, 64]
                ).asType(dtype)
                let original = Embedding(weight: weights)
                let converted = quantizeTiedEmbeddingPreservingOutputDType(
                    original, groupSize: 32, bits: 6)
                let fixture = Fixture(converted)
                XCTAssertEqual(pinPreservedAffineEmbeddingOutputDTypes(fixture), 0)

                let tokens = MLXArray([Int32(1), 3])
                let rows = converted(tokens)
                let scale = MLXArray(Float(0.75), dtype: dtype)
                // These mixed operations model the router/layer-scale boundary
                // that promoted the fp16 Gemma activation stream to fp32.
                let scaled = rows * scale
                let logits = converted.asLinear(scaled)
                eval(rows, scaled, logits)
                XCTAssertEqual(rows.dtype, dtype)
                XCTAssertEqual(scaled.dtype, dtype)
                XCTAssertEqual(logits.dtype, dtype)
                XCTAssertLessThan(
                    abs(rows.asType(.float32) - original(tokens).asType(.float32))
                        .max().item(Float.self), 0.02)
                XCTAssertEqual(original.weight.dtype, dtype)
            }
        }
    }

    func testCheckpointQuantizedEmbeddingStillReceivesExistingAlignment() throws {
        try MLXMetalTestLock.withLock {
            let original = Embedding(weight: MLXArray.ones([8, 64], dtype: .float16))
            let checkpointEmbedding = QuantizedEmbedding(original, groupSize: 32, bits: 4)
            let fixture = Fixture(checkpointEmbedding)
            XCTAssertEqual(pinPreservedAffineEmbeddingOutputDTypes(fixture), 1)
            XCTAssertEqual(checkpointEmbedding(MLXArray([Int32(1)])).dtype, .bfloat16)
            XCTAssertEqual(pinPreservedAffineEmbeddingOutputDTypes(fixture), 0)
        }
    }
}
