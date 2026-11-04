// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon
import Testing

@Suite("synthetic Qwen35 router dequantization", .serialized)
struct Qwen35RouterDequantizationTests {
    @Test("expected input dimension resolves ambiguous packed geometry")
    func inferredPolicy() {
        // Eight U32 words and one scale group can encode (8,32), (4,64)
        // or (2,128); packed shape alone does not identify the policy.
        let eight = JangLoader.inferBitWidthAndGroupSize(
            packedDim: 8, numGroups: 1, knownGroupSize: 64,
            bitWidthsUsed: [4], expectedInDim: 32)
        #expect(eight.bits == 8 && eight.groupSize == 32)
        let four = JangLoader.inferBitWidthAndGroupSize(
            packedDim: 8, numGroups: 1, knownGroupSize: 32,
            bitWidthsUsed: [8], expectedInDim: 64)
        #expect(four.bits == 4 && four.groupSize == 64)
        let noHint = JangLoader.inferBitWidthAndGroupSize(
            packedDim: 8, numGroups: 1, knownGroupSize: 64)
        #expect(noHint.bits == 4 && noHint.groupSize == 64)

        // An incompatible hint does not throw: the loader falls back to a
        // guess, so an admission validator must check the returned policy and
        // unpacked dimension against the independently pinned contract.
        let incompatible = JangLoader.inferBitWidthAndGroupSize(
            packedDim: 8, numGroups: 1, knownGroupSize: 32,
            expectedInDim: 96)
        #expect(incompatible.bits == 8 && incompatible.groupSize == 32)
        #expect(8 * 32 / incompatible.bits != 96)
    }

    @Test("dequantized router and BF16 conversion match independent affine values")
    func numericAffine() throws {
        try MLXMetalTestLock.withLock {
            let base = "language_model.model.layers.0.mlp.gate"
            let quantized = (0 ..< 32).map { UInt32($0) }
            let packed = (0 ..< 8).map { word in
                (0 ..< 4).reduce(UInt32(0)) { value, lane in
                    value | (quantized[word * 4 + lane] << UInt32(8 * lane))
                }
            }
            var tensors: [String: MLXArray] = [
                base + ".weight": MLXArray(packed, [1, 8]),
                base + ".scales": MLXArray([Float(0.30078125)], [1, 1]).asType(.bfloat16),
                base + ".biases": MLXArray([Float(0)], [1, 1]).asType(.bfloat16),
            ]
            JangLoader.dequantizeMoEGates(weights: &tensors, groupSize: 32,
                bitWidthsUsed: [4], hiddenSizeHint: 32)
            let result = try #require(tensors[base + ".weight"])
            #expect(result.dtype == .float32 && result.shape == [1, 32])
            #expect(tensors[base + ".scales"] == nil && tensors[base + ".biases"] == nil)
            let rounded = result.asType(.bfloat16).asType(.float32)
            MLX.eval(result, rounded)
            let reference = quantized.map { Float($0) * 0.30078125 }
            // Round a Float bit pattern to BF16 using round-to-nearest-even,
            // independently of MLX. Inputs are finite and normal here.
            let expectedBF16 = reference.map { value in
                let bits = value.bitPattern
                let rounded = bits &+ 0x7fff &+ ((bits >> 16) & 1)
                return Float(bitPattern: rounded & 0xffff0000)
            }
            // Dequantization inherits the BF16 companion dtype before the
            // loader's asType(.float32); F32 cannot restore discarded bits.
            #expect(result.asArray(Float.self) == expectedBF16)
            #expect(rounded.asArray(Float.self) == expectedBF16)
        }
    }
}
