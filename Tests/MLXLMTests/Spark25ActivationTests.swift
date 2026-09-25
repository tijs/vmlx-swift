import MLX
import MLXNN
import Testing

@testable import MLXLLM

@Suite(.serialized)
struct Spark25ActivationTests {
    private func expectSame(_ gate: MLXArray, _ up: MLXArray) {
        let expected = gelu(gate) * up
        let actual = Spark25Activation.geluMultiply(gate, up)
        eval(expected, actual)
        #expect(actual.shape == expected.shape)
        #expect(actual.dtype == expected.dtype)
        if actual.dtype == .bfloat16 {
            let a = actual.view(dtype: .uint16).asArray(UInt16.self)
            let e = expected.view(dtype: .uint16).asArray(UInt16.self)
            func nan(_ b: UInt16) -> Bool { b & 0x7f80 == 0x7f80 && b & 0x007f != 0 }
            let differences = zip(a, e).filter { $0 != $1 && !(nan($0) && nan($1)) }.count
            #expect(differences == 0)
        } else {
            #expect((actual .== expected).all().item(Bool.self))
        }
    }

    @Test func everyBF16EncodingPreservesReferenceRounding() {
        let gate = MLXArray((0 ..< 65536).map { UInt16($0) }).view(dtype: .bfloat16).reshaped(128, 512)
        for factor: Float in [1, -3, 0, 0.125] {
            expectSame(gate, full(gate.shape, values: factor).asType(.bfloat16))
        }
    }

    @Test func raptorShapesAndNoncontiguousInputs() {
        MLXRandom.seed(7)
        for rows in [1, 3, 127, 128, 129, 512] {
            let gate = MLXRandom.normal([rows, 10240]).asType(.bfloat16)
            let up = MLXRandom.normal([rows, 10240]).asType(.bfloat16)
            expectSame(gate, up)
            expectSame(gate.transposed(), up.transposed())
        }
        // Exercise a fused shape whose element count is not threadgroup aligned.
        expectSame(
            MLXRandom.normal([129, 3]).asType(.bfloat16),
            MLXRandom.normal([129, 3]).asType(.bfloat16))
    }

    @Test func otherDtypesAndBroadcastUseReferencePath() {
        for dtype: DType in [.float16, .float32, .bfloat16] {
            let gate = tiled(MLXArray([Float(-2), 0, 3]), repetitions: [128, 1]).asType(dtype)
            expectSame(gate, tiled(MLXArray([Float(1), -1, 2]), repetitions: [128, 1]).asType(dtype))
            expectSame(gate, MLXArray(Float(2)).asType(dtype))
        }
    }

    @Test func cpuUsesReferencePath() {
        Device.withDefaultDevice(.cpu) {
            expectSame(
                tiled(MLXArray([Float(-2), 0, 3]), repetitions: [128, 1]).asType(.bfloat16),
                tiled(MLXArray([Float(1), -1, 2]), repetitions: [128, 1]).asType(.bfloat16))
        }
    }

    @Test func compiledAndBackwardPathsMatchReference() {
        let g = tiled(MLXArray([Float(-2), -0.25, 0, 0.25, 3]), repetitions: [128, 1]).asType(.bfloat16)
        let u = tiled(MLXArray([Float(1), -1, 2, 0.5, 3]), repetitions: [128, 1]).asType(.bfloat16)
        let cotangent = ones(g.shape, dtype: .bfloat16)
        let reference = vjp({ [gelu($0[0]) * $0[1]] }, primals: [g, u], cotangents: [cotangent])
        let actual = vjp(
            { [Spark25Activation.geluMultiply($0[0], $0[1])] }, primals: [g, u],
            cotangents: [cotangent])
        eval(reference.1, actual.1)
        for (a, b) in zip(reference.1, actual.1) {
            #expect((a .== b).all().item(Bool.self))
        }
        let compiled = compile { (a: MLXArray, b: MLXArray) in Spark25Activation.geluMultiply(a, b)
        }
        #expect((compiled(g, u) .== gelu(g) * u).all().item(Bool.self))
    }

    @Test func scopedStreamsRespectTheirActualDevice() {
        Device.withDefaultDevice(.gpu) {
            for device in [Device.cpu, Device.gpu] {
                Stream.withNewDefaultStream(device: device) {
                    expectSame(
                        tiled(MLXArray([Float(-2), 0, 3]), repetitions: [128, 1]).asType(.bfloat16),
                        tiled(MLXArray([Float(1), -1, 2]), repetitions: [128, 1]).asType(.bfloat16))
                }
            }
        }
    }

    @Test func forwardDerivativesAndVectorizationUseReferenceTransforms() {
        let g = tiled(MLXArray([Float(-2), -0.25, 0, 0.25, 3]), repetitions: [128, 1]).asType(.bfloat16)
        let u = tiled(MLXArray([Float(1), -1, 2, 0.5, 3]), repetitions: [128, 1]).asType(.bfloat16)
        let tangents = [ones(g.shape, dtype: .bfloat16), ones(u.shape, dtype: .bfloat16)]
        let reference = jvp({ [gelu($0[0]) * $0[1]] }, primals: [g, u], tangents: tangents)
        let actual = jvp(
            { [Spark25Activation.geluMultiply($0[0], $0[1])] }, primals: [g, u], tangents: tangents)
        eval(reference.1, actual.1)
        #expect((reference.1[0] .== actual.1[0]).all().item(Bool.self))
        let inputs = stacked([g, g * 2])
        let referenceMap = vmap { gelu($0) * u }
        let actualMap = vmap { Spark25Activation.geluMultiply($0, u) }
        #expect((referenceMap(inputs) .== actualMap(inputs)).all().item(Bool.self))
    }

    @Test func signedMetalGridOverflowUsesLazyReferencePath() {
        // A zero-stride broadcast creates the logical shape without allocating
        // a multi-gigabyte buffer. Do not evaluate this diagnostic graph.
        let input = broadcast(MLXArray(Float(1)).asType(.bfloat16), to: [128, 16_777_216])
        let actual = Spark25Activation.geluMultiply(input, input)
        #expect(actual.shape == input.shape)
        #expect(actual.dtype == .bfloat16)
    }

    @Test func emptyInputsPreserveShape() {
        let empty = zeros([0, 10240], dtype: .bfloat16)
        let actual = Spark25Activation.geluMultiply(empty, empty)
        eval(actual)
        #expect(actual.shape == [0, 10240])
    }
}
