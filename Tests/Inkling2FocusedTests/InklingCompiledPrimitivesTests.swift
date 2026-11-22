import Foundation
import MLX
import XCTest
@testable import MLXLLM

final class InklingCompiledPrimitivesTests: XCTestCase {
    private func array(_ value: Any, dtype: DType = .float32) -> MLXArray {
        func shape(_ value: Any) -> [Int] {
            guard let a = value as? [Any] else { return [] }
            return [a.count] + (a.first.map(shape) ?? [])
        }
        func values(_ value: Any) -> [Float] {
            if let a = value as? [Any] { return a.flatMap(values) }
            return [(value as! NSNumber).floatValue]
        }
        return MLXArray(values(value), shape(value)).asType(dtype)
    }
    private func check(_ got: MLXArray, _ expected: Any, tolerance: Float,
                       file: StaticString = #filePath, line: UInt = #line) {
        let want = (expected as? MLXArray) ?? array(expected)
        XCTAssertEqual(got.shape, want.shape, file: file, line: line)
        XCTAssertTrue(all(isFinite(got)).item(Bool.self), file: file, line: line)
        XCTAssertLessThanOrEqual(max(abs(got.asType(.float32) - want)).item(Float.self),
                                 tolerance, file: file, line: line)
    }
    private func golden() throws {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "InklingCompiledPrimitives", withExtension: "json"))
        let fixture = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        for row in fixture["clips"] as! [[String: Any]] {
            let dtype: DType = row["dtype"] as! String == "bfloat16" ? .bfloat16 : .float32
            let x = array(row["x"]!, dtype: dtype)
            let scale = row["scale"] is NSNull ? nil : array(row["scale"]!)
            let got = inklingL2Clip(x, threshold: (row["threshold"] as! NSNumber).floatValue, scale: scale)
            XCTAssertEqual(got.dtype, dtype)
            check(got, row["expected"]!, tolerance: dtype == .bfloat16 ? 0 : 2e-6)
            check(inklingL2Clip(-x, threshold: (row["threshold"] as! NSNumber).floatValue, scale: scale),
                  -array(row["expected"]!), tolerance: dtype == .bfloat16 ? 0 : 2e-6)
            check(inklingL2Clip(x, threshold: (row["threshold"] as! NSNumber).floatValue / 2, scale: scale),
                  array(row["expected"]!) / 2, tolerance: dtype == .bfloat16 ? 0 : 2e-6)
            if let scale {
                check(inklingL2Clip(x, threshold: (row["threshold"] as! NSNumber).floatValue, scale: -scale),
                      -array(row["expected"]!), tolerance: dtype == .bfloat16 ? 0 : 2e-6)
            }
        }
        for row in fixture["activations"] as! [[String: Any]] {
            let dtype: DType = row["dtype"] as! String == "bfloat16" ? .bfloat16 : .float32
            let gateCap = (row["gate_cap"] as! NSNumber).floatValue
            let upCap = (row["up_cap"] as! NSNumber).floatValue
            let got = Inkling2CompiledPrimitives.activation(array(row["gate"]!, dtype: dtype),
                array(row["up"]!, dtype: dtype), gateCap: gateCap, upCap: upCap)
            XCTAssertEqual(got.dtype, dtype)
            check(got, row["expected"]!, tolerance: dtype == .bfloat16 ? 0 : 2e-5)
            let changed = Inkling2CompiledPrimitives.activation(array(row["gate"]!, dtype: dtype),
                -array(row["up"]!, dtype: dtype), gateCap: gateCap, upCap: upCap)
            check(changed, -array(row["expected"]!), tolerance: dtype == .bfloat16 ? 0 : 2e-5)
        }
        for row in fixture["combinations"] as! [[String: Any]] {
            let dtype: DType = row["dtype"] as! String == "bfloat16" ? .bfloat16 : .float32
            let got = Inkling2CompiledPrimitives.combine(array(row["out"]!, dtype: dtype), array(row["weights"]!))
            XCTAssertEqual(got.dtype, dtype)
            check(got, row["expected"]!, tolerance: 0)
            let changed = Inkling2CompiledPrimitives.combine(array(row["out"]!, dtype: dtype),
                array(row["weights"]!) * 2)
            check(changed, array(row["expected"]!) * 2, tolerance: 0)
        }
        let row = fixture["route"] as! [String: Any]
        let k = (row["top_k"] as! NSNumber).intValue
        let args = [array(row["x"]!, dtype: .bfloat16), array(row["router"]!, dtype: .bfloat16),
                    array(row["bias"]!), MLXArray((row["route_scale"] as! NSNumber).floatValue)]
        let eager = Inkling2CompiledPrimitives.routeEager(args, topK: k)
        let compiled = Inkling2CompiledPrimitives.makeRoute(topK: k)
        let route = Inkling2CompiledPrimitives.useCompiled ? compiled(args) : eager
        let ids = route[0].asType(.uint32).asArray(UInt32.self)
        let values = route[1].asArray(Float.self)
        let expectedIDs = array(row["indices"]!).asType(.uint32).asArray(UInt32.self)
        let expectedWeights = array(row["weights"]!).asArray(Float.self)
        XCTAssertEqual(ids, eager[0].asType(.uint32).asArray(UInt32.self))
        XCTAssertLessThanOrEqual(max(abs(route[1] - eager[1])).item(Float.self), 2e-6)
        XCTAssertEqual(route[1].dtype, .float32)
        for start in stride(from: 0, to: ids.count, by: k) {
            XCTAssertEqual(Set(ids[start..<start+k]), Set(expectedIDs[start..<start+k]))
            for index in start..<start+k {
                let matching = try XCTUnwrap((start..<start+k).first { expectedIDs[$0] == ids[index] })
                XCTAssertEqual(values[index], expectedWeights[matching], accuracy: 2e-6)
            }
        }
        if Inkling2CompiledPrimitives.useCompiled {
            // Reuse the same trace with changed request data and changed
            // parameter buffers, then restore the original inputs. No closure
            // may capture its first scoring input or router tensor.
            for changedIndex in [0, 1] {
                var changed = args
                changed[changedIndex] = .zeros(args[changedIndex].shape, dtype: args[changedIndex].dtype)
                let zero = compiled(changed)
                let zeroIDs = zero[0].asType(.uint32).asArray(UInt32.self)
                let experts = args[1].dim(0)
                for start in stride(from: 0, to: zeroIDs.count, by: k) {
                    XCTAssertEqual(Set(zeroIDs[start..<start+k]), Set((experts-k..<experts).map(UInt32.init)))
                }
                for weight in zero[1].asArray(Float.self) {
                    XCTAssertEqual(weight, 8 / Float(k), accuracy: 2e-6)
                }
            }
            var smallerScale = args
            smallerScale[3] = MLXArray(Float(4))
            let scaled = compiled(smallerScale)
            XCTAssertEqual(scaled[0].asType(.uint32).asArray(UInt32.self), ids)
            XCTAssertLessThanOrEqual(max(abs(scaled[1] - route[1] / 2)).item(Float.self), 2e-6)
            let restored = compiled(args)
            XCTAssertEqual(restored[0].asType(.uint32).asArray(UInt32.self), ids)
            XCTAssertLessThanOrEqual(max(abs(restored[1] - route[1])).item(Float.self), 2e-6)
        }
    }
    func testIndependentPrimitiveGoldenCPU() throws {
        try Device.withDefaultDevice(.cpu) { try golden() }
    }
    func testIndependentPrimitiveGoldenMetal() throws {
        guard ProcessInfo.processInfo.environment["VMLX_INKLING_METAL_FIXTURE"] == "1" else {
            throw XCTSkip("Explicit Metal fixture session required")
        }
        try Device.withDefaultDevice(.gpu) { try golden() }
    }
}
