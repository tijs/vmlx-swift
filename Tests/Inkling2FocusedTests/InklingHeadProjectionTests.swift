import Foundation
import MLX
import XCTest
@testable import MLXLLM

final class InklingHeadProjectionTests: XCTestCase {
    private func requireMetal() throws {
        guard ProcessInfo.processInfo.environment["VMLX_INKLING_METAL_FIXTURE"] == "1" else {
            throw XCTSkip("Explicit Metal fixture session required")
        }
    }

    private func check(_ x: MLXArray, _ weight: MLXArray,
                       file: StaticString = #filePath, line: UInt = #line) {
        let got = Inkling2HeadProjection.project(x, weight: weight)
        let reference = matmul(x, weight.asType(.float32).T)
        XCTAssertEqual(got.dtype, .float32, file: file, line: line)
        XCTAssertEqual(got.shape, reference.shape, file: file, line: line)
        XCTAssertTrue(all(isFinite(got)).item(Bool.self), file: file, line: line)
        XCTAssertEqual(got.asArray(Float.self).map(\.bitPattern),
                       reference.asArray(Float.self).map(\.bitPattern), file: file, line: line)
    }

    func testMixedStorageMatchesStockFP32GEMV() throws {
        try requireMetal()
        try Device.withDefaultDevice(.gpu) {
            for k in [128, 2304] {
                let n = 4096
                let raw = (0..<n*k).map { Float(($0 * 37 + $0 / k * 13) % 509 - 254) / 127 }
                let weight = MLXArray(raw, [n, k]).asType(.bfloat16)
                let x = MLXArray((0..<k).map { Float(($0 * 17) % 257 - 128) / 63 }, [1, 1, k])
                    .asType(.bfloat16).asType(.float32)
                check(x, weight)
                check(-x, weight)
                // Reentry with changed buffers must not retain the first input
                // or promoted weight, then restore the original pair.
                check(x * 2, -weight)
                check(x, weight)
                let got = Inkling2HeadProjection.project(x, weight: weight).asArray(Float.self)
                let xv = x.asArray(Float.self)
                let wv = weight.asType(.float32).asArray(Float.self)
                for row in [0, 127, 2049, n - 1] {
                    let oracle = (0..<k).reduce(0.0) { $0 + Double(xv[$1]) * Double(wv[row*k+$1]) }
                    let magnitude = (0..<k).reduce(0.0) { $0 + abs(Double(xv[$1]) * Double(wv[row*k+$1])) }
                    XCTAssertEqual(Double(got[row]), oracle, accuracy: magnitude * 2e-6)
                }
            }
        }
    }

    func testStridedCheckpointViewAndEligibility() throws {
        try requireMetal()
        Device.withDefaultDevice(.gpu) {
            let k = 128, n = 4096
            let raw = (0..<n*k).map { Float(($0 * 29) % 251 - 125) / 64 }
            let weight = MLXArray(raw, [k, n]).asType(.bfloat16).T
            let x = MLXArray((0..<k).map { Float($0 % 31 - 15) / 16 }, [1, 1, k])
            XCTAssertTrue(Inkling2HeadProjection.supports(x, weight: weight))
            check(x, weight)
            XCTAssertFalse(Inkling2HeadProjection.supports(x, weight: weight.asType(.float32)))
            XCTAssertFalse(Inkling2HeadProjection.supports(.zeros([1, 2, k]), weight: weight))
            XCTAssertFalse(Inkling2HeadProjection.supports(.zeros([2, 1, k]), weight: weight))
            XCTAssertFalse(Inkling2HeadProjection.supports(.zeros([1, 1, 127]),
                weight: .zeros([n, 127], dtype: .bfloat16)))
            XCTAssertFalse(Inkling2HeadProjection.supports(x,
                weight: .zeros([n + 1, k], dtype: .bfloat16)))
        }
        Device.withDefaultDevice(.cpu) {
            XCTAssertFalse(Inkling2HeadProjection.supports(.zeros([1, 1, 128]),
                weight: .zeros([4096, 128], dtype: .bfloat16)))
        }
    }
}
