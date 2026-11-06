import Foundation
import XCTest

@testable import MLXLMCommon

final class JANGHTensorIndexPlanTests: XCTestCase {
    private typealias Plan = JANGHTensorIndexPlan
    private let file = "model-00001-of-00001.safetensors"
    private let prefix = "model.layers.0.mlp.switch_mlp."

    private struct Fixture {
        let contract: JANGHFormatContract
        var dimensions: [String: Plan.Dimensions]
        var index: [String: String]
        var shards: [String: Plan.ShardHeader]

        func validate() throws -> Plan {
            try Plan(contract: contract, dimensions: dimensions, weightMap: index, shards: shards)
        }
    }

    private func fixture() throws -> Fixture {
        var books: [String: Any] = [:]
        var quant: [String: Any] = [:]
        var dimensions: [String: Plan.Dimensions] = [:]
        var index: [String: String] = [:]
        var headers: [String: Plan.TensorHeader] = [:]
        var offset = 0
        for (role, bits) in [("gate_proj", 2), ("up_proj", 3), ("down_proj", 6)] {
            let module = prefix + role
            let center = Double((1 << bits) - 1) / 2
            books[String(bits)] = [
                "alpha": 0.25, "beta": 0,
                "levels": (0..<(1 << bits)).map { (Double($0) - center) * 0.25 },
            ]
            quant[module] = ["mode": "jangtq2", "bits": bits, "rotation": "hadamard32"]
            dimensions[module] = .init(experts: 2, input: 96, output: 3)
            for (suffix, dtype, shape, bytes) in [
                ("tq2_packed", "U32", [2, 3, 3 * bits], 2 * 3 * 3 * bits * 4),
                ("tq2_scales", "F16", [2, 3], 12),
            ] {
                let name = module + "." + suffix
                index[name] = file
                headers[name] = .init(dtype: dtype, shape: shape, start: offset, end: offset + bytes)
                offset += bytes
            }
        }
        let config: [String: Any] = [
            "jangtq": ["version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
                       "codebook_family": "odd-cubic", "rotation": "hadamard32", "codebooks": books],
            "quantization": quant,
        ]
        let contract = try JANGHFormatContract(configuration: JSONSerialization.data(withJSONObject: config))
        return Fixture(contract: contract, dimensions: dimensions, index: index,
                       shards: [file: .init(fileBytes: 4096 + offset, dataStart: 4096, tensors: headers)])
    }

    private func replace(
        _ fixture: inout Fixture, name: String, header: Plan.TensorHeader
    ) {
        let shard = fixture.shards[file]!
        var headers = shard.tensors
        headers[name] = header
        fixture.shards[file] = .init(fileBytes: shard.fileBytes, dataStart: shard.dataStart, tensors: headers)
    }

    func testMixedBitLocatorsWithoutTensorAllocation() throws {
        let f = try fixture()
        let plan = try f.validate()
        XCTAssertEqual(plan.projections.count, 3)
        for (role, bits) in [("gate_proj", 2), ("up_proj", 3), ("down_proj", 6)] {
            let banks = try XCTUnwrap(plan.projections[prefix + role])
            XCTAssertEqual(banks.packed.byteCount, 2 * 3 * 3 * bits * 4)
            XCTAssertEqual(banks.scales.byteCount, 12)
            XCTAssertEqual(banks.packed.shard, file)
            XCTAssertTrue(banks.packed.fileOffset.isMultiple(of: 4))
        }
    }

    func testMissingUnownedAndDuplicateBanks() throws {
        var f = try fixture()
        f.index.removeValue(forKey: prefix + "up_proj.tq2_scales")
        XCTAssertThrowsError(try f.validate())
        f = try fixture()
        let other = "other.safetensors"
        f.index["ordinary.weight"] = other
        f.shards[other] = f.shards[file]
        XCTAssertThrowsError(try f.validate())
        f = try fixture()
        f.shards.removeAll()
        XCTAssertThrowsError(try f.validate())
        f = try fixture()
        let extra = "extra.tq2_packed"
        f.index[extra] = file
        XCTAssertThrowsError(try f.validate())
    }

    func testBoundsAlignmentOverlapAndOverflow() throws {
        let name = prefix + "gate_proj.tq2_packed"
        for replacement in [
            Plan.TensorHeader(dtype: "U32", shape: [2, 3, 6], start: -1, end: 143),
            .init(dtype: "U32", shape: [2, 3, 6], start: 1, end: 145),
            .init(dtype: "U32", shape: [2, 3, 6], start: 0, end: Int.max),
            .init(dtype: "U32", shape: [Int.max, 3, 6], start: 0, end: 144),
            .init(dtype: "U32", shape: [2, 3, 6], start: 0, end: 140),
        ] {
            var f = try fixture()
            replace(&f, name: name, header: replacement)
            XCTAssertThrowsError(try f.validate())
        }
        var f = try fixture()
        let shard = f.shards[file]!
        f.shards[file] = .init(fileBytes: shard.fileBytes + 1, dataStart: 4097, tensors: shard.tensors)
        XCTAssertThrowsError(try f.validate())
        f = try fixture()
        replace(&f, name: "ordinary.weight", header: .init(dtype: "F16", shape: [2], start: 0, end: 4))
        XCTAssertThrowsError(try f.validate())
    }

    func testWrongArchitectureGeometryDTypeAndLegacyCollision() throws {
        var f = try fixture()
        f.dimensions.removeValue(forKey: prefix + "down_proj")
        XCTAssertThrowsError(try f.validate())
        f = try fixture()
        f.dimensions[prefix + "gate_proj"] = .init(experts: 2, input: 95, output: 3)
        XCTAssertThrowsError(try f.validate())
        f = try fixture()
        replace(&f, name: prefix + "gate_proj.tq2_packed",
                header: .init(dtype: "I32", shape: [2, 3, 6], start: 0, end: 144))
        XCTAssertThrowsError(try f.validate())
        for suffix in [".tq_packed", ".tq_norms", ".weight"] {
            f = try fixture()
            f.index[prefix + "gate_proj" + suffix] = file
            XCTAssertThrowsError(try f.validate())
        }
    }

    func testRejectsUnsafeShardPathsAndMissingHeader() throws {
        for path in ["../model.safetensors", "/model.safetensors", "sub/model.safetensors", "model.bin"] {
            var f = try fixture()
            let original = f.shards[file]!
            f.shards = [path: original]
            f.index = f.index.mapValues { _ in path }
            XCTAssertThrowsError(try f.validate())
        }
        var f = try fixture()
        let original = f.shards[file]!
        var headers = original.tensors
        headers.removeValue(forKey: prefix + "gate_proj.tq2_scales")
        f.shards[file] = .init(fileBytes: original.fileBytes, dataStart: original.dataStart, tensors: headers)
        XCTAssertThrowsError(try f.validate())
    }
    func testInvalidFileExtentAndZeroCustomDimension() throws {
        for (fileBytes, dataStart) in [(4095, 4096), (-1, 4096), (8192, 7)] {
            var f = try fixture()
            f.shards[file] = .init(fileBytes: fileBytes, dataStart: dataStart, tensors: f.shards[file]!.tensors)
            XCTAssertThrowsError(try f.validate())
        }
        var f = try fixture()
        replace(&f, name: prefix + "gate_proj.tq2_packed",
                header: .init(dtype: "U32", shape: [0, 3, 6], start: 0, end: 0))
        XCTAssertThrowsError(try f.validate())
    }

    func testPackedAndScalesMayResideInDifferentShards() throws {
        var f = try fixture()
        let original = f.shards[file]!
        let scalesFile = "scales.safetensors"
        let scaleHeaders = original.tensors.filter { $0.key.hasSuffix(".tq2_scales") }
        let packedHeaders = original.tensors.filter { $0.key.hasSuffix(".tq2_packed") }
        f.shards[file] = .init(fileBytes: original.fileBytes, dataStart: original.dataStart, tensors: packedHeaders)
        f.shards[scalesFile] = .init(fileBytes: original.fileBytes, dataStart: original.dataStart, tensors: scaleHeaders)
        for name in scaleHeaders.keys { f.index[name] = scalesFile }
        let plan = try f.validate()
        XCTAssertEqual(plan.projections.count, 3)
        XCTAssertEqual(plan.projections[prefix + "gate_proj"]?.packed.shard, file)
        XCTAssertEqual(plan.projections[prefix + "gate_proj"]?.scales.shard, scalesFile)
    }

}
