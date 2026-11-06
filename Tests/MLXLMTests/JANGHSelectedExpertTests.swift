import Cmlx
import CmlxGraphShim
import Foundation
import MLX
import XCTest
@testable import MLXLMCommon

final class JANGHSelectedExpertTests: XCTestCase {
    private let parent = "model.layers.0.mlp.switch_mlp"
    private let ids: [UInt32] = [0, 3, 5, 9, 15, 1, 3, 14]

    private struct Fixture {
        let directory: URL
        let source: JANGHMappedBanks.SourceLease
        let packedBytes: Int
    }

    private func fixture(
        bits: [Int] = [2, 3, 4], inputRotation: String = "none",
        downRotation: String = "none", experts: Int = 16, width: Int = 64, hidden: Int = 32
    ) throws -> Fixture {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var books: [String: Any] = [:], quant: [String: Any] = [:], headers: [String: Any] = [:]
        var map: [String: String] = [:]
        var dimensions: [String: JANGHTensorIndexPlan.Dimensions] = [:]
        var payload = Data(), packedBytes = 0
        for (roleIndex, role) in ["gate_proj", "up_proj", "down_proj"].enumerated() {
            let b = bits[roleIndex], input = roleIndex == 2 ? hidden : width
            let output = roleIndex == 2 ? width : hidden
            let module = parent + "." + role
            let alpha = 0.125, beta = 0.0001
            books[String(b)] = ["alpha": alpha, "beta": beta, "levels": (0..<(1 << b)).map { code in
                let u = Double(code) - Double((1 << b) - 1) / 2
                return Double(Float(u * (alpha + beta * u * u)))
            }]
            quant[module] = ["mode": "jangtq2", "bits": b,
                             "rotation": roleIndex == 2 ? downRotation : inputRotation]
            dimensions[module] = .init(experts: experts, input: input, output: output)
            var words = [UInt32](repeating: 0, count: experts * output * input * b / 32)
            for i in 0..<(experts * output * input) {
                let code = UInt32((i * 17 + i / input * 7 + roleIndex * 11) % (1 << b))
                for bit in 0..<b where (code >> bit) & 1 != 0 {
                    let offset = i * b + bit
                    words[offset / 32] |= 1 << (offset % 32)
                }
            }
            let scales = (0..<(experts * output)).map { Float16(Float($0 % 7 + 1) / 8) }
            let packed = words.withUnsafeBytes { Data($0) }
            if roleIndex == 0 { packedBytes = packed.count }
            for (suffix, dtype, shape, bytes) in [
                ("tq2_packed", "U32", [experts, output, input * b / 32], packed),
                ("tq2_scales", "F16", [experts, output], scales.withUnsafeBytes { Data($0) }),
            ] {
                let name = module + "." + suffix
                headers[name] = ["dtype": dtype, "shape": shape,
                                 "data_offsets": [payload.count, payload.count + bytes.count]]
                map[name] = "model.safetensors"
                payload.append(bytes)
            }
        }
        var header = try JSONSerialization.data(withJSONObject: headers, options: .sortedKeys)
        guard header.count <= 4088 else { throw NSError(domain: "fixture", code: 1) }
        header.append(Data(repeating: 0x20, count: 4088 - header.count))
        var length = UInt64(header.count).littleEndian
        var file = withUnsafeBytes(of: &length) { Data($0) }
        file.append(header); file.append(payload)
        try file.write(to: directory.appendingPathComponent("model.safetensors"))
        try JSONSerialization.data(withJSONObject: ["weight_map": map]).write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))
        let config: [String: Any] = ["jangtq": ["version": 2, "packing": "lsb-bitstream",
            "scale_dtype": "float16", "codebook_family": "odd-cubic", "rotation": inputRotation,
            "codebooks": books], "quantization": quant]
        let contract = try JANGHFormatContract(configuration: JSONSerialization.data(withJSONObject: config))
        let metadata = try JANGHHeaderAdapter.read(directory: directory, indexName: "model.safetensors.index.json")
        let source = try JANGHMappedBanks.SourceLease(directory: directory, metadata: metadata,
                                                     contract: contract, dimensions: dimensions)
        return Fixture(directory: directory, source: source, packedBytes: packedBytes)
    }

    private func kernel(_ owner: JANGHExpertMappedBanks) throws -> JANGHSelectedExpertDecode {
        try JANGHSelectedExpertDecode(owner: owner, gateModule: parent + ".gate_proj",
                                     upModule: parent + ".up_proj", downModule: parent + ".down_proj")
    }

    func testLazyOwnerMapsOnlySelectedExpertSpansAndBoundsItsOwnCache() throws {
        try MLXMetalTestLock.withLock {
            let f = try fixture(experts: 64, width: 512, hidden: 256)
            defer { try? FileManager.default.removeItem(at: f.directory) }
            let before = MLX.Memory.activeMemory
            let owner = try JANGHExpertMappedBanks(source: f.source, cacheByteLimit: 256 * 1024)
            XCTAssertEqual(MLX.Memory.activeMemory, before)
            XCTAssertEqual(owner.cachedOwnedMappedBytes, 0)
            let s = try owner.selection(module: parent + ".gate_proj", expertIDs: ids)
            XCTAssertEqual(s.expertIDs, ids)
            XCTAssertTrue(s.packed[1] === s.packed[6])
            XCTAssertLessThan(s.mappedPackedBytes, f.packedBytes / 2)
            XCTAssertLessThanOrEqual(owner.cachedOwnedMappedBytes, 256 * 1024)
            XCTAssertGreaterThan(MLX.Memory.activeMemory, before)
            XCTAssertThrowsError(try owner.selection(module: parent + ".gate_proj", expertIDs: []))
            XCTAssertThrowsError(try owner.selection(module: parent + ".gate_proj", expertIDs: Array(repeating: 0, count: 14)))
            XCTAssertThrowsError(try owner.selection(module: parent + ".gate_proj", expertIDs: Array(repeating: 64, count: 8)))
            XCTAssertThrowsError(try owner.selection(module: "unknown", expertIDs: ids))
            owner.removeAllCachedViews()
            XCTAssertEqual(owner.cachedOwnedMappedBytes, 0)
            XCTAssertEqual(s.packed[0].shape, [1, 256, 32])
        }
    }

    func testWholeBankViewDiagnosticPreservesOffsetsWithoutAllocatingPackedCopies() throws {
        try MLXMetalTestLock.withLock {
            let f = try fixture(experts: 16, width: 512, hidden: 256)
            defer { try? FileManager.default.removeItem(at: f.directory) }
            let whole = try JANGHExpertMappedBanks(source: f.source, cacheByteLimit: 0,
                                                 storage: .wholeBankViews)
            let mapped = MLX.Memory.activeMemory
            for role in ["gate_proj", "up_proj", "down_proj"] {
                let module = parent + "." + role
                let views = try whole.selection(module: module, expertIDs: ids)
                XCTAssertEqual(MLX.Memory.activeMemory, mapped)
                XCTAssertTrue(views.packed[1] === views.packed[6])
                let independent = try JANGHExpertMappedBanks(source: f.source, cacheByteLimit: 0)
                let reference = try independent.selection(module: module, expertIDs: ids)
                for (a, b) in zip(views.packed, reference.packed) {
                    XCTAssertEqual(a.shape, b.shape)
                    XCTAssertEqual(a.asArray(UInt32.self), b.asArray(UInt32.self))
                }
                XCTAssertEqual(views.scales.asArray(Float16.self), reference.scales.asArray(Float16.self))
            }
            XCTAssertEqual(whole.cachedOwnedMappedBytes, 0)
        }
    }

    func testExpertCacheHitRefreshesRecencyAndEvictsOnlyOldestView() throws {
        try MLXMetalTestLock.withLock {
            let page = Int(getpagesize())
            let f = try fixture(experts: 16, width: page / 32, hidden: 256)
            defer { try? FileManager.default.removeItem(at: f.directory) }
            // Include the real page-prefix bytes, including on 16 KiB hosts.
            let pair = try XCTUnwrap(f.source.plan.projections[parent + ".gate_proj"])
            let expertSpan = pair.packed.byteCount / 16 + pair.packed.fileOffset % page
            let scaleSpan = pair.scales.byteCount + pair.scales.fileOffset % page
            let cap = scaleSpan + 2 * expertSpan
            let owner = try JANGHExpertMappedBanks(source: f.source, cacheByteLimit: cap,
                                                 storage: .stableFileMappings)
            func select(_ expert: UInt32) throws -> JANGHExpertMappedBanks.Selection {
                try owner.selection(module: parent + ".gate_proj", expertIDs: Array(repeating: expert, count: 8))
            }
            let first = try select(0)
            let second = try select(1)
            XCTAssertTrue(try select(0).packed[0] === first.packed[0])
            _ = try select(2)
            XCTAssertLessThanOrEqual(owner.cachedOwnedMappedBytes, cap)
            XCTAssertTrue(try select(0).packed[0] === first.packed[0])
            XCTAssertFalse(try select(1).packed[0] === second.packed[0])
            owner.removeAllCachedViews()
            XCTAssertEqual(owner.cachedOwnedMappedBytes, 0)
            XCTAssertFalse(try select(0).packed[0] === first.packed[0])
        }
    }

    func testSelectedDecodeMatchesProvenWholeBankCompositionAcrossRotationsAndDTypes() throws {
        try checkSelectedDecode(storage: .independentMappings)
    }

    func testTopTenAndMetalBufferBoundaryMatchWholeBanksWithoutClamp() throws {
        try MLXMetalTestLock.withLock {
            let f = try fixture(bits: [2, 3, 4], inputRotation: "hadamard32", downRotation: "hadamard32")
            defer { try? FileManager.default.removeItem(at: f.directory) }
            let owner = try JANGHExpertMappedBanks(source: f.source, cacheByteLimit: 32768)
            let whole = try JANGHMappedBanks(source: f.source)
            let reference = try JANGHRoutedDecodeBlock(banks: whole, parentModule: parent, activationLimit: nil)
            XCTAssertEqual(JANGHSelectedExpertDecode.maxRoutes, 12)
            for count in [1, 2, 8, 10, 12] {
                let selected = try JANGHSelectedExpertDecode(owner: owner,
                    gateModule: parent + ".gate_proj", upModule: parent + ".up_proj",
                    downModule: parent + ".down_proj", routes: count)
                let ids = (0 ..< count).map { UInt32(($0 * 7) % 16) }
                let gate = try owner.selection(module: parent + ".gate_proj", expertIDs: ids)
                let up = try owner.selection(module: parent + ".up_proj", expertIDs: ids)
                let down = try owner.selection(module: parent + ".down_proj", expertIDs: ids)
                let routes = MLXArray(ids, [1, count])
                let scores = MLXArray((0 ..< count).map { Float($0 + 1) / Float(count * (count + 1) / 2) }, [1, count])
                for dtype in [DType.float16, .bfloat16, .float32] {
                    let x = MLXArray((0 ..< 64).map { Float(($0 * 19) % 59 - 29) / 31 }, [1, 64]).asType(dtype)
                    let hidden = try selected.activatePreparedInput(selected.prepareInput(x), gate: gate, up: up, limit: nil)
                    let actual = try selected.projectPreparedHidden(hidden, down: down, scores: scores, outputDType: dtype)
                    let expected = try reference(x, indices: routes, scores: scores, outputDType: dtype)
                    let a = actual.asType(.float32).asArray(Float.self)
                    let e = expected.asType(.float32).asArray(Float.self)
                    let tolerance: Float = dtype == .bfloat16 ? 0.008 : (dtype == .float16 ? 0.002 : 0.0005)
                    for i in a.indices {
                        XCTAssertTrue(a[i].isFinite)
                        XCTAssertEqual(a[i], e[i], accuracy: max(0.00005, abs(e[i]) * tolerance),
                            "routes=\(count), dtype=\(dtype), column=\(i)")
                    }
                    XCTAssertThrowsError(try selected.projectPreparedHidden(hidden, down: down,
                        scores: MLXArray.ones([1, count + 1]), outputDType: dtype))
                }
            }
            for count in [0, 13, 14] {
                XCTAssertThrowsError(try JANGHSelectedExpertDecode(owner: owner,
                    gateModule: parent + ".gate_proj", upModule: parent + ".up_proj",
                    downModule: parent + ".down_proj", routes: count))
                XCTAssertThrowsError(try owner.selection(module: parent + ".gate_proj",
                    expertIDs: Array(repeating: 0, count: count)))
            }
        }
    }

    func testStableMappingDecodeMatchesWholeBanksAcrossRotationsAndDTypes() throws {
        try checkSelectedDecode(storage: .stableFileMappings)
    }

    private func checkSelectedDecode(storage: JANGHExpertMappedBanks.Storage) throws {
        try MLXMetalTestLock.withLock {
            for bits in [[2, 2, 2], [2, 3, 4], [6, 8, 4]] {
                for inputRotation in ["none", "hadamard32"] {
                    for downRotation in ["none", "hadamard32"] {
                        let f = try fixture(bits: bits, inputRotation: inputRotation, downRotation: downRotation)
                        defer { try? FileManager.default.removeItem(at: f.directory) }
                        let owner = try JANGHExpertMappedBanks(source: f.source, cacheByteLimit: 32768,
                                                             storage: storage)
                        let selected = try kernel(owner)
                        let whole = try JANGHMappedBanks(source: f.source)
                        let reference = try JANGHRoutedDecodeBlock(banks: whole, parentModule: parent, activationLimit: 0.75)
                        let gate = try owner.selection(module: parent + ".gate_proj", expertIDs: ids)
                        let up = try owner.selection(module: parent + ".up_proj", expertIDs: ids)
                        let down = try owner.selection(module: parent + ".down_proj", expertIDs: ids)
                        let routes = MLXArray(ids, [1, 8])
                        let scores = MLXArray([Float(0.05), 0.10, 0.15, 0.20, 0.10, 0.05, 0.15, 0.20], [1, 8])
                        for inputDType in [DType.float16, .bfloat16, .float32] {
                            let x = MLXArray((0..<64).map { Float(($0 * 19) % 59 - 29) / 31 }, [1, 64]).asType(inputDType)
                            let h = try selected.activatePreparedInput(selected.prepareInput(x), gate: gate, up: up, limit: 0.75)
                            for outputDType in [DType.float16, .bfloat16, .float32] {
                                let actual = try selected.projectPreparedHidden(h, down: down, scores: scores, outputDType: outputDType)
                                let expected = try reference(x, indices: routes, scores: scores, outputDType: outputDType)
                                let a = actual.asType(.float32).asArray(Float.self)
                                let e = expected.asType(.float32).asArray(Float.self)
                                XCTAssertEqual(a.count, 64)
                                let tolerance: Float = outputDType == .bfloat16 ? 0.008 : (outputDType == .float16 ? 0.002 : 0.0005)
                                for i in a.indices {
                                    XCTAssertTrue(a[i].isFinite)
                                    XCTAssertEqual(a[i], e[i], accuracy: max(0.00005, abs(e[i]) * tolerance),
                                                   "bits=\(bits) rotations=\(inputRotation)/\(downRotation) input=\(inputDType) output=\(outputDType) row=\(i)")
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    func testSelectionsCannotCrossModelOwnersEvenWithIdenticalMetadata() throws {
        try MLXMetalTestLock.withLock {
            let f = try fixture()
            defer { try? FileManager.default.removeItem(at: f.directory) }
            let a = try JANGHExpertMappedBanks(source: f.source, cacheByteLimit: 0)
            let b = try JANGHExpertMappedBanks(source: f.source, cacheByteLimit: 0)
            let ka = try kernel(a), kb = try kernel(b)
            let x = MLXArray.ones([1, 64])
            let ga = try a.selection(module: parent + ".gate_proj", expertIDs: ids)
            let ua = try a.selection(module: parent + ".up_proj", expertIDs: ids)
            let ub = try b.selection(module: parent + ".up_proj", expertIDs: ids)
            XCTAssertThrowsError(try ka.activatePreparedInput(x, gate: ga, up: ub, limit: nil))
            let h = try ka.activatePreparedInput(x, gate: ga, up: ua, limit: nil)
            let db = try b.selection(module: parent + ".down_proj", expertIDs: ids)
            XCTAssertThrowsError(try kb.projectPreparedHidden(h, down: db, scores: MLXArray.ones([1, 8]), outputDType: .float32))
        }
    }

    func testCachedViewsRevalidateTheSourceBeforeReuse() throws {
        try MLXMetalTestLock.withLock {
            let f = try fixture()
            defer { try? FileManager.default.removeItem(at: f.directory) }
            let owner = try JANGHExpertMappedBanks(source: f.source, cacheByteLimit: 1 << 20)
            let held = try owner.selection(module: parent + ".gate_proj", expertIDs: ids)
            let path = f.directory.appendingPathComponent("model.safetensors")
            let bytes = try Data(contentsOf: path)
            try bytes.write(to: path, options: .atomic)
            XCTAssertThrowsError(try owner.selection(module: parent + ".gate_proj", expertIDs: ids))
            // Existing allocations retain the original inode and remain readable.
            XCTAssertEqual(held.packed[0].asArray(UInt32.self).count, 128)
        }
    }

    func testZeroCacheSelectionsSurviveOwnerReleaseAndUnlink() throws {
        try MLXMetalTestLock.withLock {
            let f = try fixture()
            defer { try? FileManager.default.removeItem(at: f.directory) }
            var owner: JANGHExpertMappedBanks? = try JANGHExpertMappedBanks(source: f.source, cacheByteLimit: 0)
            let selection = try owner!.selection(module: parent + ".gate_proj", expertIDs: ids)
            XCTAssertEqual(owner!.cachedOwnedMappedBytes, 0)
            let expected = selection.packed[0].asArray(UInt32.self)
            owner = nil
            try FileManager.default.removeItem(at: f.directory.appendingPathComponent("model.safetensors"))
            XCTAssertEqual(selection.packed[0].asArray(UInt32.self), expected)
        }
    }

    func testStableMappedSelectionsSurviveOwnerReleaseAndUnlink() throws {
        try MLXMetalTestLock.withLock {
            let f = try fixture()
            defer { try? FileManager.default.removeItem(at: f.directory) }
            var owner: JANGHExpertMappedBanks? = try JANGHExpertMappedBanks(
                source: f.source, cacheByteLimit: 0, storage: .stableFileMappings)
            let selection = try owner!.selection(module: parent + ".gate_proj", expertIDs: ids)
            let expected = selection.packed[1].asArray(UInt32.self)
            owner = nil
            try FileManager.default.removeItem(at: f.directory.appendingPathComponent("model.safetensors"))
            XCTAssertEqual(selection.packed[1].asArray(UInt32.self), expected)
        }
    }

    func testStableMappingRejectsInvalidViewsAndChangedSourceWithoutWholeFileGPUAllocation() throws {
        try MLXMetalTestLock.withLock {
            let f = try fixture()
            defer { try? FileManager.default.removeItem(at: f.directory) }
            let path = f.directory.appendingPathComponent("model.safetensors")
            let file = try FileHandle(forReadingFrom: path)
            defer { try? file.close() }
            var handle: UnsafeMutableRawPointer?
            defer { vmlx_mapped_file_release(&handle) }
            let before = MLX.Memory.activeMemory
            XCTAssertEqual(try withError { vmlx_mapped_file_create(file.fileDescriptor, &handle) }, 0)
            XCTAssertEqual(MLX.Memory.activeMemory, before)
            func reject(offset: UInt64, length: Int, shape: [Int32]) {
                var dimensions = shape
                var raw = mlx_array_new()
                defer { mlx_array_free(raw) }
                XCTAssertThrowsError(try withError {
                    vmlx_mapped_file_array(&raw.ctx, handle, offset, length, &dimensions,
                                           Int32(dimensions.count), Int32(DType.uint32.cmlxDtype.rawValue))
                })
            }
            reject(offset: UInt64.max, length: 64, shape: [16])
            reject(offset: 4097, length: 64, shape: [16])
            reject(offset: 4096, length: 60, shape: [16])
            reject(offset: 4096, length: 64, shape: [-1])
            reject(offset: 4096, length: 64, shape: [Int32.max, Int32.max, Int32.max])
            try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSinceNow: 10)],
                                                  ofItemAtPath: path.path)
            reject(offset: 4096, length: 64, shape: [16])
            XCTAssertEqual(MLX.Memory.activeMemory, before)
        }
    }

    func testDiagnosticLayerMatchesDecodeAndPrefillWithoutRetainingWholeBanks() throws {
        try MLXMetalTestLock.withLock {
            let f = try fixture(inputRotation: "hadamard32", downRotation: "hadamard32", hidden: 64)
            defer { try? FileManager.default.removeItem(at: f.directory) }
            let owner = try JANGHExpertMappedBanks(source: f.source, cacheByteLimit: 32768)
            let layer = try JANGHSelectedRoutedExpertLayer(source: f.source, owner: owner,
                parentModule: parent, inputDimensions: 64, activationLimit: 0.75)
            let banks = try JANGHMappedBanks(source: f.source)
            let reference = try JANGHRoutedDecodeBlock(banks: banks, parentModule: parent, activationLimit: 0.75)
            let tracked = mlx_safetensors_mmap_tracked_buffer_bytes()
            for tokens in [1, 2, 9] {
                let input = MLXArray((0..<(tokens * 64)).map { Float(($0 * 19) % 59 - 29) / 31 }, [tokens, 64]).asType(.bfloat16)
                let routes = MLXArray(Array(repeating: ids, count: tokens).flatMap { $0 }, [tokens, 8])
                let scores = MLXArray.ones([tokens, 8]) * Float(0.125)
                let actual = try layer.routed(input, indices: routes, scores: scores)
                let expected = try reference.routed(input, indices: routes, scores: scores, outputDType: input.dtype)
                let a = actual.asType(.float32).asArray(Float.self)
                let e = expected.asType(.float32).asArray(Float.self)
                for i in a.indices {
                    XCTAssertEqual(a[i], e[i], accuracy: max(0.00005, abs(e[i]) * 0.008), "tokens=\(tokens) row=\(i)")
                }
                XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(), tracked,
                               "Prefill must release temporary whole-bank resources after evaluation")
            }
            let x = MLXArray.ones([1,64])
            XCTAssertThrowsError(try layer.routed(x, indices: MLXArray([UInt32(0)], [1,1]), scores: MLXArray.ones([1,1])))
            let other = try fixture()
            defer { try? FileManager.default.removeItem(at: other.directory) }
            XCTAssertThrowsError(try JANGHSelectedRoutedExpertLayer(source: other.source, owner: owner,
                parentModule: parent, inputDimensions: 64, activationLimit: nil))
        }
    }

    func testPartialBankMappingRefusesUnknownOrEmptySubsets() throws {
        try MLXMetalTestLock.withLock {
            let f = try fixture()
            defer { try? FileManager.default.removeItem(at: f.directory) }
            XCTAssertThrowsError(try JANGHMappedBanks(source: f.source, modules: []))
            XCTAssertThrowsError(try JANGHMappedBanks(source: f.source, modules: ["unknown"]))
            let only = try JANGHMappedBanks(source: f.source, modules: [parent + ".gate_proj"])
            XCTAssertNoThrow(try only.projection(parent + ".gate_proj"))
            XCTAssertThrowsError(try only.projection(parent + ".up_proj"))
        }
    }

}
