//
//  SaveTests.swift
//
//
//  Created by Rounak Jain on 4/2/24.
//

import MLX
import Cmlx
import Darwin
import XCTest

// All access to the recorded callbacks is protected by the lock.
private final class ProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var progress = [LoadProgress]()

    func record(_ progress: LoadProgress) {
        lock.withLock {
            self.progress.append(progress)
        }
    }

    var reported: [LoadProgress] {
        lock.withLock { progress }
    }

    var values: [Double] {
        reported.map { $0.fractionCompleted }
    }

    /// Fractions reported for a single file, in order.
    func values(for url: URL) -> [Double] {
        reported.filter { $0.url == url }.map { $0.fractionCompleted }
    }

    /// Aggregate fraction across every file seen, by bytes.
    var aggregateFraction: Double {
        var completed = [URL: Int64]()
        var total = [URL: Int64]()
        for progress in reported {
            completed[progress.url] = progress.completedUnitCount
            total[progress.url] = progress.totalUnitCount
        }
        let totalBytes = total.values.reduce(0, +)
        guard totalBytes > 0 else { return 0 }
        return Double(completed.values.reduce(0, +)) / Double(totalBytes)
    }
}

final class SaveTests: XCTestCase {

    let temporaryPath = FileManager.default.temporaryDirectory.appending(
        path: UUID().uuidString,
        directoryHint: .isDirectory
    )

    override func setUpWithError() throws {
        prepareMLXMetallibForTests()
        setDefaultDevice()
        try FileManager.default.createDirectory(
            at: temporaryPath,
            withIntermediateDirectories: false
        )
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: temporaryPath)
    }

    public func testSaveArrays() throws {
        let safetensorsPath = temporaryPath.appending(
            path: "arrays.safetensors",
            directoryHint: .notDirectory
        )

        let arrays: [String: MLXArray] = [
            "foo": MLX.ones([1, 2]),
            "bar": MLX.zeros([2, 1]),
        ]

        try MLX.save(arrays: arrays, url: safetensorsPath)

        let loadedArrays = try MLX.loadArrays(url: safetensorsPath)
        XCTAssertEqual(loadedArrays.keys.sorted(), arrays.keys.sorted())

        assertEqual(try XCTUnwrap(loadedArrays["foo"]), try XCTUnwrap(arrays["foo"]))
        assertEqual(try XCTUnwrap(loadedArrays["bar"]), try XCTUnwrap(arrays["bar"]))
    }

    public func testLoadArraysProgressReportsThroughEvaluation() throws {
        try MLX.Device.withDefaultDevice(.cpu) {
            let safetensorsPath = temporaryPath.appending(
                path: "arrays.safetensors",
                directoryHint: .notDirectory
            )

            let arrays: [String: MLXArray] = [
                "foo": MLX.ones([128, 128]),
                "bar": MLX.zeros([64, 256]),
            ]
            try MLX.save(arrays: arrays, url: safetensorsPath)

            let recorder = ProgressRecorder()
            let loadedArrays = try MLX.loadArrays(
                url: safetensorsPath
            ) { @Sendable progress in
                recorder.record(progress)
            }

            assertEqual(try XCTUnwrap(loadedArrays["foo"]), try XCTUnwrap(arrays["foo"]))
            assertEqual(try XCTUnwrap(loadedArrays["bar"]), try XCTUnwrap(arrays["bar"]))

            let fractions = recorder.values
            XCTAssertGreaterThan(fractions.count, 1)
            XCTAssertEqual(fractions.first, 0)
            XCTAssertEqual(fractions.last, 1)
            XCTAssertEqual(fractions, fractions.sorted())
        }
    }

    public func testLoadArraysProgressFailsOnTruncatedTensorData() throws {
        try MLX.Device.withDefaultDevice(.cpu) {
            let safetensorsPath = temporaryPath.appending(
                path: "truncated.safetensors",
                directoryHint: .notDirectory
            )

            let arrays: [String: MLXArray] = [
                "foo": MLX.ones([128, 128]),
                "bar": MLX.zeros([64, 256]),
            ]
            try MLX.save(arrays: arrays, url: safetensorsPath)

            var data = try Data(contentsOf: safetensorsPath)
            data.removeLast(32)
            try data.write(to: safetensorsPath)

            // A truncated file has to be reported either eagerly, while the header is
            // parsed (mlx >= 0.32.1 validates the tensor data offsets against the size of
            // the file), or lazily, when the arrays are evaluated and the short read is
            // turned into an error (ml-explore/mlx-c#130).
            do {
                let loadedArrays = try MLX.loadArrays(url: safetensorsPath) { _ in }
                try checkedEval(Array(loadedArrays.values) as [Any])
                XCTFail("a truncated safetensors file must not load successfully")
            } catch {
                // expected
            }
        }
    }

    /// mlx validates the tensor data offsets against the size of the file while it parses
    /// the header, so a file that is _already_ truncated fails eagerly. A file that is
    /// truncated after the header is parsed can only be caught when the lazy arrays are
    /// evaluated and the read comes up short -- `mlx_io_reader` turns that into an error
    /// rather than leaving the destination buffer uninitialized (ml-explore/mlx-c#130).
    public func testLoadFailsWhenFileIsTruncatedBeforeEvaluation() throws {
        try MLX.Device.withDefaultDevice(.cpu) {
            let safetensorsPath = temporaryPath.appending(
                path: "truncated-late.safetensors",
                directoryHint: .notDirectory
            )

            let arrays: [String: MLXArray] = [
                "foo": MLX.ones([128, 128]),
                "bar": MLX.zeros([64, 256]),
            ]
            try MLX.save(arrays: arrays, url: safetensorsPath)

            // the header is parsed and validated against the size of the file here, but
            // the tensor data is only read once the lazy arrays are evaluated
            let recorder = ProgressRecorder()
            let loadedArrays = try MLX.loadArrays(url: safetensorsPath) {
                @Sendable in recorder.record($0)
            }
            let size = try XCTUnwrap(recorder.reported.first?.totalUnitCount)

            // ... so truncate the file out from under them. `truncate()` shortens the
            // inode the reader already has open, unlike a rewrite which may replace it.
            XCTAssertEqual(
                truncate(safetensorsPath.path(percentEncoded: false), off_t(size - 32)), 0)

            do {
                try checkedEval(Array(loadedArrays.values) as [Any])
                XCTFail("evaluating arrays read from a truncated file must fail")
            } catch {
                // expected
            }

            // the bytes that were read are still accounted for, monotonically, and the
            // aggregate stops short of the original size of the file
            let fractions = recorder.values
            XCTAssertEqual(fractions, fractions.sorted())
            XCTAssertEqual(fractions.first, 0)
            XCTAssertLessThan(try XCTUnwrap(fractions.last), 1)
        }
    }

    public func testScopedLoadProgressHandler() throws {
        try MLX.Device.withDefaultDevice(.cpu) {
            let safetensorsPath = temporaryPath.appending(
                path: "scoped.safetensors",
                directoryHint: .notDirectory
            )

            let arrays: [String: MLXArray] = [
                "foo": MLX.ones([128, 128]),
                "bar": MLX.zeros([64, 256]),
            ]
            try MLX.save(arrays: arrays, url: safetensorsPath)

            let recorder = ProgressRecorder()

            // note: the plain loadArrays(url:) -- no progress handler passed at the call site
            let loadedArrays = try withLoadProgressHandler({ @Sendable in recorder.record($0) }) {
                let loadedArrays = try MLX.loadArrays(url: safetensorsPath)
                MLX.eval(Array(loadedArrays.values))
                return loadedArrays
            }

            assertEqual(try XCTUnwrap(loadedArrays["foo"]), try XCTUnwrap(arrays["foo"]))

            let fractions = recorder.values
            XCTAssertGreaterThan(fractions.count, 1)
            XCTAssertEqual(fractions.first, 0)
            XCTAssertEqual(fractions.last, 1)
            XCTAssertEqual(fractions, fractions.sorted())
            XCTAssertEqual(Set(recorder.reported.map(\.url)), [safetensorsPath])
        }
    }

    public func testScopedLoadProgressHandlerIsScoped() throws {
        try MLX.Device.withDefaultDevice(.cpu) {
            let safetensorsPath = temporaryPath.appending(
                path: "unscoped.safetensors",
                directoryHint: .notDirectory
            )
            try MLX.save(arrays: ["foo": MLX.ones([128, 128])], url: safetensorsPath)

            let recorder = ProgressRecorder()
            withLoadProgressHandler({ @Sendable in recorder.record($0) }) {
            }

            // outside the scope nothing is reported
            let loadedArrays = try MLX.loadArrays(url: safetensorsPath)
            MLX.eval(Array(loadedArrays.values))

            XCTAssertTrue(recorder.reported.isEmpty)
        }
    }

    public func testScopedLoadProgressAggregatesAcrossFiles() throws {
        try MLX.Device.withDefaultDevice(.cpu) {
            let shards = try (0 ..< 3).map { index -> URL in
                let url = temporaryPath.appending(
                    path: "shard-\(index).safetensors",
                    directoryHint: .notDirectory
                )
                try MLX.save(arrays: ["w\(index)": MLX.ones([64, 128])], url: url)
                return url
            }

            let recorder = ProgressRecorder()
            try withLoadProgressHandler({ @Sendable in recorder.record($0) }) {
                // this mimics a model loader: several shards loaded lazily, then evaluated
                var weights = [String: MLXArray]()
                for url in shards {
                    let (w, _) = try MLX.loadArraysAndMetadata(url: url)
                    weights.merge(w) { _, new in new }
                }
                MLX.eval(Array(weights.values))
            }

            XCTAssertEqual(Set(recorder.reported.map(\.url)), Set(shards))
            for url in shards {
                XCTAssertEqual(recorder.values(for: url).last, 1)
            }
            XCTAssertEqual(recorder.aggregateFraction, 1, accuracy: 1e-9)
        }
    }

    public func testProgressChunksLargeTensorReads() throws {
        let url = temporaryPath.appending(path: "large.safetensors")
        let array = MLXArray(Array(repeating: Float(3), count: 3 * 1024 * 1024))
        try MLX.save(arrays: ["large": array], metadata: ["format": "test"], url: url)

        let recorder = ProgressRecorder()
        let (loaded, metadata) = try MLX.loadArraysAndMetadata(url: url) {
            recorder.record($0)
        }
        let headerBytes = try XCTUnwrap(recorder.reported.last?.completedUnitCount)
        XCTAssertLessThan(headerBytes, Int64(array.nbytes))
        XCTAssertEqual(metadata, ["format": "test"])
        XCTAssertEqual(
            try XCTUnwrap(loaded["large"]).asArray(Float.self), array.asArray(Float.self))

        let reports = recorder.reported
        XCTAssertEqual(reports.first?.completedUnitCount, 0)
        XCTAssertEqual(reports.last?.fractionCompleted, 1)
        XCTAssertEqual(
            reports.map(\.completedUnitCount), reports.map(\.completedUnitCount).sorted())
        let tensorReports = reports.filter { $0.completedUnitCount > headerBytes }
        XCTAssertGreaterThanOrEqual(tensorReports.count, 3)
        for (previous, current) in zip(reports, reports.dropFirst()) {
            XCTAssertLessThanOrEqual(
                current.completedUnitCount - previous.completedUnitCount, 4 * 1024 * 1024)
        }
    }

    public func testProgressRetainsHandlerAndSkipsUnevaluatedArrays() throws {
        let url = temporaryPath.appending(path: "partial.safetensors")
        let values = Array(repeating: Float(7), count: 1024)
        try MLX.save(arrays: ["used": MLXArray(values), "unused": MLXArray(values)], url: url)

        let recorder = ProgressRecorder()
        let arrays = try withLoadProgressHandler({ recorder.record($0) }) {
            try MLX.loadArrays(url: url)
        }
        let headerBytes = try XCTUnwrap(recorder.reported.last?.completedUnitCount)
        XCTAssertEqual(try XCTUnwrap(arrays["used"]).asArray(Float.self), values)
        let partial = try XCTUnwrap(recorder.reported.last)
        XCTAssertEqual(
            partial.completedUnitCount, headerBytes + Int64(values.count * MemoryLayout<Float>.size)
        )
        XCTAssertLessThan(partial.fractionCompleted, 1)
        XCTAssertEqual(try XCTUnwrap(arrays["unused"]).asArray(Float.self), values)
        XCTAssertEqual(recorder.reported.last?.fractionCompleted, 1)
    }

    public func testNestedProgressHandlersRestoreAfterThrowAndExplicitHandlerWins() throws {
        enum TestError: Error { case expected }
        let url = temporaryPath.appending(path: "nested.safetensors")
        try MLX.save(arrays: ["weight": MLXArray([Float(1), 2])], url: url)
        let outer = ProgressRecorder()
        let inner = ProgressRecorder()
        let explicit = ProgressRecorder()

        try withLoadProgressHandler({ outer.record($0) }) {
            XCTAssertThrowsError(
                try withLoadProgressHandler({ inner.record($0) }) {
                    let arrays = try MLX.loadArrays(url: url)
                    try checkedEval(Array(arrays.values) as [Any])
                    throw TestError.expected
                })
            XCTAssertTrue(outer.reported.isEmpty)
            XCTAssertEqual(inner.reported.last?.fractionCompleted, 1)

            let explicitlyLoaded = try MLX.loadArrays(url: url) { explicit.record($0) }
            try checkedEval(Array(explicitlyLoaded.values) as [Any])
            XCTAssertTrue(outer.reported.isEmpty)
            XCTAssertEqual(explicit.reported.last?.fractionCompleted, 1)

            let arrays = try MLX.loadArrays(url: url)
            try checkedEval(Array(arrays.values) as [Any])
        }
        XCTAssertEqual(outer.reported.last?.fractionCompleted, 1)
    }

    public func testAsyncProgressScopeInheritsHandler() async throws {
        let url = temporaryPath.appending(path: "async.safetensors")
        try MLX.save(arrays: ["weight": MLXArray([Float(1), 2])], url: url)
        let recorder = ProgressRecorder()
        try await withLoadProgressHandler({ recorder.record($0) }) {
            try await Task {
                let arrays = try MLX.loadArrays(url: url)
                try checkedEval(Array(arrays.values) as [Any])
            }.value
        }
        XCTAssertEqual(recorder.reported.first?.fractionCompleted, 0)
        XCTAssertEqual(recorder.reported.last?.fractionCompleted, 1)
    }

    public func testSaveArray() throws {
        // single array npy file
        let path = temporaryPath.appending(
            path: "array.npy",
            directoryHint: .notDirectory
        )

        let array = MLX.ones([2, 4])

        try MLX.save(array: array, url: path)

        let loaded = try MLX.loadArray(url: path)

        assertEqual(array, loaded)
    }

    public func testSaveArraysData() throws {
        let arrays: [String: MLXArray] = [
            "foo": MLX.ones([1, 2]),
            "bar": MLX.zeros([2, 1]),
        ]

        let data = try saveToData(arrays: arrays)
        let loadedArrays = try loadArrays(data: data)
        XCTAssertEqual(loadedArrays.keys.sorted(), arrays.keys.sorted())

        assertEqual(try XCTUnwrap(loadedArrays["foo"]), try XCTUnwrap(arrays["foo"]))
        assertEqual(try XCTUnwrap(loadedArrays["bar"]), try XCTUnwrap(arrays["bar"]))
    }

    public func testSaveArraysMetadataData() throws {
        let arrays: [String: MLXArray] = [
            "foo": MLX.ones([1, 2]),
            "bar": MLX.zeros([2, 1]),
        ]
        let metadata = [
            "key": "value",
            "key2": "value2",
        ]

        let data = try saveToData(arrays: arrays, metadata: metadata)
        let (loadedArrays, loadedMetadata) = try loadArraysAndMetadata(data: data)
        XCTAssertEqual(loadedArrays.keys.sorted(), arrays.keys.sorted())

        assertEqual(try XCTUnwrap(loadedArrays["foo"]), try XCTUnwrap(arrays["foo"]))
        assertEqual(try XCTUnwrap(loadedArrays["bar"]), try XCTUnwrap(arrays["bar"]))
        XCTAssertEqual(loadedMetadata, metadata)
    }

    /// `loadArrays(data:)` seeks to the end to size its input, so the in-memory IO
    /// stream has to report the size of the whole buffer rather than an offset
    /// relative to the read position.
    public func testLoadFromDataLargerThanHeader() throws {
        // enough data that a SEEK_END mistake shows up as a truncated size
        let arrays: [String: MLXArray] = [
            "big": MLX.ones([64, 64])
        ]

        let data = try saveToData(arrays: arrays)
        XCTAssertGreaterThan(data.count, 8)

        let loaded = try loadArrays(data: data)
        assertEqual(try XCTUnwrap(loaded["big"]), try XCTUnwrap(arrays["big"]))
    }

    /// A truncated in-memory buffer must be reported as an error. The in-memory reader
    /// reports a read that runs past the end of the buffer as zero bytes so that
    /// `mlx_io_reader` throws instead of leaving the destination uninitialized
    /// (ml-explore/mlx-c#130).
    public func testLoadFromTruncatedDataFails() throws {
        try MLX.Device.withDefaultDevice(.cpu) {
            let arrays: [String: MLXArray] = [
                "big": MLX.ones([64, 64])
            ]

            var data = try saveToData(arrays: arrays)
            data.removeLast(32)

            do {
                let loaded = try loadArrays(data: data)
                try checkedEval(Array(loaded.values) as [Any])
                XCTFail("a truncated safetensors buffer must not load successfully")
            } catch {
                // expected
            }
        }
    }

}


extension SaveTests {
    public func testLoadSafetensorsExcludingKeysBeforeMaterialization() throws {
        let safetensorsPath = temporaryPath.appending(
            path: "filtered-arrays.safetensors",
            directoryHint: .notDirectory
        )
        // Keys must match the loader's layer pattern, or the zero checks go blind.
        let keepKey = "layers.44.input_layernorm.weight"
        let dropKey = "layers.44.mlp.switch_mlp.gate_proj.weight"
        // A dtype-aligned layout (`MLX.save` does not pad its header), so the
        // retired mapped loader would have mapped the kept tensor.
        try writeSparseFloat32Safetensors(
            at: safetensorsPath,
            firstKey: keepKey,
            firstValues: [1, 2, 3, 4],
            secondKey: dropKey,
            secondValues: [5, 6, 7, 8],
            gapBytes: 0,
            metadataJSON: #"{"owner":"selected-key-test"}"#)

        try withEnvironment("MLX_SAFETENSORS_MMAP", value: "1") {
            let trackedBefore = mlx_safetensors_mmap_tracked_buffer_bytes()
            let (loaded, metadata) = try MLX.loadArraysAndMetadata(
                url: safetensorsPath,
                excludingKeys: [dropKey],
                stream: .cpu)
            XCTAssertEqual(
                mlx_safetensors_mmap_advise_layer(1, 44),
                0,
                "a filtered load must not register layer regions")
            XCTAssertEqual(
                mlx_safetensors_mmap_tracked_buffer_bytes() - trackedBefore,
                0,
                "a filtered load must not keep a mapped Metal buffer")
            XCTAssertEqual(loaded.keys.sorted(), [keepKey])
            XCTAssertNil(loaded[dropKey])
            XCTAssertEqual(metadata["owner"], "selected-key-test")
            assertEqual(
                try XCTUnwrap(loaded[keepKey]),
                MLXArray([Float(1), 2, 3, 4]))
        }
    }

    public func testFilteredExactTensorBuffersLoadMapsNothing() throws {
        try withMLXMetallibForTests {
            let safetensorsPath = temporaryPath.appending(
                path: "filtered-exact-tensor-buffers.safetensors",
                directoryHint: .notDirectory
            )
            // Keys must match the loader's layer pattern, or the zero checks go blind.
            let keepKey = "layers.0.input_layernorm.weight"
            let dropKey = "layers.0.mlp.switch_mlp.gate_proj.weight"
            try writeSparseFloat32Safetensors(
                at: safetensorsPath,
                firstKey: keepKey,
                firstValues: [1, 2, 3, 4],
                secondKey: dropKey,
                secondValues: [5, 6, 7, 8],
                gapBytes: 1 << 20)

            // The legacy switches stay set: they must not bring a mapping back.
            try withEnvironment("MLX_SAFETENSORS_MMAP", value: "1") {
                try withEnvironment("MLX_SAFETENSORS_MMAP_TENSOR_BUFFERS", value: "0") {
                    let trackedBefore = mlx_safetensors_mmap_tracked_buffer_bytes()
                    let (loaded, _) = try MLX.loadArraysAndMetadata(
                        url: safetensorsPath,
                        excludingKeys: [dropKey],
                        exactTensorBuffers: true,
                        stream: .cpu)
                    let keep = try XCTUnwrap(loaded[keepKey])
                    XCTAssertNil(loaded[dropKey])

                    XCTAssertEqual(
                        mlx_safetensors_mmap_tracked_buffer_bytes() - trackedBefore,
                        0,
                        "a filtered load must not keep any part of the shard in a mapped Metal buffer")

                    let result = MLX.sum(keep, stream: .gpu)
                    XCTAssertEqual(result.item(Float.self), 10.0, accuracy: 0.001)
                }
            }
        }
    }

    public func testNamedMmapBankRegistersLayerAndFeedsGPUAfterColdAdvice() throws {
        try withMLXMetallibForTests {
            let path = temporaryPath.appending(
                path: "named-mmap-bank.bin",
                directoryHint: .notDirectory
            )
            let offset = 4096
            try writeUInt32Bank(at: path, offset: offset, values: Array(1 ... 8))

            let trackedBefore = mlx_safetensors_mmap_tracked_buffer_bytes()
            let bank = try mapNamedUInt32Bank(
                at: path,
                offset: offset,
                shape: [2, 2, 2],
                name: "model.layers.42.mlp.switch_mlp.gate_proj.tq2_packed")
            withExtendedLifetime(bank) {
                XCTAssertEqual(
                    mlx_safetensors_mmap_advise_layer(1, 42),
                    32,
                    "a named bank must register its 32 bytes as a layer 42 region")
                XCTAssertGreaterThan(
                    mlx_safetensors_mmap_tracked_buffer_bytes() - trackedBefore,
                    0,
                    "a named bank must keep its mapped Metal buffer tracked")
                XCTAssertEqual(
                    mlx_safetensors_mmap_advise_layer(0, 42),
                    32,
                    "cold advice must reach the registered region")
                XCTAssertEqual(
                    mlx_safetensors_mmap_advise_routed(0, 50),
                    16,
                    "half-cold routed advice must reach one of the bank's two experts")

                let result = MLX.sum(
                    (bank.asType(.float32) * MLXArray(3.0)) + MLXArray(1.0), stream: .gpu)
                XCTAssertEqual(result.item(Float.self), 116.0, accuracy: 0.001)
            }
        }
    }

    public func testSafetensorsLoadReadsDtypeUnalignedTensorBesideAlignedOne() throws {
        try withMLXMetallibForTests {
            let safetensorsPath = temporaryPath.appending(
                path: "unaligned-u32.safetensors",
                directoryHint: .notDirectory
            )
            // Keys must match the loader's layer pattern, or the zero checks go blind.
            let unalignedKey = "layers.97.mlp.gate_proj.weight"
            let alignedKey = "layers.98.mlp.gate_proj.weight"
            try writeMixedAlignmentUInt32Safetensors(
                at: safetensorsPath,
                unalignedKey: unalignedKey,
                unalignedValues: [1, 2, 3, 4],
                alignedKey: alignedKey,
                alignedValues: [5, 6, 7, 8],
                shape: [2, 2]
            )

            let trackedBefore = mlx_safetensors_mmap_tracked_buffer_bytes()
            let loadedArrays = try MLX.loadArrays(url: safetensorsPath, stream: .cpu)
            let unaligned = try XCTUnwrap(loadedArrays[unalignedKey])
            let aligned = try XCTUnwrap(loadedArrays[alignedKey])
            XCTAssertEqual(
                mlx_safetensors_mmap_advise_layer(1, 98),
                0,
                "the owned loader must not register the aligned tensor's layer")
            XCTAssertEqual(
                mlx_safetensors_mmap_tracked_buffer_bytes() - trackedBefore,
                0,
                "the owned loader must not keep a mapped Metal buffer")
            XCTAssertEqual(unaligned.asArray(UInt32.self), [1, 2, 3, 4])
            XCTAssertEqual(aligned.asArray(UInt32.self), [5, 6, 7, 8])

            let unalignedResult = MLX.sum(unaligned.asType(.float32), stream: .gpu)
            let alignedResult = MLX.sum(aligned.asType(.float32), stream: .gpu)
            XCTAssertEqual(unalignedResult.item(Float.self), 10.0, accuracy: 0.001)
            XCTAssertEqual(alignedResult.item(Float.self), 26.0, accuracy: 0.001)

            // mmap_file_region rejects a dtype-unaligned offset before it registers
            // anything. The named-bank tests map this layout at offset 4096.
            let bankPath = temporaryPath.appending(
                path: "unaligned-named-bank.bin",
                directoryHint: .notDirectory
            )
            try writeUInt32Bank(at: bankPath, offset: 4096, values: Array(1 ... 8))
            let bankTrackedBefore = mlx_safetensors_mmap_tracked_buffer_bytes()
            XCTAssertThrowsError(
                try mapNamedUInt32Bank(
                    at: bankPath,
                    offset: 4098,
                    shape: [1, 2, 2],
                    name: "model.layers.45.mlp.switch_mlp.gate_proj.tq2_packed")
            ) { error in
                XCTAssertTrue("\(error)".contains("dtype-unaligned offset"), "\(error)")
            }
            XCTAssertEqual(mlx_safetensors_mmap_advise_layer(1, 45), 0)
            XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes() - bankTrackedBefore, 0)
        }
    }

    public func testLegacyMmapEnvironmentNoLongerMapsSafetensors() throws {
        try withMLXMetallibForTests {
            let safetensorsPath = temporaryPath.appending(
                path: "legacy-mmap-environment.safetensors",
                directoryHint: .notDirectory
            )
            // Keys must match the loader's layer pattern, or the zero checks go blind.
            let firstKey = "layers.41.mlp.switch_mlp.gate_proj.tq_packed"
            let secondKey = "layers.41.mlp.switch_mlp.up_proj.tq_packed"
            try writeSparseFloat32Safetensors(
                at: safetensorsPath,
                firstKey: firstKey,
                firstValues: [1, 2, 3, 4],
                secondKey: secondKey,
                secondValues: [5, 6, 7, 8],
                gapBytes: 1 << 20)

            try withEnvironment("MLX_SAFETENSORS_MMAP", value: "1") {
                try withEnvironment("MLX_SAFETENSORS_MMAP_TENSOR_BUFFERS", value: "1") {
                    let trackedBefore = mlx_safetensors_mmap_tracked_buffer_bytes()
                    let loadedArrays = try MLX.loadArrays(url: safetensorsPath, stream: .cpu)
                    let first = try XCTUnwrap(loadedArrays[firstKey])
                    let second = try XCTUnwrap(loadedArrays[secondKey])

                    XCTAssertEqual(
                        mlx_safetensors_mmap_advise_layer(1, 41),
                        0,
                        "the legacy environment must not register layer regions")
                    XCTAssertEqual(
                        mlx_safetensors_mmap_tracked_buffer_bytes() - trackedBefore,
                        0,
                        "the legacy environment must not keep a mapped Metal buffer")

                    let result = MLX.sum(first + second, stream: .gpu)
                    XCTAssertEqual(result.item(Float.self), 36.0, accuracy: 0.001)
                }
            }

            // Positive control: the probes above can see layer 41. A named bank
            // there registers its 16 bytes and a tracked buffer.
            let bankPath = temporaryPath.appending(
                path: "legacy-mmap-environment-control.bin",
                directoryHint: .notDirectory
            )
            try writeUInt32Bank(at: bankPath, offset: 4096, values: [1, 2, 3, 4])
            let controlTrackedBefore = mlx_safetensors_mmap_tracked_buffer_bytes()
            let bank = try mapNamedUInt32Bank(
                at: bankPath,
                offset: 4096,
                shape: [1, 2, 2],
                name: "model.layers.41.mlp.switch_mlp.gate_proj.tq2_packed")
            withExtendedLifetime(bank) {
                XCTAssertEqual(mlx_safetensors_mmap_advise_layer(1, 41), 16)
                XCTAssertGreaterThan(
                    mlx_safetensors_mmap_tracked_buffer_bytes() - controlTrackedBefore, 0)
            }
        }
    }

    public func testExactTensorBuffersWithNoExcludedKeysMapsNothing() throws {
        try withMLXMetallibForTests {
            let url = temporaryPath.appending(path: "exact-no-exclusions.safetensors")
            // Keys must match the loader's layer pattern, or the zero checks go blind.
            let firstKey = "layers.0.mlp.switch_mlp.gate_proj.tq_packed"
            let secondKey = "layers.0.mlp.switch_mlp.up_proj.tq_packed"
            try writeSparseFloat32Safetensors(
                at: url, firstKey: firstKey, firstValues: [1, 2, 3, 4],
                secondKey: secondKey, secondValues: [5, 6, 7, 8], gapBytes: 1 << 20)
            // The legacy switches stay set: they must not bring a mapping back.
            try withEnvironment("MLX_SAFETENSORS_MMAP", value: "1") {
                try withEnvironment("MLX_SAFETENSORS_MMAP_TENSOR_BUFFERS", value: "0") {
                    let before = mlx_safetensors_mmap_tracked_buffer_bytes()
                    let (arrays, _) = try MLX.loadArraysAndMetadata(
                        url: url, excludingKeys: [], exactTensorBuffers: true)
                    let tracked = mlx_safetensors_mmap_tracked_buffer_bytes() - before
                    XCTAssertEqual(tracked, 0,
                        "an empty exclusion list must not map the shard either")
                    XCTAssertEqual(arrays.keys.sorted(), [firstKey, secondKey].sorted())
                    let first = try XCTUnwrap(arrays[firstKey])
                    let second = try XCTUnwrap(arrays[secondKey])
                    XCTAssertEqual(MLX.sum(first + second, stream: .gpu).item(Float.self), 36)
                }
            }
        }
    }

    public func testNamedMmapBankForceInvalidateCanRefaultGPUComputation() throws {
        try withMLXMetallibForTests {
            let path = temporaryPath.appending(
                path: "named-mmap-bank-force-invalidate.bin",
                directoryHint: .notDirectory
            )
            let offset = 4096
            try writeUInt32Bank(at: path, offset: offset, values: [1, 2, 3, 4])
            let bank = try mapNamedUInt32Bank(
                at: path,
                offset: offset,
                shape: [1, 2, 2],
                name: "model.layers.43.mlp.switch_mlp.gate_proj.tq2_packed")

            withEnvironment("MLX_SAFETENSORS_MMAP_COLD_ADVICE", value: "force") {
                withExtendedLifetime(bank) {
                    XCTAssertEqual(
                        mlx_safetensors_mmap_advise_layer(0, 43),
                        16,
                        "force invalidate should advise the registered layer region")

                    let result = MLX.sum(
                        (bank.asType(.float32) * MLXArray(2.0)) + MLXArray(1.0), stream: .gpu)
                    XCTAssertEqual(result.item(Float.self), 24.0, accuracy: 0.001)
                }
            }
        }
    }

    public func testMmapFileRegionCanFeedGPUComputation() throws {
        try withMLXMetallibForTests {
            let path = temporaryPath.appending(
                path: "mmap-region.bin",
                directoryHint: .notDirectory
            )
            let offset = 4096
            var data = Data(repeating: 0, count: offset)
            for value in [Float(1), Float(2), Float(3), Float(4)] {
                var bits = value.bitPattern.littleEndian
                withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
            }
            try data.write(to: path)

            var array = mlx_array_new()
            var shape = [Int32(4)]
            let rc = path.withUnsafeFileSystemRepresentation { pathPtr -> Int32 in
                guard let pathPtr else { return 1 }
                return mlx_array_new_mmap_file_region(
                    &array,
                    pathPtr,
                    UInt64(offset),
                    4 * MemoryLayout<Float>.stride,
                    &shape,
                    Int32(shape.count),
                    DType.float32.cmlxDtype)
            }
            XCTAssertEqual(rc, 0)

            let loaded = MLXArray(array)
            let result = MLX.sum((loaded * MLXArray(4.0)) + MLXArray(1.0), stream: .gpu)
            XCTAssertEqual(result.item(Float.self), 44.0, accuracy: 0.001)
        }
    }
}

private func writeMixedAlignmentUInt32Safetensors(
    at url: URL,
    unalignedKey: String,
    unalignedValues: [UInt32],
    alignedKey: String,
    alignedValues: [UInt32],
    shape: [Int]
) throws {
    let unalignedByteCount = unalignedValues.count * MemoryLayout<UInt32>.stride
    let alignmentPadding = 2
    let alignedStart = unalignedByteCount + alignmentPadding
    let alignedEnd = alignedStart + alignedValues.count * MemoryLayout<UInt32>.stride
    var header = """
    {"__metadata__":{},"\(unalignedKey)":{"dtype":"U32","shape":\(shape),"data_offsets":[0,\(unalignedByteCount)]},"\(alignedKey)":{"dtype":"U32","shape":\(shape),"data_offsets":[\(alignedStart),\(alignedEnd)]}}
    """
    while (8 + header.utf8.count) % MemoryLayout<UInt32>.stride != 2 {
        header.append(" ")
    }

    var data = Data()
    var headerLength = UInt64(header.utf8.count).littleEndian
    withUnsafeBytes(of: &headerLength) { data.append(contentsOf: $0) }
    data.append(contentsOf: header.utf8)
    for value in unalignedValues {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }
    data.append(contentsOf: repeatElement(UInt8(0), count: alignmentPadding))
    for value in alignedValues {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }
    try data.write(to: url)
}

private func writeSparseFloat32Safetensors(
    at url: URL,
    firstKey: String,
    firstValues: [Float],
    secondKey: String,
    secondValues: [Float],
    gapBytes: Int,
    metadataJSON: String = "{}"
) throws {
    let firstByteCount = firstValues.count * MemoryLayout<Float>.stride
    let gapStart = ((firstByteCount + 15) / 16) * 16
    let secondStart = gapStart + max(0, gapBytes)
    let secondEnd = secondStart + secondValues.count * MemoryLayout<Float>.stride
    var header = """
    {"__metadata__":\(metadataJSON),"\(firstKey)":{"dtype":"F32","shape":[\(firstValues.count)],"data_offsets":[0,\(firstByteCount)]},"\(
        secondKey)":{"dtype":"F32","shape":[\(secondValues.count)],"data_offsets":[\(secondStart),\(secondEnd)]}}
    """
    while (8 + header.utf8.count) % MemoryLayout<Float>.stride != 0 {
        header.append(" ")
    }

    var data = Data()
    var headerLength = UInt64(header.utf8.count).littleEndian
    withUnsafeBytes(of: &headerLength) { data.append(contentsOf: $0) }
    data.append(contentsOf: header.utf8)
    for value in firstValues {
        var bits = value.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
    }
    data.append(Data(repeating: 0, count: secondStart - firstByteCount))
    for value in secondValues {
        var bits = value.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
    }
    try data.write(to: url)
}

private func withEnvironment<T>(_ key: String, value: String, body: () throws -> T) rethrows -> T {
    let oldValue = getenv(key).map { String(cString: $0) }
    setenv(key, value, 1)
    defer {
        if let oldValue {
            setenv(key, oldValue, 1)
        } else {
            unsetenv(key)
        }
    }
    return try body()
}

private func writeUInt32Bank(at url: URL, offset: Int, values: [UInt32]) throws {
    var data = Data(repeating: 0, count: offset)
    for value in values {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }
    try data.write(to: url)
}

private struct NamedBankMappingFailed: Error {
    let code: Int32
}

/// Maps the UInt32 bank of `shape` at `offset` in `url` as the named JANGH bank
/// `name`, which registers its bytes under the layer in the name.
private func mapNamedUInt32Bank(
    at url: URL,
    offset: Int,
    shape: [Int32],
    name: String
) throws -> MLXArray {
    var raw = mlx_array_new()
    var dimensions = shape
    let length = shape.reduce(1) { $0 * Int($1) } * MemoryLayout<UInt32>.stride
    let rc: Int32
    do {
        rc = try withError {
            url.withUnsafeFileSystemRepresentation { file -> Int32 in
                guard let file else { return 1 }
                return name.withCString { tensor in
                    mlx_array_new_mmap_file_region_named(
                        &raw, file, UInt64(offset), length, &dimensions,
                        Int32(dimensions.count), DType.uint32.cmlxDtype, tensor)
                }
            }
        }
    } catch {
        mlx_array_free(raw)
        throw error
    }
    guard rc == 0 else {
        mlx_array_free(raw)
        throw NamedBankMappingFailed(code: rc)
    }
    return MLXArray(raw)
}
