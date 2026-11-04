// Copyright © 2026 Apple Inc.
import Foundation
import MLX
import Testing
@testable import MLXLMCommon

struct ExperimentalSafetensorsRegionReaderTests {
    private let resident = "language_model.model.layers.0.input_layernorm.weight"
    private let routed = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
    private let perExpert = "language_model.model.layers.0.mlp.experts.0.gate_proj.weight"
    private func fixture() throws -> (URL, Qwen35AffineResearchManifest.Inspection) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let shard = root.appendingPathComponent("model.safetensors")
        var header: [String: Any] = ["__metadata__": ["format": "mlx"]]
        var payload = Data()
        func add(_ key: String, _ dtype: String, _ shape: [Int], _ bytes: Data) {
            let start = payload.count; payload.append(bytes)
            header[key] = ["dtype": dtype, "shape": shape, "data_offsets": [start, payload.count]] as [String: Any]
        }
        for name in ["gate_proj", "up_proj", "down_proj"] {
            add("language_model.model.layers.0.mlp.switch_mlp.\(name).weight", "U32", [1, 64, 8], Data(repeating: 0, count: 2048))
        }
        for name in ["gate_proj", "up_proj", "down_proj"] { for suffix in ["scales", "biases"] {
            add("language_model.model.layers.0.mlp.switch_mlp.\(name).\(suffix)", "BF16", [1, 64, 1], Data(repeating: 0, count: 128))
        }}
        var numbers = Data()
        for value in ([1, 2, 3, 4] as [Float]) {
            var bits = value.bitPattern.littleEndian
            numbers.append(Data(bytes: &bits, count: 4))
        }
        add(resident, "F32", [4], numbers)
        add(perExpert, "F32", [1], Data(repeating: 0, count: 4))
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        json.append(Data(repeating: 32, count: (4096 - (8 + json.count) % 4096) % 4096))
        var size = UInt64(json.count).littleEndian
        var bytes = Data(bytes: &size, count: 8); bytes.append(json); bytes.append(payload)
        try bytes.write(to: shard)
        let map = Dictionary(uniqueKeysWithValues: header.keys.filter { $0 != "__metadata__" }.map { ($0, "model.safetensors") })
        try JSONSerialization.data(withJSONObject: ["weight_map": map]).write(to: root.appendingPathComponent("model.safetensors.index.json"))
        return (root, try Qwen35AffineResearchManifest.inspect(modelDirectory: root, selectedShards: [shard],
            layerCount: 1, numExperts: 1, inputDims: 64, hiddenDims: 64))
    }
    @Test("selected resident is mapped while expert keys and low budget are rejected")
    func selected() throws {
        let (root, inspection) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let gate = ExperimentalRequestArenaGate()
        let streams = [MLX.Stream.cpu, StreamOrDevice.default.stream]
        var copied: [Float] = []
        try ExperimentalOwnedStreamRequest.run(gate: gate, streams: streams.map { ExperimentalOwnedStream(stream: $0) }, operation: { arena in
            for key in [routed, perExpert, "unknown"] {
                #expect(throws: ExperimentalSafetensorsRegionReader.ReadError.self) {
                    try ExperimentalSafetensorsRegionReader.read(key: key, inspection: inspection, arena: arena, maxMappedBytes: 1 << 20)
                }
            }
            #expect(throws: ExperimentalSafetensorsRegionReader.ReadError.self) {
                try ExperimentalSafetensorsRegionReader.read(key: resident, inspection: inspection, arena: arena, maxMappedBytes: 1)
            }
            let value = try ExperimentalSafetensorsRegionReader.read(key: resident, inspection: inspection, arena: arena, maxMappedBytes: 1 << 20)
            #expect(value.shape == [4] && value.dtype == .float32)
            copied = value.asArray(Float.self)
        }, invalidate: {})
        #expect(copied == [1, 2, 3, 4])
    }
    @Test("missing and truncated shards fail without accepting a mapping")
    func missingAndTruncated() throws {
        for truncate in [false, true] {
            let (root, inspection) = try fixture()
            defer { try? FileManager.default.removeItem(at: root) }
            let file = root.appendingPathComponent("model.safetensors")
            if truncate {
                let handle = try FileHandle(forWritingTo: file); try handle.truncate(atOffset: 0); try handle.close()
            } else { try FileManager.default.removeItem(at: file) }
            let gate = ExperimentalRequestArenaGate()
            #expect(throws: (any Error).self) {
                let _: Void = try gate.withRequest(operation: { arena in
                    _ = try ExperimentalSafetensorsRegionReader.read(key: resident, inspection: inspection, arena: arena, maxMappedBytes: 1 << 20)
                    return .finished(())
                }, drain: { _ in }, invalidate: { _ in })
            }
        }
    }
    @Test("misaligned descriptor fails before C mapping")
    func misaligned() throws {
        let (root, inspection) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let original = try #require(inspection.tensors[resident])
        let bad = Qwen35AffineResearchManifest.TensorDescriptor(shard: original.shard, dtype: original.dtype,
            shape: original.shape, byteOffset: original.byteOffset + 1, byteLength: original.byteLength)
        let forged = Qwen35AffineResearchManifest.Inspection(modelDirectory: inspection.modelDirectory,
            tensors: [resident: bad], metadata: inspection.metadata)
        let gate = ExperimentalRequestArenaGate()
        #expect(throws: ExperimentalSafetensorsRegionReader.ReadError.self) {
            let _: Void = try gate.withRequest(operation: { arena in
                _ = try ExperimentalSafetensorsRegionReader.read(key: resident, inspection: forged, arena: arena, maxMappedBytes: 1 << 20)
                return .finished(())
            }, drain: { _ in }, invalidate: { _ in })
        }
    }
    @Test("arena aggregate charges repeated mappings and rejects cap changes")
    func aggregateBudget() throws {
        let (root, inspection) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let descriptor = try #require(inspection.tensors[resident])
        let page = UInt64(getpagesize())
        let span = descriptor.byteOffset % page + descriptor.byteLength
        let charge = ((span + page - 1) / page) * page
        let gate = ExperimentalRequestArenaGate()
        let _: Void = try gate.withRequest(operation: { arena in
            _ = try ExperimentalSafetensorsRegionReader.read(key: resident, inspection: inspection, arena: arena, maxMappedBytes: charge)
            #expect(throws: ExperimentalSafetensorsRegionReader.ReadError.self) {
                try ExperimentalSafetensorsRegionReader.read(key: resident, inspection: inspection, arena: arena, maxMappedBytes: charge)
            }
            #expect(throws: ExperimentalSafetensorsRegionReader.ReadError.self) {
                try ExperimentalSafetensorsRegionReader.read(key: resident, inspection: inspection, arena: arena, maxMappedBytes: charge * 2)
            }
            return .finished(())
        }, drain: { _ in }, invalidate: { _ in })
    }

    @Test("failed mmap reservation is returned to the arena budget")
    func failedMappingReleasesReservation() throws {
        let (root, inspection) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let descriptor = try #require(inspection.tensors[resident])
        let page = UInt64(getpagesize())
        let span = descriptor.byteOffset % page + descriptor.byteLength
        let charge = ((span + page - 1) / page) * page
        let file = root.appendingPathComponent("model.safetensors")
        let content = try Data(contentsOf: file)
        let gate = ExperimentalRequestArenaGate()
        let _: Void = try gate.withRequest(operation: { arena in
            try FileManager.default.removeItem(at: file)
            #expect(throws: (any Error).self) {
                try ExperimentalSafetensorsRegionReader.read(key: resident, inspection: inspection, arena: arena, maxMappedBytes: charge)
            }
            // Test-only recreation occurs before any mapping succeeds. Real callers
            // must keep the inspected snapshot immutable for its whole lifetime.
            try content.write(to: file)
            let value = try ExperimentalSafetensorsRegionReader.read(key: resident, inspection: inspection, arena: arena, maxMappedBytes: charge)
            #expect(value.shape == [4])
            return .finished(())
        }, drain: { _ in }, invalidate: { _ in })
    }

    @Test("concurrent reservations cannot exceed aggregate limit")
    func concurrentReservations() throws {
        final class State: @unchecked Sendable {
            let arena: ExperimentalRequestArena
            let lock = NSLock()
            var accepted = 0
            init(_ arena: ExperimentalRequestArena) { self.arena = arena }
            func reserve() {
                if (try? arena.reserveMappedRegion(bytes: 1, limit: 8)) == true {
                    lock.lock(); accepted += 1; lock.unlock()
                }
            }
        }
        let gate = ExperimentalRequestArenaGate()
        let _: Void = try gate.withRequest(operation: { arena in
            let state = State(arena)
            DispatchQueue.concurrentPerform(iterations: 16) { _ in state.reserve() }
            #expect(state.accepted == 8)
            let extra = try arena.reserveMappedRegion(bytes: 1, limit: 8)
            #expect(extra == false)
            return .finished(())
        }, drain: { _ in }, invalidate: { _ in })
    }

}
