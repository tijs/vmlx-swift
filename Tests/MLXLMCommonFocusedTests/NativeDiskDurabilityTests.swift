import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Native disk boundary durability", .serialized)
struct NativeDiskDurabilityTests {
    private let tokens = Array(71 ... 78)

    private func configuration(_ root: URL, paged: Bool = false) -> CacheCoordinatorConfig {
        CacheCoordinatorConfig(
            usePagedCache: paged, enableDiskCache: true, pagedBlockSize: 4,
            maxCacheBlocks: 40, diskCacheMaxGB: 0.1,
            diskCacheDir: root, modelKey: "native-disk-durability")
    }

    private func nativeCoordinator(_ root: URL, paged: Bool = false) -> CacheCoordinator {
        let result = CacheCoordinator(config: configuration(root, paged: paged))
        result.setHybrid(
            true, requiresRecurrentSSMCompanion: true,
            requiresSeparateRecurrentPayload: false)
        return result
    }

    private func sourceCache(nested: Bool = false) -> [any KVCache] {
        let kv = KVCacheSimple()
        _ = kv.update(
            keys: MLXArray.ones([1, 1, tokens.count, 4]),
            values: MLXArray.ones([1, 1, tokens.count, 4]) * 2)
        let state = MambaCache(slots: 6, persistentSlotCount: 4)
        state[0] = MLXArray([Int32(11)])
        state[2] = MLXArray([Int32(13)])
        state[3] = MLXArray([Int32(14)])
        state[4] = MLXArray([Int32(999)])  // scratch must not be persisted
        state.offset = tokens.count
        if nested {
            let list = CacheList(state)
            list.offset = tokens.count
            return [kv, list]
        }
        return [kv, state]
    }

    private func withRoot(_ body: (URL) throws -> Void) throws {
        try FocusedMLXTestSupport.withLock {
            let root = FileManager.default.temporaryDirectory
                .appendingPathComponent("native-durable-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            try body(root)
        }
    }

    @Test(
        "native occupancy, holes and exact media/token keys survive reopen",
        arguments: [false, true])
    func roundTrip(nested: Bool) throws {
        try withRoot { root in
            let writer = nativeCoordinator(root)
            let source = sourceCache(nested: nested)
            #expect(source.allSatisfy { $0.offset == tokens.count })
            writer.storeAfterGeneration(
                promptTokens: tokens, perLayerData: [],
                ssmStates: extractSSMStates(from: source),
                cache: source, mediaSalt: "image-a")
            #expect(writer.hasValidatedDiskEntry(tokens: tokens, mediaSalt: "image-a"))
            let reader = nativeCoordinator(root)
            #expect(reader.hasDurableDiskEntry(tokens: tokens, mediaSalt: "image-a"))
            #expect(!reader.hasValidatedDiskEntry(tokens: tokens, mediaSalt: "image-a"))
            #expect(!reader.hasDurableDiskEntry(tokens: tokens, mediaSalt: "image-b"))
            #expect(!reader.hasDurableDiskEntry(tokens: tokens))
            #expect(!reader.hasDurableDiskEntry(tokens: tokens + [79], mediaSalt: "image-a"))
            #expect(
                !reader.hasDurableDiskEntry(tokens: Array(tokens.dropLast()), mediaSalt: "image-a"))
            guard
                case .hit(let matched, _, let detail, _, let companion, let payload) = reader.fetch(
                    tokens: tokens + [79], mediaSalt: "image-a")
            else {
                Issue.record("typed native disk boundary must restore without a sidecar")
                return
            }
            #expect(matched == tokens.count && detail == .disk && companion == nil)
            let arrays = try #require(payload)
            let target = MambaCache(slots: 6, persistentSlotCount: 4)
            var restored: [any KVCache] =
                nested
                ? [KVCacheSimple(), CacheList(target)]
                : [KVCacheSimple(), target]
            #expect(restoreFromDiskArrays(arrays, into: &restored) == tokens.count)
            #expect(target.offset == tokens.count)
            #expect(target[0]?.item(Int32.self) == 11 && target[2]?.item(Int32.self) == 13)
            #expect(target[3]?.item(Int32.self) == 14)
            #expect(target[1] == nil && target[4] == nil && target[5] == nil)
            #expect(reader.hasValidatedDiskEntry(tokens: tokens, mediaSalt: "image-a"))
        }
    }

    @Test(
        "incomplete native or legacy KV-only entries cannot suppress boundary production",
        arguments: 0 ..< 10)
    func rejectsIncomplete(damage: Int) throws {
        try withRoot { root in
            let writer = nativeCoordinator(root)
            var arrays = TQDiskSerializer.serialize(cache: sourceCache())
            switch damage {
            case 0: arrays.removeValue(forKey: "__mamba_1_slots__")
            case 1: arrays.removeValue(forKey: "__mamba_1_occupied__")
            case 2: arrays.removeValue(forKey: "mamba_1_state3")
            case 3: arrays["__mamba_1_occupied__"] = MLXArray([Int32(0), 2, 2])
            case 4: arrays["__mamba_1_slots__"] = MLXArray([Int32(2)])
            case 5: arrays.removeValue(forKey: "__mamba_1_offset__")
            case 6: arrays[TQDiskSerializer.formatVersionKey] = MLXArray([Int32(1)])
            case 7:
                arrays = arrays.filter { !$0.key.contains("mamba_") }
            case 8:
                arrays = TQDiskSerializer.serialize(cache: [sourceCache()[0]])
            default: arrays["mamba_1_state4"] = MLXArray([Int32(999)])
            }
            writer.diskCache?.store(tokens: tokens, arrays: arrays)
            #expect(!writer.hasValidatedDiskEntry(tokens: tokens))
            #expect(!writer.hasDurableDiskEntry(tokens: tokens))
            let reader = nativeCoordinator(root)
            #expect(!reader.hasDurableDiskEntry(tokens: tokens))
        }
    }

    @Test("missing, truncated and changed native files revoke validation")
    func fileIntegrity() throws {
        try withRoot { root in
            let writer = nativeCoordinator(root)
            writer.diskCache?.store(
                tokens: tokens, arrays: TQDiskSerializer.serialize(cache: sourceCache()))
            let disk = try #require(writer.diskCache)
            let file = disk.cacheDir.appendingPathComponent(
                DiskCache.hashTokens(tokens, modelKey: disk.modelKey) + ".safetensors")
            #expect(writer.hasValidatedDiskEntry(tokens: tokens))
            try FileManager.default.setAttributes(
                [.modificationDate: Date(timeIntervalSinceNow: 5)],
                ofItemAtPath: file.path)
            #expect(!writer.hasValidatedDiskEntry(tokens: tokens))
            let handle = try FileHandle(forWritingTo: file)
            try handle.truncate(atOffset: 16)
            try handle.close()
            #expect(!writer.hasDurableDiskEntry(tokens: tokens))
            try FileManager.default.removeItem(at: file)
            #expect(!writer.hasDurableDiskEntry(tokens: tokens))
        }
    }

    @Test("native ownership does not bypass the store's offset/key mismatch guard")
    func refusesMismatchedBoundary() throws {
        try withRoot { root in
            let coordinator = nativeCoordinator(root)
            let source = sourceCache(nested: true)
            let container = try #require(source[1] as? CacheList)
            container.offset = 0
            coordinator.storeAfterGeneration(
                promptTokens: tokens, perLayerData: [],
                ssmStates: nil, cache: source)
            #expect(!coordinator.hasDurableDiskEntry(tokens: tokens))
            #expect(!coordinator.hasValidatedDiskEntry(tokens: tokens))
            #expect(coordinator.diskCache?.snapshotStats().stores == 0)
        }
    }

    @Test(
        "separate and unknown topologies retain complete-sidecar requirement",
        arguments: [false, true])
    func requiresSidecar(unknown: Bool) throws {
        try withRoot { root in
            let writer = CacheCoordinator(config: configuration(root))
            if unknown {
                writer.setHybrid(true)
            } else {
                let arrays = ArraysCache(size: 2)
                arrays[0] = MLXArray.ones([1, 4])
                arrays[1] = MLXArray.ones([1, 4]) * 2
                arrays.offset = tokens.count
                let topology = ModelCacheTopologySnapshot(cache: [sourceCache()[0], arrays])
                #expect(topology.arraysLayerCount == 1)
                #expect(topology.requiresSeparateRecurrentPayloadState)
                writer.setHybrid(
                    true,
                    requiresRecurrentSSMCompanion: topology.requiresRecurrentSSMCompanionState,
                    requiresSeparateRecurrentPayload: topology.requiresSeparateRecurrentPayloadState
                )
            }
            writer.diskCache?.store(
                tokens: tokens, arrays: TQDiskSerializer.serialize(cache: sourceCache()))
            #expect(!writer.hasDurableDiskEntry(tokens: tokens))
            #expect(!writer.hasValidatedDiskEntry(tokens: tokens))
            let states = [MLXArray.ones([1, 4])]
            writer.ssmStateCache.store(
                ssmStates: states, tokens: tokens, boundary: tokens.count,
                isComplete: false)
            #expect(!writer.hasDurableDiskEntry(tokens: tokens))
            writer.ssmStateCache.store(ssmStates: states, tokens: tokens, boundary: tokens.count)
            #expect(writer.hasDurableDiskEntry(tokens: tokens))
            #expect(writer.hasValidatedDiskEntry(tokens: tokens))
        }
    }

    @Test("native disk durability does not admit a companion-less paged hit")
    func pagedStillRequiresCompanion() throws {
        try withRoot { root in
            let coordinator = nativeCoordinator(root, paged: true)
            let source = sourceCache()
            coordinator.storeAfterGeneration(
                promptTokens: tokens,
                perLayerData: [(keys: source[0].state[0], values: source[0].state[1]), nil],
                ssmStates: nil, cache: source)
            #expect(coordinator.hasDurableDiskEntry(tokens: tokens))
            if case .hit(_, _, let detail, _, _, _) = coordinator.fetch(tokens: tokens + [79]) {
                #expect(detail == .disk, "paged KV alone must not count as a hybrid hit")
            } else {
                Issue.record("native disk fallback should remain usable")
            }
        }
    }
}
