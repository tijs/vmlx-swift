// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("QSA raw-key capacity and ownership", .serialized)
struct QSAKVCacheStorageTests {
    private func rows(
        _ count: Int, batch: Int = 1, width: Int = 8,
        start: Int = 0, dtype: DType = .bfloat16
    ) -> MLXArray {
        let values = (0 ..< (batch * count * width * 2)).map {
            Float(($0 + start * 17) % 211 - 105) / 16
        }
        // A non-contiguous input must not silently change cache layout/data.
        return MLXArray(values, [batch, count, width * 2]).asType(dtype)[
            0..., 0..., .stride(by: 2)]
    }

    private func bits(_ x: MLXArray) -> [UInt32] {
        x.asType(.float32).asArray(Float.self).map(\.bitPattern)
    }

    private func commit(_ cache: QSAKVCache, _ raw: MLXArray) {
        _ = cache.updateIndexerKeys(raw)
        let kv = raw.reshaped(raw.dim(0), 1, raw.dim(1), raw.dim(2))
        _ = cache.update(keys: kv, values: kv)
    }

    @Test("prefill, AR, pending replacement and rollback match frozen concat exactly")
    func sequenceParity() throws {
        try MLXMetalTestLock.withLock {
            var checks = 0
            for dtype in [DType.bfloat16, .float16, .float32] {
                for batch in [1, 2] {
                    for prefix in [1, 255, 2040, 8339, 34939] {
                        let cache = QSAKVCache(useIndexerCapacity: true)
                        var reference: MLXArray?
                        func append(_ length: Int, commitKV: Bool = true) throws {
                            let input = rows(length, batch: batch, start: checks, dtype: dtype)
                            // Frozen pre-change recurrence, independent of the candidate.
                            if let old = reference, old.dim(1) > 0 {
                                let valid =
                                    old.dim(1) > cache.offset
                                    ? old[0..., ..<cache.offset, 0...] : old
                                reference = concatenated([valid, input], axis: 1)
                            } else {
                                reference = input
                            }
                            let previous = cache.offset
                            let actual = cache.updateIndexerKeys(input)
                            let expected = try #require(reference)
                            #expect(cache.offset == previous)
                            #expect(actual.shape == expected.shape)
                            #expect(actual.dtype == expected.dtype)
                            #expect(bits(actual) == bits(expected))
                            if commitKV {
                                let kv = input.reshaped(batch, 1, length, 8)
                                _ = cache.update(keys: kv, values: kv)
                                #expect(cache.state[2].dim(1) == cache.offset)
                            }
                            checks += 1
                        }
                        try append(prefix)
                        try append(1)
                        try append(4, commitKV: false)
                        #expect(cache.state[2].dim(1) == cache.offset)
                        try append(2)  // Replaces the four uncommitted rows.
                        cache.derivedPooledBlocks = MLXArray.ones([batch, 1, 8])
                        cache.derivedPooledBlockCount = 1
                        let capacity = cache.indexerStorageCapacity
                        #expect(cache.trim(3) == min(prefix + 3, 3))
                        reference = reference?[0..., ..<cache.offset, 0...]
                        #expect(cache.derivedPooledBlocks == nil)
                        #expect(cache.derivedPooledBlockCount == 0)
                        #expect(cache.indexerStorageCapacity == capacity)
                        try append(64)
                        let all = cache.offset
                        #expect(cache.trim(all + 1) == all)
                        reference = reference?[0..., ..<0, 0...]
                        try append(1)
                    }
                }
            }
            print("[QSARawStorage] exact_sequence_checks=\(checks)")
        }
    }

    @Test("dtype transitions preserve concat promotion rather than assignment casts")
    func dtypeTransitions() throws {
        try MLXMetalTestLock.withLock {
            for initial in [DType.bfloat16, .float16, .float32] {
                for next in [DType.bfloat16, .float16, .float32] {
                    for committed in [false, true] {
                        let candidate = QSAKVCache(useIndexerCapacity: true)
                        let original = QSAKVCache(useIndexerCapacity: false)
                        let first = rows(5, dtype: initial)
                        for cache in [candidate, original] {
                            if committed {
                                commit(cache, first)
                            } else {
                                _ = cache.updateIndexerKeys(first)
                            }
                        }
                        let input = rows(3, start: 17, dtype: next)
                        let expected = original.updateIndexerKeys(input)
                        let actual = candidate.updateIndexerKeys(input)
                        #expect(actual.dtype == expected.dtype)
                        #expect(bits(actual) == bits(expected))
                    }
                }
            }
        }
    }

    @Test("retained views, copy and disk state own the original prefix across reuse")
    func copyAndDiskContinuation() throws {
        try MLXMetalTestLock.withLock {
            let cache = QSAKVCache(useIndexerCapacity: true)
            commit(cache, rows(2040))
            let retained = try #require(cache.indexerKeys)
            let expected = bits(retained)
            let copy = try #require(cache.copy() as? QSAKVCache)
            MLX.eval(copy.innerState())
            #expect(copy.indexerStorageCapacity == 2040)
            #expect(copy.state[2].dim(1) == 2040)
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("qsa-raw-capacity-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let file = directory.appendingPathComponent("cache.safetensors")
            try MLX.save(arrays: TQDiskSerializer.serialize(cache: [cache]), url: file)
            let serialized = try MLX.loadArrays(url: file)
            var restored: [any KVCache] = [QSAKVCache(useIndexerCapacity: true)]
            #expect(restoreFromDiskArrays(serialized, into: &restored) == 2040)
            let fromDisk = try #require(restored[0] as? QSAKVCache)
            #expect(fromDisk.indexerStorageCapacity == 2040)
            cache.trim(8)
            commit(cache, rows(16, start: 99))
            MLX.eval(cache.innerState())
            #expect(bits(retained) == expected)
            #expect(bits(try #require(copy.indexerKeys)) == expected)
            #expect(bits(try #require(fromDisk.indexerKeys)) == expected)
            // Continue the restored/copy prefix over the real 2048 budget.
            for restoredCache in [copy, fromDisk] {
                commit(restoredCache, rows(16, start: 71))
                #expect(restoredCache.offset == 2056)
                #expect(restoredCache.indexerKeys?.dim(1) == 2056)
                #expect(restoredCache.state[2].dim(1) == 2056)
                #expect(restoredCache.indexerStorageCapacity == 2304)
            }
            #expect(bits(copy.state[2]) == bits(fromDisk.state[2]))
            #expect(bits(cache.state[2]) != bits(copy.state[2]))
        }
    }

    @Test("exact-capacity snapshots survive rollback without mutating restored source state")
    func exactCapacitySnapshotOwnership() throws {
        try MLXMetalTestLock.withLock {
            for capacity in [false, true] {
                for dtype in [DType.bfloat16, .float16, .float32] {
                    for prefix in [256, 512] {
                        for materialized in [false, true] {
                            let cache = QSAKVCache(useIndexerCapacity: capacity)
                            let input = rows(prefix, dtype: dtype)
                            let expected = bits(input)
                            commit(cache, input)
                            let retained = try #require(cache.indexerKeys)
                            let payload = cache.state
                            if materialized { MLX.eval(cache.innerState()) }

                            // A restored cache must not adopt the same mutable
                            // MLXArray wrapper as either the payload or source.
                            let restored = QSAKVCache(useIndexerCapacity: capacity)
                            restored.state = payload
                            #expect(restored.trim(8) == 8)
                            commit(restored, rows(4, start: 99, dtype: dtype))
                            MLX.eval(restored.innerState())
                            #expect(bits(payload[2]) == expected)
                            #expect(bits(try #require(cache.indexerKeys)) == expected)

                            // Reuse without growing: the old full-capacity
                            // public view must retain all original prefix rows.
                            #expect(cache.trim(8) == 8)
                            commit(cache, rows(4, start: 71, dtype: dtype))
                            MLX.eval(cache.innerState())
                            #expect(bits(retained) == expected)
                            #expect(bits(payload[2]) == expected)
                            #expect(cache.indexerKeys?.dim(1) == prefix - 4)
                            #expect(bits(cache.state[2]) != bits(restored.state[2]))
                        }
                    }
                }
            }
        }
    }

    @Test("empty raw updates never expose reserved rows or preserve an old pending suffix")
    func emptyUpdates() throws {
        try MLXMetalTestLock.withLock {
            for capacity in [false, true] {
                let cache = QSAKVCache(useIndexerCapacity: capacity)
                #expect(cache.updateIndexerKeys(rows(0)).shape == [1, 0, 8])
                commit(cache, rows(3))
                _ = cache.updateIndexerKeys(rows(4, start: 2))
                #expect(cache.indexerKeys?.dim(1) == 7)
                #expect(cache.updateIndexerKeys(rows(0)).dim(1) == 3)
                #expect(cache.state[2].dim(1) == 3)
            }
        }
    }

    @Test("capacity grows at step boundaries and legacy/short lanes stay fail-closed")
    func growthAndMissingState() throws {
        try MLXMetalTestLock.withLock {
            let cache = QSAKVCache(useIndexerCapacity: true)
            cache.step = 16
            let input = rows(1)
            for _ in 0 ..< 65 { commit(cache, input) }
            MLX.eval(cache.innerState())
            #expect(cache.indexerStorageGrowths == 5)
            #expect(cache.indexerStorageCapacity == 80)
            #expect(cache.indexerKeys?.dim(1) == 65)
            #expect(cache.state[2].dim(1) == 65)
            cache.state = Array(cache.state.prefix(2))
            #expect(cache.indexerKeys == nil)
            #expect(cache.indexerStorageCapacity == 0)
            #expect(cache.updateIndexerKeys(input).dim(1) == 1)
            // Never synthesize the missing 65 raw rows to match the K/V offset.
            #expect(cache.state[2].dim(1) == 1)
            var destination: [any KVCache] = [QSAKVCache()]
            let damaged = TQDiskSerializer.serialize(cache: [cache])
            #expect(restoreFromDiskArrays(damaged, into: &destination) == 0)
        }
    }

    @Test("isolated 12-lane raw-cache cost at short, 8K and 35K contexts")
    func rawStorageCost() throws {
        guard RuntimeEnvironment.flag("VMLX_QSA_RAW_STORAGE_BENCH") else { return }
        try MLXMetalTestLock.withLock {
            for context in [1024, 8339, 34939] {
                let prefix = rows(context, width: 128)
                let input = rows(1, width: 128, start: 73)
                MLX.eval(prefix, input)
                for pair in 0 ..< 3 {
                    for capacity in (pair % 2 == 0 ? [false, true] : [true, false]) {
                        let caches = (0 ..< 12).map { _ in QSAKVCache(useIndexerCapacity: capacity)
                        }
                        for cache in caches {
                            _ = cache.updateIndexerKeys(prefix)
                            cache.offset = context
                        }
                        MLX.eval(caches.flatMap { $0.innerState() })
                        let start = ProcessInfo.processInfo.systemUptime
                        for _ in 0 ..< 256 {
                            for cache in caches {
                                _ = cache.updateIndexerKeys(input)
                                cache.offset += 1
                            }
                            MLX.eval(caches.flatMap { $0.innerState() })
                        }
                        let ms = (ProcessInfo.processInfo.systemUptime - start) * 1000 / 256
                        #expect(caches.allSatisfy { $0.indexerKeys?.dim(1) == context + 256 })
                        print(
                            "[QSARawStorageBench] context=\(context) pair=\(pair) capacity=\(capacity) lanes=12 tokens=256 ms=\(ms) growths=\(caches[0].indexerStorageGrowths) model_speed_claim=0"
                        )
                    }
                }
            }
        }
    }
}
