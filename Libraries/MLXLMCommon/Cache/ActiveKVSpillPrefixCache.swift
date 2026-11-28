import Foundation
import MLX

/// One process-local checkpoint at a complete prefill chunk. The next request
/// may reuse it for the same token prefix, model, and prefill geometry.
/// Access on the generation executor.
public final class ActiveKVSpillPrefixCache {
#if canImport(Darwin)
    private struct Entry {
        let model: ObjectIdentifier
        let tokens: [Int]
        let step: Int
        let salt: String?
        let cache: [KVCache]
    }

    private let maxResidentBytes: Int
    private let maxDiskBytes: Int
    private var entry: Entry?

    func matchingTokenCount(model: any LanguageModel, tokens: [Int], step: Int, salt: String?) -> Int {
        guard let entry, entry.model == ObjectIdentifier(model), entry.step == step,
              entry.salt == salt, entry.tokens.count < tokens.count,
              tokens.starts(with: entry.tokens) else { return 0 }
        return entry.tokens.count
    }

    func restore(into fresh: [KVCache]) -> [KVCache]? {
        guard let entry, fresh.count == entry.cache.count else { return nil }
        var restored: [KVCache] = []
        for (source, target) in zip(entry.cache, fresh) {
            if let source = source as? any ActiveKVSpillPrefixLayer,
               let history = source.activeKVSpillHistory {
                guard let target = target as? any ActiveKVSpillPrefixLayer,
                      let targetHistory = target.activeKVSpillHistory,
                      let fork = history.fork(into: targetHistory) else { return nil }
                restored.append(source.prefixCopy(spillHistory: fork))
            } else {
                restored.append(source.copy())
            }
        }
        return restored
    }

    func capture(model: any LanguageModel, tokens: [Int], step: Int, salt: String?, cache: [KVCache]) throws {
        guard step > 0, !tokens.isEmpty, tokens.count.isMultiple(of: step),
              cache.allSatisfy({ $0.offset == tokens.count }) else { return }
        let active = cache.compactMap {
            ($0 as? any ActiveKVSpillPrefixLayer)?.activeKVSpillHistory
        }
        let usedRAM = cache.reduce(0) { bytes, layer in
            bytes + layer.innerState().reduce(0) { $0 + $1.nbytes }
        }
        guard !active.isEmpty,
              active.reduce(0, { $0 + $1.storedBytes }) <= maxDiskBytes,
              usedRAM <= maxResidentBytes else { return }
        var saved: [KVCache] = []
        for layer in cache {
            if let spill = layer as? any ActiveKVSpillPrefixLayer, let history = spill.activeKVSpillHistory {
                guard let fork = history.fork() else { return }
                saved.append(spill.prefixCopy(spillHistory: fork))
            } else {
                saved.append(layer.copy())
            }
        }
        try withError { eval(saved) }
        entry = Entry(model: ObjectIdentifier(model), tokens: tokens, step: step, salt: salt, cache: saved)
    }
#endif

    public init(maxResidentBytes: Int, maxDiskBytes: Int = 4 * 1024 * 1024 * 1024) {
#if canImport(Darwin)
        self.maxResidentBytes = maxResidentBytes
        self.maxDiskBytes = maxDiskBytes
#endif
    }

    public func discard() {
#if canImport(Darwin)
        entry = nil
#endif
    }

}

#if canImport(Darwin)

package protocol ActiveKVSpillPrefixLayer: ActiveKVSpillHistoryProviding {
    func prefixCopy(spillHistory: ActiveKVSpillHistory) -> any KVCache
}
#endif
