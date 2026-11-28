/// Marks live KV that cannot be copied or restored as an ordinary cache snapshot.
package protocol ActiveKVSpillCache: KVCache {
    var usesActiveKVSpill: Bool { get }
}

#if canImport(Darwin)
package protocol ActiveKVSpillHistoryProviding: ActiveKVSpillCache {
    var activeKVSpillHistory: ActiveKVSpillHistory? { get }
}
#endif

public struct ActiveKVSpillUsage: Sendable {
    public let bytesRead: UInt64
    public let bytesWritten: UInt64
    public let prefillBytesRead: UInt64

    /// Call on the generation executor after its work has drained. These are
    /// logical file bytes and exclude filesystem or device read amplification.
    public static func measure(cache: [KVCache], prefillBytesRead: UInt64 = 0) -> Self? {
#if canImport(Darwin)
        let histories = cache.compactMap {
            ($0 as? any ActiveKVSpillHistoryProviding)?.activeKVSpillHistory
        }
        guard !histories.isEmpty else { return nil }
        return Self(
            bytesRead: histories.reduce(0) { $0 + $1.bytesRead },
            bytesWritten: histories.reduce(0) { $0 + $1.bytesWritten },
            prefillBytesRead: prefillBytesRead)
#else
        return nil
#endif
    }
}
