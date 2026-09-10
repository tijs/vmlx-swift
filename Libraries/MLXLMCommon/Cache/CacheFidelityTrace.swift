import Foundation
import MLX

/// `VMLX_FIDELITY_TRACE=1` only. Fingerprints the exact tensors that cross the
/// disk boundary, so a store and a later restore can be compared directly.
///
/// Why this exists: restoring the cross-conversation anchor changes greedy
/// output on a 20,406-token prompt, while a self-restore 27 tokens later is
/// byte-identical, and reading the source has not explained the difference.
/// The capture is provably innocent (cold with and without boundary capture
/// agree byte for byte) and the Mamba record is written and read without
/// conversion, yet the answer changes. Fingerprinting both ends settles
/// whether the round-trip is lossy or whether the divergence appears after it.
///
/// Fingerprints the SERIALIZED dictionary rather than the cache objects, so it
/// needs no per-layer-type knowledge and measures precisely what is persisted.
public enum CacheFidelityTrace {
    public static var isEnabled: Bool {
        ProcessInfo.processInfo.environment["VMLX_FIDELITY_TRACE"] == "1"
    }

    /// One line per tensor: name, shape, dtype, and two reductions that are
    /// cheap and catch any bit difference that matters — the sum and the
    /// maximum magnitude, both printed at full double precision.
    public static func dump(_ arrays: [String: MLXArray], label: String) {
        guard isEnabled else { return }
        var lines = ["[vmlx][fidelity] === \(label) (\(arrays.count) tensors) ==="]
        for key in arrays.keys.sorted() {
            let a = arrays[key]!
            let shape = a.shape.map(String.init).joined(separator: "x")
            // Reduce in float32: integer metadata tensors and bf16 state both
            // survive it, and the sum of a large bf16 array in its own dtype
            // would itself lose precision and hide a real difference.
            let f = a.asType(.float32)
            let sum = f.sum().item(Float.self)
            let amax = MLX.abs(f).max().item(Float.self)
            lines.append(String(format: "[vmlx][fidelity] %@ %@ %@ sum=%.9g absmax=%.9g",
                                key, shape, String(describing: a.dtype), sum, amax))
        }
        FileHandle.standardError.write(Data((lines.joined(separator: "\n") + "\n").utf8))
    }

    /// Fingerprint a live cache by serializing it the same way the disk store
    /// would. Used on the restore side, where only the cache objects exist.
    public static func dumpCache(_ cache: [any KVCache], label: String) {
        guard isEnabled else { return }
        dump(TQDiskSerializer.serialize(cache: cache), label: label)
    }
}
