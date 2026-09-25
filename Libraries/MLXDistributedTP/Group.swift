import Foundation
import MLX
import CmlxDistributedShim

/// Swift wrapper around an MLX `mlx_distributed_group` handle. Created
/// once per process via `Group.init(strict:backend:)`; subsequent
/// `split` returns child groups that share the same lifecycle.
///
/// Phase 5 scope:
/// - rank, size, split queryable.
/// - Lifecycle: shared handle storage frees the pinned C ABI group on last release.
public struct Group: Sendable {
    /// Opaque mlx_distributed_group handle (just a void* ctx wrapper).
    public let handle: MLXDistributedGroupHandle

    /// Initialise the global distributed group for this process.
    /// - Parameters:
    ///   - strict: when true, throws if the requested backend can't init
    ///     (env vars missing, etc). When false, returns a trivial size-1
    ///     group on failure — same semantics as Python's
    ///     `mx.distributed.init()`.
    ///   - backend: optional backend hint ("jaccl", "ring", "mpi",
    ///     "nccl"). nil lets MLX pick.
    public init(strict: Bool = false, backend: String? = nil) {
        if let backend {
            self.handle = backend.withCString { bk in
                MLXDistributedGroupHandle(_MLXDistributedGroupRaw(ctx: vmlx_group_init(strict, bk)))
            }
        } else {
            self.handle = MLXDistributedGroupHandle(_MLXDistributedGroupRaw(ctx: vmlx_group_init(strict, nil)))
        }
    }

    /// Number of ranks in this group.
    public var size: Int {
        Int(vmlx_group_size(handle.raw.ctx))
    }

    /// Local rank within this group.
    public var rank: Int {
        Int(vmlx_group_rank(handle.raw.ctx))
    }

    /// Returns true if this group has more than one rank — i.e. real
    /// multi-host work is happening. False means a no-op "group of 1".
    public var isMultiRank: Bool { size > 1 }

    /// Split into a sub-group; ranks with the same `color` end up in
    /// the same returned group, with `key` controlling rank order.
    /// On a size-1 group this is a no-op (mlx-c rejects splits of the
    /// trivial group; we return self rather than expose the empty
    /// handle that would result).
    public func split(color: Int, key: Int) -> Group {
        guard isMultiRank else { return self }
        let raw = _MLXDistributedGroupRaw(ctx: vmlx_group_split(handle.raw.ctx, Int32(color), Int32(key)))
        return Group(handle: MLXDistributedGroupHandle(raw))
    }

    private init(handle: MLXDistributedGroupHandle) {
        self.handle = handle
    }
}

/// Sendable wrapper around the mlx_distributed_group struct (just a
/// `void* ctx` pointer). The underlying C++ Group is reference-counted
/// internally; passing this handle by value is safe.
public struct MLXDistributedGroupHandle: @unchecked Sendable {
    private final class Storage {
        let raw: _MLXDistributedGroupRaw
        init(_ raw: _MLXDistributedGroupRaw) { self.raw = raw }
        deinit { vmlx_group_free(raw.ctx) }
    }
    private let storage: Storage
    var raw: _MLXDistributedGroupRaw { storage.raw }

    init(_ raw: _MLXDistributedGroupRaw) { storage = Storage(raw) }
}

/// Opaque context shared with the C collective bridge.
public struct _MLXDistributedGroupRaw {
    public var ctx: UnsafeMutableRawPointer?
    public init(ctx: UnsafeMutableRawPointer? = nil) { self.ctx = ctx }
}
