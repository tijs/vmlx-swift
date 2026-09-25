# Native recurrent disk-boundary admission

Status: PARTIAL — regression reproduced and generated matrix passed; app proof pending.

## Mechanism

`TokenIterator.storeCacheAfterGeneration` asks
`CacheCoordinator.hasDurableDiskEntry` before obtaining a stable-prefix snapshot.
For non-trimmable recurrent state, a miss can replay the whole prefix after the
last answer token. The disk-only Mamba writer deliberately stores recurrent
state in the typed v2 payload, without a duplicate sidecar. The admission query
still required that omitted sidecar. A restorable native boundary was therefore
reconstructed before `DiskCache.store` recognized it and skipped the write.

The existing native round-trip fixture now checks current-process durability,
current-process validation, and durability after reopening the disk directory.
All three assertions failed on main while the disk fetch itself succeeded.
Local receipt: `SWIFTTEST_NativeDiskDurabilityRed0913__225443.log`, one test,
three failed assertions, exit 1, cleanup 0/0/0, unchanged 0.49 GiB swap.

## Candidate contract

- Use cache topology, not model names, quant presets, or a new runtime flag.
- Native disk Mamba state can satisfy disk admission without a sidecar.
- ArraysCache and unknown recurrent topologies retain the complete-sidecar
  requirement. Paged hybrid hits retain their own companion gate.
- Require native typed recurrent state before bypassing the sidecar. Validate
  declared occupied slots, holes, offsets and tensor presence, including nested
  CacheList children. Legacy or incomplete layouts must not suppress repair.
- Cold admission reads the safetensors header and at most 64 KiB of integer
  metadata, not the state tensors. Warm admission retains file-fingerprint and
  index checks. File formats, model weights, sampling and decode math do not change.

## Required verification

`SWIFTTEST_NativeDiskDurabilityMatrixD0913__231844.log` completed 63 tests in
eight suites, exit 0, peak tracked physical footprint 1.66 GiB, flat 0.49 GiB
swap, cleanup 0/0/0. The selected matrix includes `NativeDiskDurabilityTests`,
`CacheCoordinatorTopologyFocusedTests`, `MambaPersistentDiskTests`, and
`FlashPersistentDiskContinuationTests.earlySubmissionConnectedState`. The
Flash fixture compares exact next logits and companion state after a real
coordinator disk-store/reopen/fetch at four mixed-quant geometries, with early
submission both disabled and enabled. These are generated small models, not
installed-model or general media-quality proof.

Retained unsuccessful attempts: the initial green run omitted the new file
from the explicit SwiftPM source list; it is not its coverage receipt. Matrix
and MatrixC stopped at test compilation (argument order and a read-only
protocol property). MatrixB exposed an invalid nested fixture offset; the
fixture now matches the boundary, with a separate offset-mismatch refusal test.
No runtime guard was relaxed for those fixture corrections.

Pending: six app pins, fresh isolated optimized development build, and
matched 2L/4S connected UI turns with completion-tail timing, Stop/recovery,
raw streaming rates, effective cache topology and physical-footprint receipts.

This is a completion-latency correction, not evidence of higher decode tok/s.
No MTP, general VLM quality, RAM-soak, release or universal speed-floor claim.
