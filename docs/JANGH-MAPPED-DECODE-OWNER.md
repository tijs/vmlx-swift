# Experimental JANGH mapped decode owner

Status: source prepared; no compile, mapped-file execution or Metal test evidence
for this owner/block yet. Existing primitive tests do not establish this new path.
No model factory is enabled and no model performance or low-footprint claim is made.

`JANGHMappedBanks.SourceLease` validates the header/index plan and retains regular
file descriptors with exact recorded identities. `JANGHMappedBanks` checks those
identities again before mapping and after all bank construction. It passes
`/dev/fd/N` to the existing core `mlx_array_new_mmap_file_region` operation while
the descriptors remain alive. That operation opens the supplied descriptor path
directly, maps a page-aligned region, and creates a tensor view backed by the core
mmap/Metal-buffer owner. It does not canonicalize or reopen an original filename.
Closing the Swift descriptors after construction does not destroy mappings retained
by MLX arrays. Atomic pathname replacement cannot redirect this route to the new
file; strict ctime checks can reject the now-stale lease. In-place modification of
mapped model files remains unsupported throughout their lifetime.

Only validated U32 packed banks and F16 row scales are mapped. No tensor payload is
read into Swift, evaluated, repacked, stacked, or expanded to dense weights. The
normal Swift raw-pointer mmap prototype is not used: it documents GPU/unaligned
pointer limitations. The region operation has no environment-controlled fallback
to the ordinary lazy reader. Ready/row-contiguous admission is required afterward.

The current core helper represents its byte-base span as a single `ShapeElem`.
This first owner explicitly rejects a bank mapping span above `Int32.max`; full
large-bank support requires an audited core fix, not an unchecked narrowing cast.
The actual local 2/3-bit GLM and Naive bank geometries fit this subset. Legitimate
HF snapshot file symlinks remain unsupported by the first header/lease adapter.
Neither restriction should be described as a corrupt model format.

`JANGHRoutedDecodeBlock` composes the tested input preparation, mixed-bit gate/up,
raw activation clamp and SwiGLU, optional down-input H32, and weighted down
primitives over these owned banks. The caller supplies the activation limit and
output dtype explicitly. Hidden state and route accumulation preserve the existing
F32 contract. This block is not a generic `SwitchGLU` replacement or a prefill path.

Five prepared real-file tests cover tiny mapped decode against an independent
scalar reference after dropping the owner and unlinking the shard, stale shard and
index identities, wrong architecture geometry, invalid module/activation requests,
pathname replacement after a retained source lease, and release of mapped regions
when the last bank owner is dropped. Failure-before-mapping
rows compare mapped-byte counters; this is not a large-model footprint metric.

GLM integration still needs an architecture-owned routed-module construction hook
that avoids creating dense `SwitchLinear` placeholders or traversing custom banks
through affine quantization. Its current property is concretely `SwitchGLU`, and
its fast path calls `qwen4ExpReduced`; a typed alternative must preserve routing,
shared-expert addition and native clamp semantics. The generic unweighted
`SwitchGLULayer` interface alone does not express the weighted F32 decode contract.
Prefill needs its own audited precision/dispatch path before any factory is enabled.
Naive still needs its independent attention/cache architecture. Config/sidecar
alias reconciliation, compiled decode compatibility, real low-memory load,
multi-turn/cache/media behavior, and measured prefill/decode remain separate gates.
