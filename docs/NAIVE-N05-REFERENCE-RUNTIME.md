# Naive-N0.5 reference runtime

This is an unregistered architecture implementation, not advertised model support.
It ports the Naive-N0.5-Flash attention/indexer and pre-norm layer graph from
vendor revision `0235b3b5ff27422b1f57cdc2acddfaf643e08356`. Routing follows the
Transformers reference at `856157a2f3e9594954310df18fdccc31ffddebe9`.

The graph supports split asymmetric Q/K/V, partial rotary positions, sliding
attention sinks, stable sparse-indexer selection, full-row E4M3 rounding,
FP32 sigmoid routing, and separate gate/up/down projections. Custom weighted
expert modules are constructed before the affine fallback; they never require
allocating placeholder affine expert banks. There is no shared expert or
activation clamp. The ordinary expert path rounds each weighted contribution
to activation dtype before its reduction; exact reduced-precision accumulation
order versus the vendor remains a separate numerical gate.

The cache owns attention K/V and sparse indexer keys together. Append, trim and
validated restoration preserve row alignment. Sliding caches retain exactly
`min(offset, window - 1)` rows while returning the full history needed by the
current call. They reject short-tail restoration and do not advertise rollback
after wrap. The LanguageModel adapter admits unpadded single-sequence text only and declares
maximum decode batch size 1 and whole-forward compilation unavailable. Throwing
prepare/replay validate cache type, layer topology, offset and geometry before
mutating any layer. Unsupported padded runtime input is rejected rather than
silently discarded; the explicit reference API retains padding support.
Cross-request companion batching and padded-position persistence remain required
before widening admission. The nonthrowing generation protocol treats bypassing
these admission invariants as a programming error, without substituted logits.

Five additional tests passed on Metal under both precision policies: actual
disk serialize/reopen/continued logits, missing late companion atomic refusal,
B1 chunk/continuation, padded/batched/wrong-cache admission refusal, and a
late-layer wrong-dtype failure preserving every cache object's identity, offset
and row values. The adapter retains immutable array references before a forward
and restores all owned caches on synchronous thrown errors; this creates no KV
copies. It does not claim rollback of asynchronous device errors or cancellation
inside an already executing forward.

Converted expert parameters use the `mlp.switch_mlp` namespace. Custom routed
construction accepts an exact set of safetensors keys excluded before generic
loading and requires that a custom routed factory accompany that set. Header
metadata validation and completeness of that set belong to loader preparation.

## Bounded validation

The original nine tests exercise FP8 rounding, rotary positions, stable ties past 2048 keys,
sink/all-masked attention, sliding history, atomic companion restore/trim,
router correction, custom construction, and a real tiny two-layer model's
full/chunk/decode equivalence with left padding. All nine passed on Metal with
`MLX_ENABLE_TF32=0` set before process startup, using the actual MLX dependencies
and an independently linked test executable. This is not a full-module build or
real-bundle test. The extended adapter suite passed all 14 methods with the same
strict precision policy; default precision passed 12 and retained the same two
strict numerical failures described below.

The strict float32 numerical suite requires that process precision policy.
On M5 with default TF32 enabled, the same binary, weights and tests passed seven
methods and failed two tight float32 bounds: the analytic sink result differed
by 0.00048828125 and full/chunk output by 0.0010845661. Those failures are retained;
no tolerance was relaxed and no production setting changed. Tests must be run
with the declared precision policy for the strict reference gate. Default-mode
and reduced-precision model qualification remain separate gates.

Before registration: prove the LanguageModel/cache adapter, companion
serialization and rejection of unsupported batching, compile the complete
modules, and run the actual bundle through coherent multi-turn, native tool,
prefix continuation, cancellation, footprint and performance checks. No model
weights were loaded for the bounded tests.
