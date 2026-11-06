# JANGH metadata and numerical primitives

This change adds strict format validation and experimental Metal projection primitives. It does not register a JANGH model loader, alter generation settings, or claim a speedup.

The validator recognizes version 2, LSB bitstream packing, F16 row scales, odd-cubic codebooks, and 2/3/4/6/8-bit projections. It checks finite coefficients and levels, complete gate/up/down groups, compatible input rotation, and packed tensor geometry. These codes are not affine weights and must not enter an affine fallback.

The single-projection primitive performs guarded packed QMV with F32 accumulation, optional H32 input rotation, and explicit invalid-route handling. Its H32 casts back to the input dtype. The separate fused gate/up primitive supports independent bit widths and codebooks, raw gate/up clamping, SwiGLU, and an optional H32 epilogue preparing the down projection's input. Fused decode preparation preserves F32 after H32 for F16/BF16 source activations; activated hidden rows and the down-input rotation remain F32. These are distinct precision contracts, covered separately by tests.

## Executed numerical evidence

On integration build `df99ffece51054251e30cee17ccdfc5c6ee6a585`, `JANGHProjectionKernelTests` passed all five methods, and `JANGHFusedGateUpKernelTests` passed all five methods on Metal. Coverage includes independent packed decoding and scalar activation oracles, mixed bit widths, nonzero cubic terms, dtype boundaries, odd output dimensions, input tails, invalid routes, multiple H32 blocks, repeated top-eight routing, and finite extreme activations. The optional private packed-range fixture test was supplied and passed, rather than skipped. Model weight excerpts are not included in this repository.

The format contract also has standalone Foundation XCTest evidence. The Metal results came from a combined integration build, not an independently rebuilt head of this branch. They establish bounded primitive numerical behavior, not model load support, coherent generation, cache correctness, low-memory operation, chip-wide coverage, or faster prefill/decode.

## Remaining integration work

A future loader must validate the complete custom format before ordinary quantization decoding, reconcile quantization aliases, check index and tensor headers, and explicitly exclude custom projections from affine fallback. Legacy JANGTQ names must not select these kernels by spelling alone. Model-specific construction, complete expert composition, sorted prefill kernels, resource behavior, and live multi-turn tests are required before enabling a factory route. Naive requires its own architecture implementation; GLM support does not imply it.

## Packed-bank storage admission

The experimental primitives require every packed bank and row-scale bank to be already available and row contiguous. Admission queries Cmlx availability and row-layout flags only; it does not evaluate tensors, inspect their payload, or repair their layout. Unavailable and noncontiguous banks raise an explicit error. Storage owners must prepare banks before dispatch.

Accepted banks pass directly to the shaders. Automatic custom-kernel row repacking is disabled, while request-sized activations, indices, and scores retain explicit contiguous preparation. This prevents an unexpected bank view from silently producing another full-bank allocation. It does not prove total model physical footprint, eliminate loader alignment copies, or guarantee that mapped pages remain cold during execution.

Focused tests cover ready dense banks, contiguous offset views, same-shape noncontiguous packed/scales views, and rejection without evaluating lazy banks across all three primitives. These new storage-admission tests are source-prepared and have not executed yet. Earlier numerical results predate this guard and must be repeated after integration. Mapped-loader and large-model physical-footprint validation remain separate gates.

Two additional tests exercise the actual safetensors loader using a tiny, aligned synthetic U32/F16 shard. The mmap case requires an increase in tracked mapped bytes, then checks bank availability and row layout before any evaluation or payload read and runs a numerical projection. The ordinary reader case verifies that its lazy `Load` arrays are rejected without preparation. Neither test has executed yet. The guard is deliberately not relaxed for ordinary reader arrays: their storage owner must make an explicit preparation decision.
