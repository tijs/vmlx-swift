# JANGH weighted-down decode primitive

Status: experimental source preparation. Swift syntax parsing passes; this is not
module typechecking or Metal execution. No loader or factory calls this primitive.
Build, Metal execution, model integration, speed, and model-family readiness remain unproved.

`JANGHWeightedDownKernel` performs the down projection for each selected route and
accumulates router-weighted token outputs without expanding the packed expert bank.
It implements odd-cubic codebooks for 2/3/4/6/8-bit LSB-packed projections with
per-output-row F16 scales.

The API requires F32 hidden rows `[tokens * routes, hidden]` **already in the down
projection input basis**. The caller must explicitly assert that basis; this
primitive performs no rotation. Route IDs are U32 and weights F32, both shaped
`[tokens, routes]`. Dot products and route accumulation remain F32. Only the final
`[tokens, output]` result is cast to the requested F16, BF16, or F32 dtype. No
normalization or sign restriction is imposed on router weights.

Geometry and dtype mismatches throw before dispatch. An out-of-range expert ID
produces NaNs for its entire token without reading that expert's bank or scales,
including when its score is zero. This is a deterministic diagnostic policy;
production integration must decide how to surface invalid routing safely.

Prepared tests use an independent bit-by-bit unpack and Float64 arithmetic oracle,
covering all five bit widths, input tails beyond a 512-column block, nine-row output
tails, repeated/out-of-order expert IDs, zero/non-normalized/signed scores, and all
three result dtypes, top-eight routing, and coefficient/basis kernel identity. A cancellation-sensitive two-route case distinguishes F32
hidden/accumulation from premature BF16/F16 rounding and from a second rotation.
Invalid tensor layouts and invalid expert IDs have separate tests. Tests use the
shared Metal lock.

Next gates: typecheck/build; execute all focused tests on actual Metal; inspect
numerical tolerances against the independent oracle; test fused gate/up-to-down
composition with mixed projection widths and down-input basis transitions. A
real model path then needs matched decode/prefill, cache and multi-turn proof.
No speedup or allocation measurement follows from source inspection alone.

The primitive makes inputs contiguous for dispatch. Passing a noncontiguous packed
bank may therefore materialize a bank copy, though it never dequantizes or expands
that bank. Model integration must establish contiguous mapped bank layout and
measure physical memory; this primitive is not a low-RAM proof.
