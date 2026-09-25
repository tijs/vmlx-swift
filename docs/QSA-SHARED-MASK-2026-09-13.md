# Long-context shared-mask traversal — partial

Core candidate: `0c3e7c18d0ecb66c1be9fec96b0f41a6ed4f7fff`, based on
`qwen4-exp` at `73312d3e9ad0bd2e1bdf9a08d91e25571e255964`.
The canonical and Swift-generated `sdpa_vector.h` copies have SHA256
`a854ee6d08bfc37ee75c64927a277d383f3fc8e32cd7d551025ce5b492ef7a5a`.
No unrelated core-main update is included.

## Contract

In two-pass vector attention, pack a shared contiguous boolean mask into at most
32 words per threadgroup and reuse it across GQA heads. Set-bit iteration keeps
each selected key in its original split and chronological order. Score, softmax,
partial activation rounding, and second-pass reduction are unchanged. There is
no gather, extra dispatch, K/V copy, cache-format change, model allowlist, sampler
override, or MTP policy change.

Only non-causal single-query shared-head masks with more than 32,768 keys and
at most 32 words per split use the new traversal. Short contexts, multi-query
prefill, non-shared or strided masks, floating masks, and longer spans retain
the existing traversal. Normal two-pass dispatch and GPU-specific split counts
remain owned by the core host implementation.

## Evidence and remaining gates

The pre-integration private AOT comparison used the identical executable and
the dev app's exact Metal compiler/math flags, changing only the original SDPA
header. Original/candidate/candidate/original comparison matched all 126 output
arrays bitwise across BF16/F16/F32, GQA layouts, capacity-backed K/V views, sparse,
dense and fully masked rows, and fallback mask layouts.
Receipt: `SWIFTTEST_QSAAOTFallbackParity0913__044542.log`, exit 0, cleanup 0/0/0.

Isolated synthetic 12-layer attention cost: 35K keys about 1.29 ms to 0.63 ms;
131K keys about 3.63 ms to 0.89 ms. These are NOT model token/s, prefill or app
results. Short-context timings are noisy and must be checked end to end.

`SDPASharedMaskTests` adds 180 partition-order comparisons, 15 batch/sink/layout
comparisons, and canonical/generated header parity. Its original-path control
materializes identical per-head mask rows, forcing the old traversal with the
same compiler arithmetic. `SWIFTTEST_QSASharedMaskSourceTestsD0913__051126.log`
executes these three checked-in XCTest methods against the rebuilt dev app's
actual Metal library: 195 comparisons, zero failures, exit 0, peak 1.14 GiB,
unchanged swap, cleanup 0/0/0. This is a focused XCTest run, not the whole suite.
Earlier attempts retained: missing macOS test entrypoint, missing XCTest runtime
rpath, then three fixture-stride assertions inspected before lazy evaluation.
Evaluating that fixture before inspecting its stride exposes the intended
stride 2; the final test asserts it and retains bit-exact comparison.

Development app `ad06a29e6e555e3d24b8aa1cd36c23c3a33a6c38` built with engine
`8e07416e952bef49900d9cecdb266522ba2b5de2` and this core. Receipt
`SWIFTTEST_QSASharedMaskDevApp0913__045646.log`: exit 0, peak 21.46 GiB,
unchanged swap, cleanup 0/0/0. Binary SHA256
`bcead6c495275d2e6f5facf9155fbb163e9d694b0bc9420a5043e74ba262b494`;
its Metal library SHA256
`5313e50a2a7a122949b77859994177ce07f48f65ed1c13a532c88cf41f1f7ccc`
matches the original private AOT candidate exactly. Model generation is pending.

The existing Linux CMake integration failed at `Stream.runWith` linkage because
it omitted the local `stream_run_with.cpp` already compiled by SwiftPM. The
CMake target now includes that same source; Linux CI execution is still pending.

Before merge: run source-bound regression tests, rebuild the pinned local dev
app, record matched 1L/2L/4S short/long AR streams and visible multi-turn output,
check native cache restoration and resource bounds, and attach exact-head CI.
Earlier failed app-quality and resource-limit rows remain retained; generated
kernel equivalence does not turn them into successful app rows.
