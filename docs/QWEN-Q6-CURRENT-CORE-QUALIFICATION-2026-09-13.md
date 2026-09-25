# Mixed-input q6 decode requalification

Status: PARTIAL. Current-core numerical qualification passed on September 14;
exact-pin development-app dispatch, speed and quality proof remain pending.

## September 14 current-core numerical receipt

Core base `0c3e7c18d0ecb66c1be9fec96b0f41a6ed4f7fff`, tested diff SHA256
`5d1d872861ada22b326885a15865f650b4c4db68a5579967e09f11db9c262e7f`.
Test source: `Tests/MLXLMTests/Qwen4ExpQ6MixedAffineTests.swift`, SHA256
`949cf18ecef204405edb3bf09fef98c371ebc88526ad04b6db5550a915cdf417`.
Test executable SHA256:
`de6dfedd30ad18306c88947b361f72748bb25872b9adc005afd31ea160e49e14`.
The tests used the unchanged source-bound AOT library plus the new embedded
quantized JIT source; the log records actual mixed-q6 activation.

Local supervisor `SWIFTTEST_Q6CurrentCoreResume0914__023527.log` completed at
02:44:14 PDT with exit 0. Peak tracked physical footprint was 3.96 GiB, swap
stayed 0.49 GiB, and owned group/tracked/watchdog survivors were all zero.
Raw log is retained under the private `mtp-swift-2026-09-04/logs` evidence root.

- 180 actual 2L GDN projection payloads, three seeds each: 540 exact raw-F32
  comparisons; 1,694,085,120 payload bytes read without a full model load.
- Generated projection widths, raw HC consumer rounding, compiled/strided/
  broadcast layouts, CPU and unsupported-shape fallbacks passed.
- Existing q4/q8 dense/gathered tests passed; three SDPA shared-mask tests
  passed, including 180 exact partition cases.
- Enabled Swift Testing summary: 14 tests, with disable-only and optional
  timing tests skipped. Disabled summary: seven tests, with actual payload and
  optional timing tests skipped. The disabled route was reference-exact.

This is numerical/dispatch-fixture evidence, not end-to-end model-speed proof.

## Change and remaining app gate

The local development app on engine bed37a22 and core0c3e7c18 still executes
the promoted-F32 q6 path. A retained bounded dispatch capture on the same core
counts72 dense q6 QMV kernels and144 F16-to-F32 metadata copies per token.
This is unnecessary materialization, not proof that q6 explains all historical
slowdowns or that every projection in a named quant uses that quant's bit width.

The candidate retains the packed q6 weights, reads BF16 inputs and F16 affine
metadata directly, and preserves F32 accumulation **and raw F32 output**. The
input bias sum widens operands before addition to match the old F32 route.
It does not add a BF16 rounding boundary before HC residual arithmetic.

Eligibility remains affine, GPU, one input row, group64, input width divisible
by512 and output width divisible by8. CPU, unsupported rows/groups/layouts,
gathered q6 and the existing q4/q8 routes retain their previous behavior.
`VMLX_DISABLE_MIXED_Q6=1` selects the old path in the same app binary for controls.
The mixed kernel is present in both generated JIT and precompiled Metal sources.

The earlier BF16-output variant was rejected after changing actual answers.
That variant and the separately rejected shared-expert F16 guard expansion are
not part of this change. QSA, PLE, sampler, model weights, cache precision,
memory limits and MTP controller are unchanged. MTP remains Off in AR proofs.

Numerical gate: generated actual-width projections, raw HC consumer precision,
compiled/strided/broadcasted inputs, CPU/unsupported fallbacks, q4/q8 controls,
existing long-SDPA mask tests, and180 actual2L q6 GDN payloads at three seeds.
Fixture success alone is not an app-speed or model-quality claim.

The current app must then be repinned in all six dependency locations and
rebuilt as a local optimized development app, with exact source/binary/Metal
identities and actual named q6 dispatch. Required live evidence includes
counterbalanced repeated short/long AR controls, exact natural-stop output and
cache frontier,1s/5s stream windows, actual UI follow-up, other-quant fallbacks,
and relevant VLM/cache quality. No public release or merge is authorized by a
numerical-only result.
