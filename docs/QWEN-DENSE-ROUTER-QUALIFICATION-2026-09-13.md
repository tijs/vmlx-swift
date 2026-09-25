# Qwen dense router: qualification checkpoint

Status: PARTIAL. Source correction and generated numerical coverage; no current
app speedup or family-wide readiness claim.

## Owning path

The base is engine `61a663acaf436c97a3cdf30b4983531f1bbf03a3`.
`JangLoader.dequantizeMoEGates` materializes JANG routing weights as F32.
`Qwen35Language.SparseMoeBlock.compiledRouter` forwards those dense weights to
`Qwen4ExpCompiledMoE.denseRouter`, whose old BF16-only guard rejected them.
The retained local 2L app load reports 48 F32 `mlp.gate.weight` tensors.

The candidate preserves those F32 weights and the ordinary matmul, precise
softmax, partition, gather and optional score normalization. It admits the
loaded signature without rounding weights. Multi-row prefill/verification and
outer compiled traces retain the ordinary path. Invalid shapes and top-k values
fail eligibility before indexing. A typed cache key includes hidden width,
expert count, top-k, normalization policy and weight dtype; weights remain
dynamic compiled-function inputs, not captured constants.

No changes to shared-expert arithmetic/eligibility, routed experts, q6 kernels,
sampler defaults, PLE, cache contents, or the MTP controller are included.

## Why the shared-expert change is excluded

The earlier F16 shared-expert eligibility patch `f8620a90` was reverted in
`d43d001b`: it passed generated fixtures but changed all three seeded real-app
answers in disabled/enabled/disabled controls. The source comparison initially
missed that retained failure. A guard mismatch alone does not justify restoring
that patch. Its actual-model numerical boundary remains unresolved.

## Evidence and limits

Private evidence root:
`/Users/eric/vmlx-private-evidence/post-1653-qwen38-audit`.
Supervised logs:
`/Users/eric/vmlx-private-evidence/mtp-swift-2026-09-04/logs`.

- `SWIFTTEST_MoEBoundary0913B__211925.log`: unchanged app MLX/Cmlx objects;
  144 generated dense-router cases and 72 shared-region cases, including queued
  different weights and varying amplitudes. No mismatches in these fixtures.
  This does not supersede the failed shared-expert app qualification.
- `SWIFTTEST_DenseRouterTests0913__212254.log`: three production-helper tests
  passed on the first F32 eligibility candidate, before the typed-key revision.
  The suite checks 144 amplitude/dtype/top-k/normalization/replacement-weight
  cases, 12 layout/tie cases, source immutability and fallback conditions.
  Peak tracked footprint 3.92 GiB; swap remained 0.49 GiB; cleanup 0/0/0.
- `SWIFTTEST_DenseRouterProductionCost0913__212901.log`: the entire production
  helper, not merely its compiled body, compared against ordinary routing over
  12 counterbalanced rounds. Wrapper overhead erased the isolated-body gain.
  This motivates testing the typed key; it is not an app speed result.
- `SWIFTTEST_DenseRouterTypedTests0913__213309.log`: the typed-key revision
  passed the same three tests (144 precision/queued-weight cases, 12 layout/tie
  cases, eligibility assertions). Peak tracked footprint 4.02 GiB, swap
  unchanged at 0.49 GiB, cleanup 0/0/0.
- `SWIFTTEST_DenseRouterTypedCost0913__214019.log`: 12 counterbalanced rounds,
  16 batches of 48 routing calls per arm per round, using the unchanged app
  Cmlx object. Median ordinary/candidate host construction: 110.748/90.779 us;
  complete batch wall time: 1025.185/996.811 us. Candidate wall time was lower
  in 10/12 rounds. This is a small isolated routing saving, **not model token/s**
  or an explanation of the entire historical AR deficit. Peak 0.77 GiB,
  flat swap, exit 0, cleanup 0/0/0.

The core q6 experiment was already dirty in the working tree. It is excluded
from this change. The private production-cost driver explicitly links the
unchanged app Cmlx object SHA256
`88cf872ccd186f827b620246b3d21cd0ad470ddef60beebec4b74513d3928b96`
instead of that dirty core. The ordinary package test is generated-fixture
coverage, not a clean app-core performance comparison.

## Remaining promotion gates

1. Preserve the completed typed-key tests and full-helper timings above; do not
   substitute them for end-to-end proof.
2. Clean committed engine, all six Osaurus pins, fresh local optimized dev app
   and binary/Metal identity. No public release.
3. Matched control/candidate repeats: identical requests and source binary,
   observed F32 helper engagement, unchanged sampler and source weights,
   natural completion, inspected answers and native-cache receipts.
4. Actual UI sends and follow-up, short and long contexts, and real media/cache
   continuation. Record iterator and stream rates, full 1s/5s windows and stalls.
5. Qualify 2L and 4S first, then 1L/4M/6S. Do not call this a 45/50 tok/s floor
   or merge-ready on generated tests or the isolated router timings alone.
