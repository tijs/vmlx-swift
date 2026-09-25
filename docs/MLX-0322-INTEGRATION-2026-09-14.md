# MLX 0.32.2 integration checkpoint — PARTIAL

NOW: Integrate pinned core0.32.2, the matching C ABI and Swift compatibility in
`feat/mlx0322-integration`. Preserve the JANG/mmap fork and the existing Qwen
AR optimizations. The affine stream regression has app evidence; full-family
qualification and the engine/application merge remain incomplete.
DO NOT: Release, change native sampling/precision, replace the fork with stock
MLX, or mix a larger prefill chunk into the dependency-control measurement.
BATCH OWNER: Exact-source compatibility, development-app evidence and merge gates.
NEXT: Complete the final wrapper regression run, consume the final engine pin,
and rebuild locally. Keep engine/application PRs draft while runtime gates remain open.

## Current checkpoint — September 14

Eric accepted the retained model checkpoint and requested repinning with one
scope correction: only Flash Next should start MTP Off, with selectors visible;
Qwen27B must retain existing behavior. The shared engine default remains Auto.
Osaurus owns the metadata-scoped picker default and preserves explicit choices.
This scope correction does not claim the broader limitations below resolved.

Core [#9](https://github.com/osaurus-ai/mlx/pull/9) and C ABI
[#1](https://github.com/osaurus-ai/mlx-c/pull/1) are merged. Fetched heads
`0b9bdfb5858ecfac30916aeb23cc7c595d157a4f` and
`ab37d9dce0b74c48ff1948145efdb6c59e678627` contain the tested pins
`c0a51a085b9252dc6469afe1aed772b1f538c9b0` and
`2d783ac38713458eae2067ffff9ef8ebbff2ec70`, respectively, with identical trees.
Engine #474 and Osaurus #2745 are still draft/unmerged. No release/tag/install.

Development build G, receipt `SWIFTTEST_MLX0322AffineStreamDevAppG0914__064628.log`,
bound app `de65c71c82dbbfc8e972d76f9ddb7a5f0c37bcf9`, engine
`4d01ed7832e62c73ae1477aa11faefa471c9d4f2`, and those exact native pins.
Binary SHA256 `e4bb60b696b44fa27b68d6f0a908b57b8197252616a9ef835e1080a0155f5cb3`,
UUID `564FEC05-4CBF-3EBF-A1B2-8F2A33CA10C7`, metallib
`02ea075fab6e847ba1f5cc8c410657506b63309257a24efbc7eb6f69380dbe6f`.
Build, dependency identity, ABI symbols and ad-hoc strict signature checks ended
exit0; compiler peak21.46GiB, swap unchanged0.49GiB, zero owned survivors.
The compiler free-memory floor was explicitly lowered40 to32GiB after two
unnecessary aborts; cap28GiB/two WMO threads remained. This is NOT a change to
the inference8GiB free floor/66GiB tracked cap or pressure/swap gates.

Clean app-owned AR, MTP off, native temperature1/top-p0.95/top-k20, no repetition:

| Quant/workload | Retained old-stack control tok/s | G candidate tok/s |
| --- | --- | --- |
| 2L short,34 input/891 generated | 46.854816;47.539888 | 51.372102;49.895430 |
| 2L8K,8339 input/891 generated | No completed matching old8K row in this series | 47.921805;47.282857 |
| 1L short,34 input/891 generated | 47.805748;47.289407 | 47.047354;47.115732 |
| 1L8K,8339 input/891 generated | 44.845929;44.994549 | 45.056578;44.909111 |
| Fresh2L short follow-up control | 47.206161 | 47.238209 |

These count rows were exact1..250 with natural EOS. Candidate1s windows:
2L short48..52,8K46..49;1L short46..48,8K43..46; no empty windows.
Warm8K restore reported8332+7,12KV+36Mamba layers, required native disk-backed
SSM state, zero TurboQuant/paged layers. Do not relabel these as fresh long prefill.
Real osascript/Accessibility UI2L explanation/follow-up ran45.156/44.976tok/s,
screenshots and complete answers inspected. UI1L first turn44.761, but its
follow-up29.566 is a retained performance failure, not just the displayed rate.
Scientific inaccuracies in some answers are not quality passes.

Fresh1L exact3536-token submitted-input replays also slowed on the unchanged
old binary: old27.337/24.167/23.890, new29.025/28.063/27.720tok/s. Later2L
controls returned to47+. The cause of that variability is not established.
Do not claim stable50tok/s, universal parity, or transfer Python timings.

Instrumented G capture separately restored one queue and exactly199command
buffers/token across16complete intervals, versus the broken integration's two
queues/~320.889. Observed G rate51.668911 is instrumented, not a clean benchmark.
The source correction is the three explicit cast streams in core `mlx/ops.cpp`;
the negative/numerical receipts below isolate it without changing dtype or math.

### Unclosed runtime gates

- Both old/new4S attempts aborted before a final speed receipt at the unchanged
  8GiB free floor:6.9/6.7GiB free,57.29GiB peak each,normal pressure,flat0.49swap.
  Receipts `SWIFTTEST_mlx0322_4s_control_a_affineg1__074647.log` and
  `SWIFTTEST_mlx0322_4s_candidate_a_affineg1__074756.log`, zero survivors.
  This does not establish an OOM or a new-version memory regression.4M/6S unmeasured.
- Matched fresh35K long-context runs lack completed old/new qualification under
  the same memory guard. Earlier failed/partial observations remain retained.
- Actual UI cancellation metadata failed to settle within30s in BOTH binaries,
  despite idle backend. The candidate once ended its recovery after53tokens,
  midword, on actual native EOS. That is an unresolved semantic failure.
  Three exact-submitted3391-token raw replays per engine all ended complete;
  they do not reproduce Stop scheduling or matched RNG. Old44.82/45.16/44.54,
  new43.19/44.60/40.29tok/s; keep the slower repeat. Raw route emits no decoded
  prompt dump, so equal submitted bytes/count do not assert exact runtime token IDs.
- Actual-image OCR failed in both previous/new stacks; no blanket VLM claim.
- Concurrent RNG, CUDA and multi-host distributed execution remain unqualified.

### Final diff/build review

`27ff9205` excludes the new native stream regression test from standalone Xcode
framework membership. `SWIFTTEST_MLX0322AlternateBuildsB0914__075126.log`
then records `BUILD SUCCEEDED` at line86125. The subsequent CPU CMake build
caught a macOS-SDK/backend mismatch at `WiredMemory.swift:722`.
`cebccaef` makes the conditional follow CMake's Metal-disabled source selection;
ordinary SwiftPM/Xcode memory policy is unchanged. `bd832e02` prevents the CPU
example from eagerly creating the unused GPU stream.
`SWIFTTEST_MLX0322CMakeCPUExample0914__080130.log` builds and executes the CPU
example with exact arrays/index value4,exit0/zero survivors. Earlier failures
are not counted as passes. Full proposed diffs were whitespace-checked.

Full wrapper receipt `SWIFTTEST_MLX0322FullWrappers0914__074430.log` had one
compiled-RNG failure because public `compile()` is intentionally eager without
opt-in. Same binary20b1a455 with explicit test-process compiler opt-in in
`SWIFTTEST_MLX0322FullWrappersCompileOptIn0914__074608.log` reported544XCTest,
two existing concurrent-RNG skips,zero failures,plus83SwiftTesting passes.
`f80dac25` scopes that RNG test to the existing trusted compiler API, preserves
both RNG assertions, and adds an actual call/trace-count check for public policy.
`SWIFTTEST_MLX0322FinalWrappersB0914__080818.log` rebuilt engine`f80dac25`:
both default and opt-in runs reported545XCTest,2existing skips,0failures,
plus83SwiftTesting passes each. A separate hard-disable-over-opt-in process
passed the public-policy assertion. The linked-version test executed0.32.2.
Executable SHA256`719112f85ef8c97f2ad3cec9e3efd6d2c0c14d283b715a700cb1fa81b24a616a`;
metallib remains02ea075f. Exit0,4.18GiB peak,flat0.49swap,zero survivors.
The first rebuild invocation omitted the established DEBUG-only test hooks
and failed to compile; the corrected invocation restores those test flags.
The actual development app has no such test flags. App policy and numerical
tolerances are unchanged. The next engine commit only records this document.

## Source lineage

- Core stream follow-up: `c0a51a085b9252dc6469afe1aed772b1f538c9b0`.
  Pass the requested stream to all three affine quantized-matmul fallback
  casts. This changes scheduling only, not promotion or kernel eligibility.

- Engine starting point: `9e48d907e45f3d721b3faa42d9f6500bab16224b`.
- Core fork starting point: `be526f81f5534b3447d4882871cac279af2ad14a`.
- Core release integrated: `1f8e74e3f12f31365464a6867c6579f0e9b29d85`
  ([MLX0.32.2](https://github.com/ml-explore/mlx/releases/tag/v0.32.2)).
- C ABI starting point: `7bede8f1491384bb580f87d1d510f80cbd53660d`;
  integration target `c74db5307cc8ce122f48d97ef951b30578674e7f`.
- Integrated core merge: `1c4fcf773c00aa544bcddade6281fbd311a599c7`;
  C ABI: `2d783ac38713458eae2067ffff9ef8ebbff2ec70`.
- Core follow-up `118b983596b634409315fee46b36dcda2245e12a` applies the
  repository's clang-format21.1.8 rules and makes the invalid-enum path in
  cold-memory advice explicit for GCC. Valid advice mappings are unchanged.
  Core CI run34840230692 reports lint, both Linux sanitizer jobs and both
  Fedora architectures successful; macOS/CUDA jobs are skipped, not proof.
- Upstream Swift comparison: [#450](https://github.com/ml-explore/mlx-swift/pull/450).
  Keep this engine's Swift-tools6.1 manifest and integrated runtime products.

## Required adaptations

| Boundary | Adaptation |
|---|---|
| Quantized kernels | Preserve affine1 inference packing, mixed q4/q8 metadata, raw-F32 q6 and Metal-only admission. Preserve the32-lane exact q6 reduction for eligible promoted verifier rows; leave other wide and matrix paths intact. |
| mmap | Retain the fork's regions/advice, excluded keys, safe unaligned handling and array-owned mappings. No model-weight rewrite. |
| Stream lifetime | Use upstream cross-thread stream handles under the Swift evaluation lock; native defaults are now thread-local. Restore the target device's previous default. Preserve zero-argument Stream() as the effective Swift default, including TaskLocal scopes; generation/cache/cancellation drains explicitly select that same stream. |
| Compiler cache | Acquire evalLock before the recursive instance lock. Record stable identities and weak handles for every native worker cache used, then erase the function from those caches on destruction. Preserve the compile trust policy. |
| Custom kernels | Adopt source/options-hashed libraries; retain bound buffers through command completion and retired library/pipeline owners. Preserve output shapes in both common Metal and CUDA factories. CUDA is not execution-qualified here. |
| C/Swift surface | Adapt cumulative-axis, FFT normalization, SDPA forceFused defaultfalse, median/trace and Data SEEK_END. Update distributed-group output-parameter ABI with owned typed handles; multi-host execution remains untested. |
| Generated sources | Regenerate copied headers, embedded JIT sources and fresh AOT shaders from these sources. Discover headers with the installed Metal toolchain, but keep the macOS14 app target. Exclude optional newer-Metal fast-fence AOT; do not enable MLX_METAL_FAST_SYNCH=1 with it. |
| Alternate builds | CMake consumes the same pinned core/C ABI, not stock mlx-c0.6.0 or the removed stream shim. Framework membership/public headers regenerate from actual SwiftPM sources, with the framework owning its distributed C entry points directly. Linux CMake CI initializes the pinned submodules. |

## Live numerical evidence so far

The first local development app exposed a regression: matched 2L short AR
measured28.8559/28.4262tok/s against retained control46.8573. Instrumented
captures showed about321command buffers/token and two queues versus199
and one queue. The secondary queue's observed kernels were BF16-to-F32 casts.
Core `mlx/ops.cpp` omitted `s` in the three affine fallback casts although
the quantized matmul itself selected `s`. The explicit Swift stream no longer
coincides with the native default under the new thread-local runtime.

`SWIFTTEST_AffineStreamNegativeB0914__061714.log` reproduced144wrong-stream
assertions across48CPU/GPU cases (bits2/3/4/5/6/8,groups32/64,rows1/3).
Numerical equality still held. With only those three stream arguments added,
`SWIFTTEST_AffineStreamFixedFull0914__061831.log` ended exit0:674native
assertions,52XCTest cases,24/21/24Swift Testing selections, and the separate
strict TF32-disabled384-case matrix. No tolerance was loosened. Test binary
SHA256`20b1a455dbb9a0c9d6442175e2deea32df3468094205aebb0e27db33779717b3`;
metallib unchanged below. Peak tracked2.25GiB,swap flat.49GiB,zero survivors.
The native regression is registered in core CMake tests. Formatting then
wrapped one test expression only; product source was unchanged.

This was the pre-G status; the current G app receipts above supersede it.
The old and new apps both hit the8GiB free-memory floor in long runs; this
does not establish a new memory regression. Real-image OCR was incorrect in
both apps, so those rows are not full VLM passes. No engine/app merge claim.

Private evidence root:
`/Users/eric/vmlx-private-evidence/mtp-swift-2026-09-04/logs`.

`SWIFTTEST_MLX0322IntegrationTestsF0914__041253.log` records a freshly built
test executable SHA256
`ec5377ef1b3e9fb9d63e6eb28a34789293d4341b555b0f3907d55369d9a8b9d9`
and metallib SHA256
`02ea075fab6e847ba1f5cc8c410657506b63309257a24efbc7eb6f69380dbe6f`.
Its51 XCTest cases had zero failures. The new384-case quantization matrix
ran in two modes: default backend with an explicit reduced-precision budget
for48eligible gathered-M33 cases, and strict2e-4 throughout with
`MLX_ENABLE_TF32=0`. The largest default-mode absolute difference was0.0007005632.
App TF32 policy is unchanged; see the [MLX precision contract](https://ml-explore.github.io/mlx/build/html/usage/precision.html).

Existing exact q6 equality tests were not loosened. RMSNorm, shared-mask
attention partitions, mmap GPU computation, FFT/Data, cumulative scans,
custom-kernel identity/shapes and cross-worker compiler-cache deletion have
focused execution receipts. This is not full model quality or speed proof.

The extended F selection had one stale group128 rejection test, inherited from
before production support in c6ae0e85/#446. The integration corrects that
assertion to group256 and adds positive group128 numerical/row-isolation cases.
`SWIFTTEST_MLX0322IntegrationTestsG0914__042059.log` ended exit0. It reran the
core matrices and recorded zero issues across the21-test HC/GDN/PLE/expert/rotary
and24-test cache selections. Exact checks included GDN336,HC768,rotary72 and
QSA sequence180. Both added group128 layouts had zero batched-versus-single-row
difference for rows2/3/4. Tiny Ling3 disk restore37+suffix5 matched one-shot42;
this is not full-Qwen restore proof. Real installed-model PLE tests remain pending.

Final G test executable SHA256:
`d495c5a622ad6acdff1f7a451b592985a3285c7b46f028b26f2ab2e67f42ffda`.
The fresh metallib hash is unchanged. The bounded supervisor recorded4.20GiB
peak tracked footprint, flat0.49GiB swap and zero owned survivors.

## Remaining acceptance gates

Final caller review caught a regression before app promotion: engine f7883254
made zero-argument Stream() allocate a new queue. Eight cleanup sites drained
that unrelated queue. The live negative control
`SWIFTTEST_MLX0322DrainNegativeB0914__051251.log` reproduced six wrong native
stream identities and three GPU arrays still unavailable after the false drain.
The replacement tests CPU/GPU, nested TaskLocal defaults, availability before
readback, and exact values. Build B was stopped (exit130) with zero owned
survivors; it must not be used as application evidence. No connection to earlier
speed or crash reports is established by this newly introduced defect.

The corrected full selection is
`SWIFTTEST_MLX0322DrainFixedFullI0914__051403.log`, exit0 at05:21:05.
It records52 XCTest cases (including the six-case default-stream regression)
and the24/21/24 focused Swift Testing selections with zero failures/issues,
plus the separate strict TF32-disabled384-case matrix. Test executable SHA256:
`23dbb00f6d5dd906ad6f6723d0e481e3aad746dbeb5de4baebf9dfa9ad8bd3b4`.
Fresh macOS14 metallib SHA256 remains
`02ea075fab6e847ba1f5cc8c410657506b63309257a24efbc7eb6f69380dbe6f`.
Tracked peak4.01GiB, swap flat0.49GiB and zero owned survivors. Opt-in
installed-model branches are still not full-model execution evidence.

Fork PRs: core [#9](https://github.com/osaurus-ai/mlx/pull/9), C ABI
[#1](https://github.com/osaurus-ai/mlx-c/pull/1), engine
[#474](https://github.com/osaurus-ai/vmlx-swift/pull/474), consuming app
[#2745](https://github.com/osaurus-ai/osaurus/pull/2745).

The first development build at engine73e8d811/appd9f10ed57 stopped at the
16GiB compiler cap: `SWIFTTEST_MLX0322DevApp0914__044128.log`, exit124,
16.49GiB tracked footprint,47.5GiB free, normal pressure and flat0.49GiB swap.
It confirmed actual WMO compiler threads2 and zero owned survivors. No app
runtime row exists from this build. The final revision will retry with the
prior28GiB build cap while preserving40GiB free-memory and pressure/swap gates;
these are compiler-process limits, not changed inference memory settings.

1. Retain the bounded extended test receipt above; no test driver remains active.
2. Source-checkpoint and consume exact core/C/engine revisions in all six app pins.
   No upstream submission or public release is intended.
3. Fresh local optimized development build, actual dependency checkout identities,
   binary UUID/hash, metallib hash and new ABI symbol presence. Observe actual
   compiler thread count; xcodebuild jobs alone did not bound previous WMO memory.
4. Local app AR/MTP-Off controls using identical bundles, tokens and native sampling:
   short and long contexts, repeated runs,1s/5s min/median/peak and longest gaps,
   footprint/read pressure, coherent natural-stop multi-turn/media/cache rows.
   Account individually for JANG1L/2L/4S/4M/6S. Do not transfer Python timings.
5. Only then separately qualify masked D256 attention and bounded larger prefill
   chunks against512, preserving cache/frontier/media positions. Sustained MTP
   remains a subsequent unit, not proof furnished by this dependency update.
