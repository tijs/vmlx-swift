# Spark 2.5 / Raptor activation fusion checkpoint

No release or tag. Merge only a measured, qualified improvement. This PR is not ready to merge.

## Contract

The candidate fuses erf GELU and its following multiply, preserving BF16 rounding at each intermediate. Packed weights, projection quantization, residual precision, sampling, templates and cache policy are unchanged. CPU/non-Metal, other dtypes, broadcast/empty inputs and traced transformations retain the reference expression. `VMLX_SPARK_GELU_REFERENCE=1` is a diagnostic reference-path switch read before the first forward.

Current revision restricts fusion to at least 128 rows in the sequence dimension; shorter/decode inputs use the reference expression. f818a4ea below is the earlier unrestricted experiment. The tracer query is metadata-only. Eight isolated helper tests pass, including all 65,536 BF16 encodings with four multipliers (NaN payloads excluded), noncontiguous inputs, CPU/scoped streams, compiled/VJP/JVP/vmap paths and empty arrays. Fresh Release RunBench built successfully from a clean tree; source hashes and binary are archived.

## Evidence and rejected results

Evidence root: `/Users/eric/vmlx-private-evidence/raptor06-speed-2026-09-23/spark-mlp-r30`.

- The original CustomFunction wrapper (2420da85) slowed native parsed generation to 43.9 tok/s versus baseline median 99.3. It is rejected.
- The direct tracer-aware replacement has exact serial synthetic MLP outputs: reference/candidate median 4.995/4.570 ms for a decode-shaped 36-layer chain, 2.806/2.638 ms for M128 and 7.794/7.254 ms for M512. A separate GPU stream reproduced the synthetic benefit. These are not whole-model speed measurements.
- The replacement still slowed a complete 7,032-token parsed hash-table response to 72.3 tok/s. The same binary with reference activation completed at 94.0 tok/s. Both stopped natively with closed reasoning and matching visible answer. This single pair was initially attributed to fusion, but the later reference-only shape also measured 75.1 tok/s. Causality is unresolved; do not promote the unrestricted default decode path.
- One 10,031-token prefill/cache pair measured 2.458 s reference and 2.341 s fused. Both two-turn raw outputs and all 685 tensors across five retained checkpoints matched exactly. Quota eviction and resume occurred. One pair does not establish a statistical speed improvement. Physical footprint reached 3.86/4.32 GB; this does not establish a low-RAM family claim.
- The unchanged baseline word-count workload looped to 8,192 tokens without a visible answer. Retain that failed coherence row; no sampling or prompt masking was added to hide it.

The repeated same-binary long-prefill ABBA ran reference 2.636/2.456 s and fused 2.436/2.348 s, with eight native-stop turns. The raw-submit pair ran 94.5 tok/s fused versus 90.3 reference; host-delivery median/p95/max were 10.584/10.964/14.340 ms fused and 11.076/11.456/14.054 ms reference. These narrow wins do not excuse the parsed-decode regression. The initial shipping candidate therefore keeps decode on the reference expression. The eight revised focused tests exercise eligible prefill shapes, 127/128/129-row boundaries, exhaustive BF16 values and transformations; all passed.

The prefill-only 8a254032 native parsed row completed correctly at 75.1 tok/s even though the 44-token prompt and decode shapes took the helper reference path. Spotlight workers were active after copying app build directories; those owned copies were renamed `.noindex`. This is a potential confound, not an established root cause. The next revision returns the original MLP expression directly for short/decode input before evaluating `up(x)`, preserving graph construction and temporary lifetimes. Eight focused tests pass; full runtime/app proof must use that final revision.

The final size guard uses `Int32.max`, matching `MLXFastKernel`'s signed grid conversion. The ninth focused test constructs a lazy broadcast with 2^31 elements and verifies reference fallback without evaluating or allocating the logical tensor. All nine focused tests pass. Earlier eight-test receipts remain historical.

## Final qualification: engine 41abcab2, Osaurus f682f5be

The exact final-source Release RunBench build passed and all nine isolated actual-helper tests passed. The isolated Release Osaurus build completed in 805.25 seconds with no source changes during compilation. Osaurus CI run 36063750305 passed all eight jobs. Four engine Linux CMake builds passed; engine CI is not green: repository-wide swift-format changes (722 files, including test formatting) fail lint and macOS/CUDA runners remain queued. Engine CI was explicitly waived for this task; no repository-wide formatting changes are included.

Final same-binary prefill ABBA at 10,031 prompt tokens:

| Run | Reference seconds | Fused seconds |
| --- | ---: | ---: |
| First pair | 2.794 | 2.530 |
| Second pair | 2.485 | 2.342 |

Mean prefill latency fell 7.7% in this small two-row-per-arm comparison. Earlier final-source ABBA was 3.185/2.453 reference and 2.535/2.338 fused; the slow first reference demonstrates startup variability and must not be used to inflate a steady-state claim. Across both final matrices all sixteen turns stopped natively; each of three comparisons per matrix matched full raw outputs and all 685 retained checkpoint tensors. These are M5 Max/macOS26.4 measurements, not an all-chip claim.

Decode is not promoted as a speedup. The first final parsed candidate measured78.4 tok/s; the same-binary reference97.0, fresh archived baseline97.1 and repeat candidate95.4. The78.4 outlier remains unexplained. Final raw-submit candidate and reference both measured88.0 tok/s over7,032 tokens with identical25,113-character full output. Host-delivery median/p95/max were11.337/11.905/14.384ms candidate and11.363/11.763/14.573ms reference. These are host-delivery timings, not GPU duration or synchronization counts; the raw protocol deliberately contains the native reasoning closer.

The actual development app completed four native-stop turns at98.7/98.1/96.7/98.2 tok/s with clean visible answers. Disk Cache was disabled and saved in Settings, a reload/turn completed, caching was restored and saved, and the same app/profile was quit and relaunched. Restart restored14,030tokens and processed229remaining; TTFT1.05s. Effective topology was27rotating+9full KV layers, paged off and disk-backed restore. No low-RAM-family claim: observed app footprint after first turn peaked4,032,514,880bytes, and first-turn peak was not sampled.

No tools, schemas, prompts, routing, generation defaults or cache formats change. Full frontier agent-loop comparisons are not applicable to this exact-arithmetic-only pin; current-head hosted eval tests passed and local app/runtime/cache proof is retained. No old model eval scores are represented as new evidence.

Private evidence root: `/Users/eric/vmlx-private-evidence/raptor06-speed-2026-09-23/`:
- `spark-mlp-r30/grid-tests-run-receipt.json`, `candidate-grid-build/build-receipt.json`
- `spark-mlp-r30/grid-final-cache-review.json`, `grid-final-cache-r2-review.json`, `grid-final-raw-comparison.json`
- `spark-prefill-app-r31/live-receipt.json`, `ui-relaunch/persisted-turns.json`, `hosted-final-run.json`, `eval-applicability.json`

## Follow-ups kept separate

- Investigate decode run variability; do not claim synchronization counts from the limited CPU sample.
- Measure attention headwise gate/layout fusion, shared-input MLP projections and actual TensorOps dispatch before promoting another speedup.
- Attribute quota-pressure CLI cache tail (roughly3.4–4.4s) separately from decode. App restart/restore passed; this does not close all SSD churn/eviction cases.
- The prepared profile initially contained runtime port18331 without matching legacy server.json, so first launch listened1337. UI save and subsequent relaunch correctly used18331. Keep inconsistent-fixture/bootstrap recovery separate from this kernel; it is not evidence ordinary save/relaunch regressed.

No releases or tags. Source and live evidence remain distinct; retain negative rows and merge each independently proven change.
