# Qwen AR checkpoint — measured gains, broader qualification PARTIAL

## Tested code and app

Engine `67ccb4b347a23820b838a98f0c195b0c29c676d2`; Osaurus
`ee75e3987c106d745442bb0bd236b5295646440e`; unchanged MLX core
`73312d3e9ad0bd2e1bdf9a08d91e25571e255964`. All six app dependency/tripwire
references contain that engine SHA. Documentation-only commits after it do
not change the tested runtime tree or require a new dependency pin.

Local M5 Max/128 GiB optimized development app, isolated bundle
`com.dinoki.osaurus.mtpcalibration467`, binary SHA256
`99be13b2a18402ca70eb8e1da11d6947e2bd0d3665bd5377dd7b8eb197cf59dd`, UUID
`13D5B43C-22BE-3CE1-AF0C-74A5F7638306`. Build receipt
`SWIFTTEST_QwenARDefaultDevApp0912__213440.log`: exit 0, no DEBUG test hooks,
peak tracked physical footprint 20.46 GiB, swap unchanged, cleanup 0/0/0.
This is not a release, tag, installation or replacement of the user's app.

Source: `GenerationActivity` scopes in `generateLoopTask` and the batch loop;
`RuntimeEnvironment.value`; `Qwen4ExpEarlySubmission.allows` and the explicit
AR forward; `Qwen4ExpRotaryContext`; `Qwen4ExpQSA.selectedTokenMask`;
`Qwen4ExpHCCombine`. Live default logs emitted early submission (48 layers,
caller stream), rotary reuse, exact HC (product rounding preserved, FMA off),
QSA (scores/selection unchanged) and parallel PLE-row dispatch. Admission uses
actual tensor shapes and execution phase, not quant-name allowlists. Weight,
sampler and cache formats are unchanged; MTP was off.

## Final-binary observations

Two measured complete 1–250 counts per cell after an excluded warmup; each
contains 891 output tokens and naturally stops. Exact request/output bytes,
input counts and seed 829 match the prior opt-in observations. Executed
bundle sampler: temperature 1, top-p .95, top-k 20, no repetition penalty;
thinking explicitly off for this diagnostic. No output or sampler masking.

| Model / mode | 34 input mean tok/s | 8,339 input | 34,939 input |
|---|---:|---:|---:|
| 2L, optimization variables absent | 46.1678 | 42.9902 | 41.4721 |
| 4S, optimization variables absent, first series | 46.6458 | 38.0823 | 38.6558 |
| 4S, explicit opt-in, subsequent fresh process | 50.4058 | 45.7366 | 43.0771 |

2L exceeds 40 tok/s in all six measured default rows, not every second or
every workload. Its one-second iterator ranges are 41–49 / 38–45 / 37–44.
4S explicit ranges are 49–52 / 43–48 / 39–45. Five-second minima/peaks and
longest token/client-chunk gaps are retained separately. Iterator timestamps
are not GPU timings or UI text-chunk rates. Repeated long requests restore
native prefixes; their preparation time is not cold-prefill throughput.

Retain the slow 4S default rows: 8K repeats were 38.3514/37.8131. Later in the
SAME default process, without changing flags or restarting, an exact-request
recheck measured 47.2537, with one-second windows 46–48. Both modes dispatched
the paths above. This rules out an always-missing default path, not transient
performance variation. Sequential groups are not randomized thermal controls.

Prior same-binary 4S full-component off/on means were 39.6710→47.8775 short,
36.0027→43.6431 at 8K and 33.4495→39.8239 at 35K. Prior 2L early-only off/on
means were 36.3948→45.3106 short and 32.8596→39.0530 at 8K. Those controls
use older binaries and do not replace this final-default qualification.

## UI, correctness and adverse results

PID-scoped Accessibility/osascript, screenshots, persisted history and raw
traces bind real UI runs. 2L ice/connected iron/post-Stop recovery naturally
completed at 42.4481/41.7031/44.3000 tok/s (167/174/35 tokens). 4S default
ice/iron completed at 46.2903/45.7103 (253/226 tokens); post-cancellation
recovery completed at 43.0065 (47 tokens). Initial-load cancellation separately
settled to no loaded model/no active inference. Activity ends and subsequent
empty inference state were observed, including cancelled cache reconstruction.

This is not a full model-quality pass:

- Per quant, final-default frozen prose parity is 3/3 and connected real-image
  output parity is 2/2. Image factual quality remains 0/2 in both arms for
  both quants (wrong OCR/scattering); 2L prose case 2 mis-expands AR.
- 4S UI iron contains an incorrect buoyancy inequality and answers about
  liquid iron. Its count request declined via `complete` without a normal
  assistant answer: task FAILED, 192 raw tokens at 44.7752 iterator tok/s.
  2L's stopped UI count used two columns despite the requested one per line.
- 4S Stop was observed in the cache tail and a read-only tool-continuation
  preparation, not independently during mid-token decode. A DNS request
  naturally produced 1,070 tokens at 44.3223 iterator tok/s (1s:41–48), then
  Stop cancelled finalization. UI estimated 1,201 tokens/45.3 tok/s: those
  displayed values are not substituted for the raw trace. A later unrelated
  `osaurus_help` call is also a quality defect, not a desired DNS tool action.
- Two default 4S combined/UI sessions hit the unchanged private 8 GiB free-page
  floor and exited 124 (peak tracked footprint 62 GiB, cleanup 0/0/0). They
  remain FAILED/INCOMPLETE whole sessions. Swap stayed 0.49 GiB and sampled
  pressure normal. These were harness stops, not inferred app crashes. No
  production memory setting was altered. No general RAM-soak pass is claimed.
- Separate final default 2L, explicit 4S API, and default 4S cancellation/
  recovery sessions ended by AX Quit, exit 0, cleanup 0/0/0; peaks were
  55/62/60 GiB respectively, unchanged swap. The user's running app was untouched.

Effective cache telemetry is fp16, 12 KV plus 36 Mamba layers, disk-backed
restore, paged RAM disabled and zero TurboQuant KV layers. Final 2L disk
counters: 18 hits/100 misses/27 stores; final 4S cancellation run: 4/93/4.
Separate companion hits were zero. Boundary rederive appears in engine logs
despite zero separate companion-rederive counters; these are not sidecar-hit
or no-rederive claims. Actual 4S loader materialization remains BF16 while 2L
preserves affine metadata under mmap. Mixed bit/group sizes are not normalized
into one model-wide quant label.

Three adapted original Python essay prompts also naturally stopped on 2L at
47.5478/46.3827/45.9404 tok/s (553/462/466 tokens, original temperature 0,
max 700, no seed). Model-ID/thinking-field adaptations and omission of the
unsupported prefix-bypass field are explicit in receipts. Their invented
code explanations fail factual grading; this is not an exact cross-runtime
comparison. Python's retained 56.5 tok/s remains a target. Config/index/
generation hashes match between the 2L receipts despite differing folder names;
full shard payload identity has not been freshly established.

## Tests, receipt handles and remaining scope

`SWIFTTEST_QwenARDefaults0912__212412.log`: 43 Swift Testing cases in seven
suites plus one governor XCTest, zero failures; 72 exact rotary configurations
and 240 expected fixture IDs. It covers default/opt-out, environment precedence,
activity lifetime, rotary text/media, exact QSA/HC and mixed GDN/QSA/PLE disk
continuation. The optimized unit build uses a pre-existing DEBUG-only test
hook; the app does not. This is not real-model MTP performance evidence.

Private receipt root: `/Users/eric/vmlx-private-evidence/post-1653-qwen38-audit`.
`checkpoint-2l-default-{short,8k,35k}-comparison-0912.json`,
`checkpoint-4s-default-explicit-{short,8k,35k}-comparison-0912.json`,
`checkpoint-4s-default-8k-recheck-comparison-0912.json`, final-default per-quant
reference/vision comparisons, `checkpoint-2l-default-ui-*`,
`checkpoint-4s-default-b-ui-*`, and `checkpoint-4s-default-c-*` retain complete
requests/answers/SSE/iterator/history/screenshot/cache/engine evidence.
Supervised `.log/.mem/.procs` receipts are in the sibling
`mtp-swift-2026-09-04/logs` directory. Detailed engineering handoff stays local.

App CI run 34738289604 at ee75e398 completed test-core, test-cli, swiftlint,
shellcheck, test-packages, test-evals and test-statspack successfully. Engine
CI is NOT green: formatter and Linux matrix failed; Mac/CUDA queued. No Mac
runners or required engine checks are configured. Bookworm's undefined
`mlx_stream_run_with` also appears in merged base PR469 (jobs
103674228133/103547696587), with unchanged Linux Cmlx/Stream/workflow sources.
Other matrix failures are not silently declared identical. No admin bypass.

Next work remains AR: investigate repeated 5–6 second post-stream checkpoint
reconstruction, sustained-context variance, grouping/materialization overhead,
UI cancelled-token/rate estimation, and the retained quality failures.
Sustained MTP, full VLM/video, all-quants speed floors and the separate Qwen35
crash are not qualified by this checkpoint. No release is requested or performed.
