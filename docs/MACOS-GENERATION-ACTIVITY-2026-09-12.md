# Scoped macOS generation activity — PARTIAL

Final default-app UI/cancellation observations and their limits are in
[the AR checkpoint](QWEN-AR-CHECKPOINT-2026-09-12.md).
2L visible multi-turn/Stop/recovery and 4S cache-tail/tool-continuation
cancellation/recovery emitted scope ends on engine `67ccb4b3`. The final 4S
session logged six begins and six ends before normal AX Quit. Its prefix
checkpoint rederive was interrupted with `CancellationError`; inference then
returned empty and a 47-token follow-up completed at 43.0065 tok/s. This does
not close live batch/MTP, idle-App-Nap-return or all-model qualification.

Runtime change commit: `3bcafad49d880ea21958f9a90322b4102beda84d`.

## Observed failure

A hidden Release Osaurus app remained `PROC_FLAG_SUPPRESSED` during an active
API request. Same binary, same 8,339-token prompt, seed/sampler/cache boundary
and exact 891-token answer: unsuppressed 34.70–34.95 tok/s; suppressed 15.97.
Both runs used normal process nice0. The hidden row's one-second token windows
were 14/16/20 tok/s (minimum/median/maximum), longest token gap 96.59ms.
Read-only native process-flag samples remained suppressed throughout decode.
These are one local Flash-Next JANG_2L workload, not all-quant throughput claims.

Receipt handles: `q6-nice0-candidate-b-8k-hidden-0912`, matching
`q6-nice0-candidate-b-hidden-policy-0912.jsonl`, supervisor
`SWIFTTEST_Q6Nice0CandidateB0912__162644.log`. The earlier q6 kernel experiment
is NOT included in this branch. Its binary supplies the reproduction only;
the corrected app must be compared enabled/disabled using one new binary.

## Change

`GenerationActivity.swift` owns one scoped ProcessInfo activity. macOS uses
`userInitiatedAllowingIdleSystemSleep`: no display or idle-system sleep lock,
no model-name/quant gate, and no permanent activity for an idle loaded model.
Other platforms retain their existing behavior.

- `Evaluate.swift::generateLoopTask` covers deferred iterator construction,
  prefill, AR/MTP decode, cancellation/error exits and cache/GPU finalization.
- `BatchEngine.swift::ensureLoopRunning` covers the non-solo scheduling loop.
- Each scope ends explicitly with `defer`; destruction is an idempotent fallback.
- `VMLX_DISABLE_GENERATION_ACTIVITY=1` is a diagnostic-only process-start opt-out
  for exact-binary A/B. It is not a recommended user setting.
- Debug logs in `vmlx/GenerationActivity` record acquisition, release or opt-out.

No sampler, tensor dtype, quant dispatch, GPU fence, cache representation,
MTP depth policy or inference worker thread is changed.

## Verification boundary

`GenerationActivityTests` ran six tests: normal completion including the tail,
throwing preparation, cancellation, overlapping scopes, idempotent/destruction
release, diagnostic opt-out and exact sleep-permitting options. Receipt
`SWIFTTEST_GenerationActivityLifecycle0912__163812.log`, exit0, peak8.12GB,
cleanup group/tracked/watchdog0/0/0. The warm test checkout contains other
pre-existing changes; these tests exercise the identical new Foundation helper,
not full model precision or throughput. Package.swift explicitly registers them.

## Same-binary live activity isolation

Optimized local Osaurus app source `61e8e6d0`, engine `fe8b231d`, core
`73312d3e`, binary SHA256
`3b5c9f2ab86494b65f13f61e9768187931d70d9b2971b54829676ea18fc51638`:
the identical 8,339-token request and exact 891-token natural-stop answer ran
at 16.0468 tok/s with the activity disabled and 33.5298 tok/s enabled.
Early per-layer submission was disabled in both activity arms. Both processes
were hidden and nice 0. The disabled arm had 55/55 in-decode samples suppressed;
the enabled arm had 0/27 suppressed and logged its scope end.

The disabled arm's rolling one-second token windows were 13–19 tok/s, versus
33–34 enabled. Longest token gap fell from 124.09 to 39.33 ms. These are raw
iterator delivery observations, not display-rate or GPU-occupancy estimates.
The exact-wire/output/stop checks and PID-start-bound kernel-flag correlation
passed in private `early-app-activity-comparison-0912.json`; raw receipts are
`early-app-activity-disabled-hidden-0912` and `early-app-control-a-hidden-0912`.
The disabled supervisor `QwenActivityDisabled0912b__190126` exited 0 at a
50 GB peak footprint, unchanged 0.49 GB swap and cleanup 0/0/0. Enabled control
`QwenEarlyControlApp0912a__183836` exited 0 at 51 GB, same swap and cleanup.

Pending: idle-return observation, real UI cancellation/continuation, batch/MTP
routes and remaining quant/media coverage. The paired activity result is one
bundle/workload, not a family-wide speed floor or complete model qualification.

Apple's [app-level activity guidance](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/PrioritizeWorkAtTheAppLevel.html)
describes scoped user-initiated activities; its
[App Nap guide](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/AppNap.html)
describes priority/I/O reduction and foreground recovery.
