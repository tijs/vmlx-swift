# Gathered Metal matmul validation

The core pin backports upstream MLX #4567 and extends expert-boundary row
scheduling to sorted affine gathered matmul. It also corrects a pre-existing
NAX partial-reduction bounds error found by the expanded regression matrix.
The Swift integration includes regenerated JIT sources and the new offsets
shader in the explicit AOT resource list.

## Scope

- Floating-point sorted gather uses the upstream scheduler.
- Affine sorted gather uses the scheduler when the existing dispatch is eligible.
- Small decode, unsorted gather, explicit LHS gather, and custom quantization
  kernels retain their dispatch. This is primarily a prefill optimization.
- TF32-disabled float tests exercise the non-NAX implementation on an M5;
  they do not establish physical M3/M4 performance.

## Proof checklist

- [x] Numerical comparisons against a CPU reference for expert boundaries,
  empty experts, partial tiles, affine bit widths 1/2/3/4/5/6/8, group sizes
  32/64/128, transpose, sorted/unsorted routing, and explicit LHS indices.
- [x] Reproduce the partial-K error on the baseline and pass after correction.
- [x] Check ordinary affine matmul because it shares the corrected NAX loader.
- [ ] Rerun numerical and scheduler/cache suites at the final engine commit.
- [ ] Stable paired full-model prefill/decode measurements. Existing timing
  controls show host variability; no final full-model speed claim yet.
- [x] Real concurrent requests with bundle defaults, natural stops, throughput,
  physical footprint, and independent outputs.
- [ ] Image/video, repeated-media cache reuse, changed-media invalidation, and
  architecture companion-state restoration.
- [ ] Pinned Release app: GUI multi-turn, applicable settings, cache telemetry.
- [ ] Exact-head CI and dependent app proof before merging.

`GatherRowTileTests` includes 396 numerical cases and an opt-in synthetic
projection benchmark (`VMLX_GATHER_ROW_BENCH=1`). Benchmark numbers describe
one projection with synthetic weights, not an application speedup.

The concurrent RunBench diagnostic supports
`BENCH_BATCH_USE_GENERATION_CONFIG=1` and `BENCH_BATCH_SEED`. It records final
completion telemetry and rejects length-stopped or looping rows. Its raw
stream output is separate from parser/UI proof.

Subagent settings and residency orchestration are a separate follow-up. They
must prove same-model batching when enabled, default-ON smart swapping,
explicit-OFF coexistence, parent restoration, and cancellation through the
actual app controls before any orchestration fix is promoted.

## Runtime regressions found during validation

A real two-request Qwen3.5 batch exposed a pre-existing compiled MoE reshape
that retained the first batch width. Dynamic ExpandDims now preserves the
shapeless compiled path across batch contraction and expansion. The actual
compiled region passes widths 2, 1, 4, 3, 1, 2; real concurrent rows on three
architectures completed with natural stops and correct independent answers.

The stronger generated media-cache test also found Qwen3VLProcessor dropped
canonical history boundaries on image/video requests. Exact recurrent prompt
restores are deliberately excluded, so a direct cache probe could hit while
generation still prefills everything. The processor now carries proven
boundaries through placeholder expansion and rejects boundaries inside a media
span. Existing safe restore guards remain unchanged. Mapping regressions pass. Matched image and video runs now restore the safe
prefix during generation, reject changed media and complete parsed follow-ups.
Nemotron image requests had the same dropped-boundary gap and now publish the
canonical boundaries from their already-expanded prompt; audio is unchanged.

`BENCH_MEDIA_CACHE_PROOF=1` exercises cold, replay, changed media and a follow-up
with the actual parsed assistant answer. `BENCH_MEDIA_VIDEO_A` and
`BENCH_MEDIA_VIDEO_B` select two real video fixtures. It preserves bundle
sampling and template defaults, records full output, throughput, footprint and
restore progress, and rejects absent reuse or incomplete generation. Automated
structural checks still require manual semantic review of the full answers.

Known remaining follow-up: Nemotron video EVS resolves a pruned cache key but
still lacks a safe recurrent prefix checkpoint for generated replay. The
runtime correctly falls back to full prefill; video-cache reuse is not proven
for that path and is not a benefit claimed by this update. Preserve this as a
separate checkpoint-repair task. Gemma4 video is explicitly unsupported by its
processor. Exact timecode/frame-counter OCR accuracy is also not established.
