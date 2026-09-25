# Bonsai2 Hadamard / packed ternary runtime checkpoint

Status: **PARTIAL — focused runtime tests executed; full-model/app proof missing.**
This is a private, local implementation checkpoint, not permission to publish
the model repositories, merge a runtime PR, or advertise working model support.

## Current combined checkpoint — 20:57 PDT

SOURCE EVIDENCE: `4c6bec46f316f7c4c63ea1c4c580ad968bb37a52`, based on current
mainbfb34ff1 (remote checked20:53). `git diff --check` returned0.
LIVE EVIDENCE: `unit-7e1a13f4/SWIFTTEST_bonsai2_4c6bec46_combined__205407.log`
and `results-4c6bec46-combined-swift-testing.xml`: **146 executed tests, zero
failures, one explicitly skipped legacy Flash-Next installed-bundle test**
(147 discovered in19suites). The skipped row is not Bonsai coverage. Guard
exit0/zero survivors20:54:24, peak sampled tracked footprint0.38GiB, unchanged
swap1.81GiB. This incremental warm run took14.834s; it is not a speed benchmark.
The earlier protocol-only build/run at the same source recorded109 executed,
one skipped, zero failures; peak2.82GiB. Raw build warnings remain in its log.

Both actual local Bonsai bundle tokenizers and media processors executed:
native-default/Off/low/medium/xhigh prompt token counts528/492/516/490/528,
canonical no-generation-prompt equality, full JSON schemas and native errors.
Two-image inputs produced638 prompt tokens,128 image slots, two frames and
pixels[512,1536], preserving tool/reasoning history and media/effort isolation.
Media `cachePrefixTokenCounts=[]` remains an unclosed performance item, not a
cache-hit claim. No full weights opened and no VLM forward occurred.

Fragmented tool tests retain exact strings/types, both calls, literal think
tags and progress before EOS. Incomplete payloads do not invent a call; normal
reasoning resumes after a closer. Existing Gemma schema, MiniCPM CDATA and
reasoning-family tests also execute unchanged. Core Hadamard/packed/load and
tiny hybrid disk-reopen/rotating state-logit parity re-ran at this same source.

Open gates: app-owned bridge tests and dev build/UI, both full27B models,
natural-stop multi-turn/tool/media output, actual token/s and footprint,
full-model accepted restore and tool publication ordering, CI and PR merge.
The app preparation branch is based on current main0901780c in its own worktree;
no Gemma source is reworked here. See the active private STATUS.md for ownership.

## Retained protocol failure checkpoint — 20:44 PDT

SOURCE EVIDENCE: `382d07ff`, `Tests/MLXLMTests/Bonsai2ProtocolTests.swift`.
LIVE EVIDENCE: `unit-7e1a13f4/SWIFTTEST_bonsai2_protocol_typed__203821.log`
in the retained evidence directory below: four test functions,19 assertion
issues, guard exit1 with zero survivors. No full model weights were opened.

Two-call ordering/types and serialization passed across fragmented streams.
Actual tokenizers rendered the native default and all four modes, but exposed
schema-field loss and swallowed invalid-effort errors. Real two-image
preprocessing produced the expected pixels/token slots and isolated cache
salts, but lost structured assistant tool-call history. Literal think tags
inside tool argument values were also stripped. Two other assertions were
fixture errors: the bundle explicitly sets repetition_penalty=1.0.

The source-bound plan preserves previous fixes: retain the Gemma schema adapter
from27f5806e/3d06edee, route configured Qwen XML templates with raw schemas and
their own errors, compose Qwen3VL media content over canonical message metadata,
and protect committed Qwen tool payloads from reasoning-marker parsing. Test
both generation-prompt modes, invalid input, fragments, incomplete envelopes,
ordinary reasoning and prior MiniCPM behavior before claiming correction.

## Retained core execution checkpoint — 20:36 PDT

SOURCE EVIDENCE: `fc5fc19c2c6e23b82a24b93d7f646264a5df4bca`, runtime files and
four new contract/runtime/routing/cache test files described below. Production
code is unchanged from the initial implementation; later fixes corrected new
fixture assertion macros and initialized-parameter assignment to use MLXNN's
public update API.

LIVE EVIDENCE: retained command/results under
`/Users/eric/vmlx-private-evidence/bonsai2-swift-2026-09-17/unit-7e1a13f4/`:
`SWIFTTEST_bonsai2_fc5fc19c_focused__203236.log` and
`results-fc5fc19c-swift-testing.xml`: **37 tests, seven suites, zero failures or
skips**, 99.75s. The guard exited0 at20:34:31 and recorded zero owned survivors
at20:34:32; peak tracked physical footprint1.16GiB, swap1.81GiB unchanged.
Source-built metallib SHA256 `9ce1a2ab8d82e73152c2570176a1bdd45773dc7e33d37b8cd3dab18fa0699e6d`.
The command used exact24dependency pins, local-only dependency resolution and
serial test execution. Earlier failed attempts remain in `RUN.md`.

This covers independent packed/native expansion and tiny loader logits,
Hadamard normalization/sign order, wrapper/raw-fusion isolation, retained
ordinary-affine behavior, and cold versus disk-reopened hybrid continuation
state/logits with simple and rotating attention. It does **not** establish
complete27B model output, speed, app controls or image-forward correctness.

At that earlier checkpoint `Bonsai2ProtocolTests.swift` was authored for the actual installed bundle
tokenizers, native reasoning/schema/history semantics, two-image processor
payloads, and fragmented tool-stream fidelity. Its subsequent failures are
recorded above. The local-bundle rows are opt-in using
`BONSAI2_PROTOCOL_BUNDLE_ROOT`, open no weights, and must not be counted when
skipped. Full real-model acceptance remains listed at the end of this document.

## Isolation and source bindings

- Worktree: `/Users/eric/vmlx-bonsai2-runtime`, branch `feat/bonsai2-runtime`.
- Base: `osaurus-ai/vmlx-swift` main, `bfb34ff142817f3a35cf6502ad5d8dd742c4e87f`.
- Authoritative handoff:
  `/Users/eric/jang/docs/runtime/bonsai2-27b-2026-09-17/01-RUNTIME-HANDOFF.md`,
  SHA-256 `21dc937d27be84153dc982ba9757e5d686234cb753e806019682c1fb589b602f`.
- Python source read before editing:
  `/Users/eric/mlx/vllm-mlx/vmlx_engine/utils/jang_hadamard.py`,
  SHA-256 `94255da326cde1511743fe561f53bbbcad4322d5d57158f2e642b5db68d08a7d`;
  `jang_ternary_packed.py`, SHA-256
  `d3806c1af5e9f5918504983a0b8b7f5def0899f4dc0f55b9134e84c3ad34da78`;
  `jang_loader.py`, SHA-256
  `350af1678dd3b0fb297b769c66f1876bd163032df21b4b0ea78c2b3ad2b0d189`.
- Converter reference:
  `/Users/eric/jang/jang-tools/jang_tools/ternary_packed.py` and
  `convert_bonsai2_jang_affine.py`; no converter edits here.

Prior fixes inspected and retained:

| Commit | Owning behavior | This patch |
| --- | --- | --- |
| `948f03aec89501104a7ab93360508c7b83a1c3a8` | Native affine-1 storage and exact schema-2 manifests | Does not expand or replace affine-1 |
| `c0e869c8a3f0af5fe5db55dadd9fd9206f1e15c8` | Qwen raw-array GDN input/tail fusions | Declines only activation-rotated wrappers |
| `bf8b31995` | Later Qwen fusion policy | Retained |
| `5332a2a2b`, `4f9e2d176`, `f2b184841` | Post-load bf16 materialization policy and telemetry | Only validated Hadamard bundles preserve their stored F16/F32 contract |
| `2422cfb8e`, `7d949c263` | Affine mmap scale preservation and embedding output dtype | Ordinary bundle policy unchanged |
| `0aa728af5deab52141506828ba36a91d5fe2fd51` | Owned uncached resident reads | Unchanged |

## Bounded implementation

`Libraries/MLXLMCommon/JangHadamard.swift` validates both config owners, Prism's
sidecar, strict Boolean runtime markers, complete schema-2 manifest coverage,
non-overlapping directions, supported transform/block, exact F32 +/-1 signs,
source architecture dimensions and F16 quantization metadata. Unknown routes,
tied heads, MoE and MTP configurations refuse this port rather than silently
using plain affine weights. Wrappers reuse the existing quantized arrays;
forward is `H(signs*x)`, inverse embedding lookup is `signs*H(rows)`, with
normalized F32 computation and cast-back to the activation dtype.

`JangTernaryPacked.swift` expands UInt8 26-byte/128-trit groups into native
UInt32 affine-2 words, preserving scales and creating biases as `-scales`.
Head bytes above 242, tail bytes above 26, bad shape/dtype, missing modules and
pre-existing biases throw. Expansion runs before manifest shape inference and
sanitize in `Load.swift`. Coverage must equal the declared Hadamard set.

The real `lm_head` is 248320 x 5120: materializing all 1.27 billion trits as
UInt32 intermediates would be multi-GiB. The implementation evaluates row
chunks targeting 1,048,576 codes, minimum one full row, before final output
concatenation. This bounds each code intermediate by rows, not total process
footprint. Final output buffers, concatenation and allocator retention remain
measurement items. **No new global allocator/cache-limit mutation is made.**

Two additional source integration defects would otherwise invalidate a
wrapper-only port:

1. Qwen VLM GDN's uniform/grouped input projections and compiled output tail
   read `QuantizedLinear.weight/scales/biases` directly. Because the new wrapper
   is a subclass, an unguarded cast succeeds while bypassing its transform.
   The text GDN grouped-input path has the same issue. Targeted eligibility
   guards retain plain affine eligibility; whole-model compilation is not
   disabled.
2. Ordinary post-load bf16 conversion would round the bundle's F16 scales and
   F32 norms/state projections. Only a successfully parsed and validated
   Hadamard contract bypasses that conversion. Existing norm sanitize `+1`,
   parser, sampler, EOS, cache topology and media code are not replaced.

## Current source / template route matrix

Both private bundles have byte-identical tokenizer config, generation config
and Hadamard sidecar (hashes below). This is artifact evidence, not a rendered
template or model-output pass.

| Surface | Current source binding | Required live proof |
| --- | --- | --- |
| Storage A / B | Common `loadWeights` from `LLMModelFactory.swift:2032` and `VLMModelFactory.swift:754`; wrappers installed before parameter update | Both complete loads and identical parity logits |
| Text route | Wrapped `Qwen35Model`; standalone `Qwen35TextModel` intentionally refuses the full bundle's paths | Text multi-turn and cache restart |
| VLM route | `qwen3_5` factory entry; `Qwen3VLProcessor` registry; `Qwen35.prepare` image/video feature merge | Real image, OCR, video and same-media cache reuse |
| Reasoning | Actual `tokenizer_config.json` template reads `enable_thinking`; accepts exactly `low`, `medium`, `xhigh`, default `xhigh`; `preserve_thinking` defaults true | Off/low/medium/xhigh render context, visible answer/reasoning and no marker leakage |
| Tools | Template renders the actual `tools` array as JSON schemas; native XML `<function>` calls. Bundle declares `qwen3_coder` / `xml_function` | Real schemas, parsed arguments, tool round trip and multiple calls |
| Processor preservation | `Qwen3VL.swift:131` forwards tools and additionalContext; text return at 150 and media return at 233 retain `toolSchemas` and reasoning cache salt | End-to-end tool/media payload, not a keyword assertion |
| Cache ordering | `Evaluate.swift:4797` stores after decode and before stream finish; `CacheCoordinator.swift:915` persists typed KV and recurrent state; offset/key mismatches refuse storage | Disk write completion, subsequent hit, exact state/topology and prefill timings |

App source inspected read-only at
`/Users/eric/osaurus-native-tool-batches`, HEAD
`6267660b2811b7e8bdc13573ef88039d3501e61e` (the parent owns this worktree):

- `Packages/OsaurusCore/Services/DeclaredReasoningEffort.swift:178` resolves
  the JANG declaration before template fallback; `:331` prepends `none` to
  declared effort levels. `ModelOptions.swift:310` uses this capability for
  picker options, and preserves omission as the native default.
- `Services/ModelRuntime/MLXBatchAdapter.swift:1119` uses the declaration for
  effort transport; `:1138` only forwards explicit `preserveThinking`;
  `:1233` maps Off to `enable_thinking=false`. The template itself closes the
  think block; this port adds no forced markers or sampler overrides.
- The actual UI labels are None / Light / Medium / Extra High
  (`Models/Configuration/ModelOptions.swift:95`), with wire values
  `none / low / medium / xhigh`. The user's “med” means the native `medium`,
  not a new template value.
- `MLXBatchAdapter.swift:1675` drains the producer/cache store before yielding
  terminal info and releasing the solo lease. A tool event alone does not
  terminate the stream.

Important cache limit: a tool turn deliberately does **not** persist a
post-generated tool-call boundary (`includeGeneratedBoundary` excludes tool
calls). It persists eligible canonical prompt boundaries. A disabled disk
cache, memory-store budget refusal, unsafe offset, or quota eviction can
legitimately produce no durable entry. Do not claim “every tool always writes
to disk,” or use a cache-write screenshot as proof of a valid subsequent hit.
No cache/parser rewrite is justified by the new storage format alone.

## Tool/cache ordering trace and executable proof plan

Source-only trace, after local implementation commit
`a942bcb87028729e2a39a3d8ff9cc3ecbcbbf397`; no cache code changed.
The app remains at the HEAD above. The inspected working-file SHA-256 values
are `49c8ed60e0cdae5937745840fc1cb472b06939fe3ce82a85a18ae1c356bb9799`
for `MLXBatchAdapter.swift` and
`75d0707daa144be6cb3f9601e6e4a2c9eb77aea1ccb526b3020d79415c46a506`
for `ModelRuntime.swift`. These bind the trace even if the parent later edits
the app worktree.

| Stage | Source binding | Meaning and limit |
| --- | --- | --- |
| Tool parsed / preview | App `ModelRuntime.swift:5866` appends each invocation; preview is immediate | Not execution and not a persisted boundary |
| Core completion info | `Evaluate.swift:4785`; batched `BatchEngine.swift:3299` | CPU stats precede the synchronous store; direct engine consumers must wait for EOF before reusing the model |
| Eligible canonical store | `Evaluate.swift:2886`, `:2995`, `:3105`, `:3132`; batched `BatchEngine.swift:3442` | Captured canonical and safe stable prefixes, not a generated tool-call checkpoint |
| Typed linked persistence | `CacheCoordinator.swift:915`, `:1095` | Refuses offset/key mismatch; stores typed KV plus the required recurrent representation, then enforces combined quota |
| File/index operation | `DiskCache.swift:381`, `:498`, `:530` | Synchronous under the process-wide IO lock; complete temporary file is renamed before index insertion; errors are best-effort logged, not thrown to the model request |
| Engine EOF | `Evaluate.swift:4823`; `BatchEngine.swift:3772` | Follows store return and GPU drain, not proof that storage succeeded or survived quota |
| App terminal info | App `MLXBatchAdapter.swift:1676`, `:1741` | Holds engine info until upstream EOF and allocator teardown |
| Native batch publication | App `ModelRuntime.swift:5785`, `:5828` | Publishes the full ordered tool batch at held completion info; complete-response mode waits for its EOF |
| Next solo prefill | App `MLXBatchAdapter.swift:1375`, `:1747` | Next request acquires the process-wide solo lease after prior stream completion; input preparation follows acquisition |

For native B=1 the source ordering to exercise is:

```text
tool previews / core info
  -> eligible store returns + quota pass + GPU drain
  -> engine EOF -> app allocator teardown -> held completion info
  -> executable tool batch -> tool result -> next request's prefill
```

The next request must also acquire the prior solo lease. This is not a promise
that its acquisition comes after the batch-publication log: the lease and
consumer tasks can interleave after held completion info; both are already
after engine/store drain. In B>1, `BatchEngine` is an actor and `finishSlot`
is synchronous through storage/EOF; the adapter still holds completion info.
Concurrent independent slots are intentionally not the same invariant as a
single agent's sequential tool continuation.

The literal requirement “a new complete generated-tool checkpoint is written
after every tool call” is **not provided by this source**. Tool turns explicitly
exclude the generated boundary at `Evaluate.swift:4799`, and Qwen's canonical
hybrid boundary excludes full prompt/post-answer duplicates at `:2909` and
`:3220`. This preserves existing correctness guards; making generated
tool-call states reusable would be a separate cache-boundary parity change,
not something to infer from adding a weight storage format.

Required-tool and warmup exclusions apply equally to both Bonsai bundles:

- App `MLXBatchAdapter.swift:2238` sets `.freshRequiredToolSelection` for
  required/named tool choice. `Evaluate.swift:1765` preserves it in iterator
  restore/store policy. `:1891` skips disk-backed restore and logs that warm
  restore is not proven safe for this topology. Batched entry applies the
  same guard (`BatchEngine.swift:2028`). A required-tool success is therefore
  not a disk-hit row. This guard is not removed by the Bonsai implementation.
- Qwen constructs Mamba state for linear-attention layers
  (`MLXVLM/Models/Qwen35.swift:3169`; text `Qwen35.swift:1430`). With the actual
  64-layer / interval-4 configs, that is 48 Mamba and 16 attention caches at
  construction. `maxKVSize` chooses rotating instead of simple attention;
  later KV quantization/promotion must be reported from live effective state.
  `CacheHelpers.swift:188` and `:210` require typed disk restore for Mamba.
- Exact recurrent reusable-prefix warmup persistence is gated by
  `CacheHelpers.swift:257` and `Evaluate.swift:3034`. Processor-proven stable
  prefixes retain their safe N-1 path. Do not relax the guard to get a hit.
- For disk-only Qwen Mamba, the recurrent state is in the typed safetensors
  payload (`TQDiskSerializer.swift:285`, `:476`), not obligatorily in a second
  SSM sidecar. `CacheCoordinator.swift:1050` writes separate recurrent state
  only when that topology or a published paged payload requires it.

The concrete reproducible invariant/test matrix is:

1. Run the existing delayed-store test
   `infoArrivesWhileDelayedCacheStoreStillRunning` (included in `Package.swift`
   at 872), plus the app's `LocalToolBatchBridgeTests` and solo-gate tests.
   Compose the actual adapter/bridge in a bounded app fixture: release two
   parsed calls, block the engine store, assert previews but zero executable
   calls and no next-prefill acquisition, release the store, then assert both
   ordered calls and continuation. Existing tests cover these pieces, not a
   whole Bonsai adapter/model run.
2. For each storage format, use the same actual small Qwen hybrid graph with
   at least one GDN and one attention layer, loaded through the corresponding
   contract. Capture a processor-declared canonical boundary, store into an
   isolated temporary disk root, destroy the coordinator, reopen it, restore,
   and continue with identical token IDs. Assert all cache layer counts,
   types, offsets, state-array counts/contents and continuation logits against
   the cold reference. Repeat attention with an explicit rotating bound.
   Build on `HybridStripBoundaryPrefillTests` and `TQDiskSerializerTests`; do
   not count just a no-exception round trip as parity. This engine-level row
   passed in `Qwen35HadamardCacheTests` at the current execution checkpoint.
   The app adapter/bridge composition in row 1
   remains a proposed follow-up, not implemented by this port.
3. On each real bundle, normal tool-choice/auto rows must capture the actual
   rendered prompt/schema/media/cache salt, canonical key and offset,
   complete file plus index after the quota pass, `TOOL-BATCH published`, and
   a subsequent **accepted** disk restore before measuring reduced prefill.
   Required-tool rows instead assert the explicit fresh-selection skip and
   complete coherent tool execution. Reasoning-mode or media changes require
   their own key identity; no cross-mode/media hit is assumed.
4. Include disabled cache, insufficient store budget, failing storage and
   quota-pressure rows as graceful no-durability cases, not positive cache
   rows. `DiskCache.stores` increments before save at 442; it is not a success
   counter. A validated existing entry may skip rewriting at 455, and a quota
   pass can remove a new entry. `hasDurableDiskEntry` checks native recurrent
   geometry (`CacheCoordinator.swift:732`), but accepted restore plus state
   parity is still the stronger proof. No power-loss/fsync durability promise
   is made by this checkpoint.

No new timing delay, cache-policy override, forced tool directive, reasoning
coercion or allocator limit is proposed to make these rows pass.

## Metadata/header checks actually run, 2026-09-17

Read-only Node check: parse config/JANG/Prism/generation/index JSON; for each
indexed safetensor shard read only its 8-byte header length and JSON header;
validate every declared rotated weight/scale/sign/bias dtype and shape. No
weight data, MLX evaluation, tokenizer execution or generation was performed.

| Local bundle under `/Users/eric/models/OsaurusAI/` | Rotated layouts checked | Shard-header bytes read | `lm_head.weight` |
| --- | --- | --- | --- |
| `Bonsai-2-27B-Ternary-JANG` | 402 (401 forward, 1 inverse) | 320128 | U32 `[248320,320]` |
| `Bonsai-2-27B-1.75bit-JANG` | 402 (401 forward, 1 inverse) | 263696 | U8 `[248320,1040]` |

Each manifest declares and contains 485 modules (402 language, 83 vision).
Sign widths are 5120/6144/17408; final text norm is F32 `[5120]`.
Config EOS is 248046; generation EOS is `[248046,248044]`.

Metadata SHA-256 receipts:

| File | Ternary | Packed |
| --- | --- | --- |
| `config.json` | `f57b9c8cfc0d9d4edf35f65b75d230c1ee9c85467f40dda61a3bf8d07b3ee082` | `500b966bb0a564d601690d7673388268923f6d66a1690d4d4b4d07ef73cde16e` |
| `jang_config.json` | `387035edc801ffb7dc59c9a7dc41a00a8bca7408187d41e93f03bbe0dff021a4` | `f8104703441fb2ebc94889e76cc306771d91f919c534a18b95f724543fb63442` |
| `hadamard.json` | `7132a3ec364f0bdac1f08f905f24f0ad2f14245060f592637a0396826d3b5fe6` | same |
| `generation_config.json` | `875ee16774666031c8cff7a0d19b02ee2264c71229f0664079200c469384f5c5` | same |
| `tokenizer_config.json` | `60e96a382893c9efeb7116fb83e2c532cb149d00bfd859713640c0a8ae282019` | same |
| `model.safetensors.index.json` | `26e0c700f49648c6199011443fa96069118285e7bf3e94d2c5c7d7ad10071b88` | `7068f83e14656b9837df72a7bab31095a0f299fbb571457307d246a1c3d5aa75` |
| Combined indexed header names + raw header bytes, filename-sorted | `b61003b40bf9776ba6298b5aa91dd03b0fe85f7f405f785dce366ea3dac51783` | `a8a3982f2aaff65a067f887ca6de9103ee9b1ccbd2229dc186c78fb6c694e79f` |

## Tests and remaining proof

Executed at `fc5fc19c` in the focused run above:

- `JangHadamardContractTests`: native/packed declarations, fallback owners,
  strict flags, malformed sidecar/manifest/coverage, checked width arithmetic,
  ordinary bundle nil behavior and native reasoning vocabulary/transport.
- `JangHadamardRuntimeTests`: independent scalar trit encoding, exact UInt32
  round trips at representative widths, malformed packed bytes/layouts,
  independent F32 butterfly reference, wrapper array reuse, actual generic
  loader dtype preservation and ordinary affine bf16 regression.
- `Qwen35HadamardRoutingTests`: supported/refused routes, actual text/VLM
  sanitize +1 vs already-shifted norms, F32/F16 passthrough, raw-fusion bypass
  prevention, cache offset/count/content equality.
- `Qwen35HadamardCacheTests`: both native affine-2 and packed ternary fixtures
  load through the production loader into the same deterministic four-layer
  Qwen graph (three GDN + one attention). A 17-token canonical checkpoint is
  stored, the coordinator is discarded and reopened, and all cache counts,
  types, offsets, metadata, state-array counts, dtypes and contents are compared
  with the cold reference. Continuation logits and resulting state are compared
  exactly, including equality across the two weight representations. The
  attention row repeats with a rotating window of eight to exercise wraparound.
  The public `TokenIterator` must report an accepted 17-token disk restore for
  ordinary continuation, but must prefill fresh for `.freshRequiredToolSelection`
  without incrementing disk-hit telemetry. Changed first-token, reasoning and
  media salts must miss. The media tensors are hashing fixtures only, **not** a
  VLM-forward or image-cache proof, and token IDs represent an already-rendered
  canonical boundary, **not** tokenizer/template proof. The parsed-tool-style
  store excludes generated boundaries; no production cache policy is changed.

Registration was inspected in the actual package manifest: `MLXLMTests` at
`Package.swift:810–835` auto-discovers its directory and has no explicit source
whitelist, so all four new files are included. The separate
`MLXLMCommonFocusedTests` target does have a whitelist: existing affine-1,
Qwen-fusion, norm-convention and delayed-cache-store baselines already appear
there. No manifest edit is needed or made.

Checks actually executed:

```sh
git diff --check
xcrun swift-format lint --strict Libraries/MLXLMCommon/JangHadamard.swift Libraries/MLXLMCommon/JangTernaryPacked.swift Tests/MLXLMTests/JangHadamardContractTests.swift Tests/MLXLMTests/JangHadamardRuntimeTests.swift Tests/MLXLMTests/Qwen35HadamardRoutingTests.swift
```

Both returned exit 0 with no diagnostics. These are not compilation or runtime
tests. Parent's Gemma build/live proof owns the resource slot; no competing
build, Metal test or Bonsai model was launched.

For the later cache regression, the separate command
`xcrun swift-format lint --strict Tests/MLXLMTests/Qwen35HadamardCacheTests.swift`
and `git diff --check` also returned exit 0. Formatting is not executable
verification of the new assertions.

### Original private test-build preparation (historical plan)

The plan below was subsequently executed; current results supersede its
future-tense status. Exact command and every environment/test correction are
retained in the run directory above.

The full-Xcode path exists, while the machine's global developer selection is
Command Line Tools. Use a per-command `DEVELOPER_DIR`; do not change the global
selection. Both `.build-bonsai2` and this worktree's `Package.resolved` were
absent when planning. The existing root checkout's ignored `Package.resolved`
has SHA-256 `3af2208ba61fb3a04543a2c5d8d74fdb4af50505a124e3c5f6c8d70eccc6bf8d`.
A read-only `git cat-file -e <revision>^{commit}` check found all 24 pinned
revisions already present in local SwiftPM bare caches. The conditional docc
dependency is not requested by this environment. No resolve/fetch/build was
run to make those observations.

After the serialized slot is granted, seed only an isolated lockfile and
private dependency cache from those exact local objects; preserve the ordinary
package graph and force its resolved versions. Use private cache/config/security
paths and local manifest caching, disable prefetching and experimental prebuilt
downloads, and skip remote updates. Do not reuse or edit app SourcePackages or
DerivedData. SwiftPM's test filter restricts **execution**, not necessarily all
compilation: a clean first build can still be substantial. Run under a bounded
owned-process build window with a retained log; if the window expires, report
the compile checkpoint and resume only after slot coordination. Never represent
zero matching tests or a compile-only row as a passing regression run.

Planned command, only after an explicit serialized slot with full Xcode and
an approved isolated scratch/build path (not shared SourcePackages/DerivedData):

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test \
  --scratch-path /Users/eric/vmlx-bonsai2-runtime/.build-bonsai2 \
  --cache-path /Users/eric/vmlx-bonsai2-runtime/.build-bonsai2/cache \
  --config-path /Users/eric/vmlx-bonsai2-runtime/.build-bonsai2/config \
  --security-path /Users/eric/vmlx-bonsai2-runtime/.build-bonsai2/security \
  --manifest-cache local --force-resolved-versions --skip-update \
  --disable-prefetching --disable-experimental-prebuilts \
  -j 2 --no-parallel \
  --filter 'JangHadamardContractTests|JangHadamardRuntimeTests|Qwen35HadamardRoutingTests|Qwen35HadamardCacheTests|JangAffine1RuntimeContractTests|Qwen35FusedInputProjectionTests|NormConventionResolverTests|EarlyCompletionBeforeCachePersistTests'
```

Missing acceptance evidence: actual-tokenizer/protocol tests; exact
four-prompt Python parity and complete native-vs-packed logits; both real bundles'
coherent natural-stop multi-turn output with tok/s, TTFT/prefill and physical
footprint; native reasoning controls; real tool schema/round-trip/batch calls;
per-tool canonical disk persistence plus next-turn restore and SSM/KV state;
image/OCR/video and cache-salt fidelity; isolated app visuals and normal
unload/cancellation. No throughput number is claimed for this Swift port.

After local proof and parent review, the authorized PR/CI/merge workflow must
carry these exact receipts in GitHub comments. No GitHub comment, push, merge,
model publication, release or app installation has occurred in this lane.
