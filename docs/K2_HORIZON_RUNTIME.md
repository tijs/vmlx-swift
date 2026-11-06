# K2-Horizon runtime contract

The text runtime recognizes `model_type: "k2_horizon"`. It implements grouped RMS normalization, full causal attention with GQA or MHA, full-head non-traditional RoPE, and a SiLU gated MLP. Checkpoint configuration determines dimensions, normalization groups, rotary base, and embedding/head tying.

## Weight layouts

| `mlp_layout` | Contract |
| --- | --- |
| `dense` (default) | Ordinary dense or affine-quantized MLP projections under `model.layers.N.mlp`. Per-projection affine metadata remains authoritative. |
| `switch1` | One-expert MLP under `model.layers.N.mlp.switch_mlp`, with every token assigned to expert zero. JANGH admission requires complete gate/up/down banks for every layer. |
| `dense_jangh_down` | Ordinary gate/up projections with a nonempty declared subset of JANGH down projections. Every remaining down projection requires an explicit affine entry. |

JANGH uses the validated version-2 `jangtq2` contract: LSB bitstream packing, FP16 row scales, odd-cubic codebooks, and declared `none` or `hadamard32` rotation. Supported code widths are 2, 3, 4, 6, and 8 bits; this is format admission, not a claim that every width has an actual-model qualification. Packed tensor shapes, codebooks, module paths, metadata ownership, and configuration/sidecar agreement are checked before mapping. Custom banks are excluded from generic affine loading and retain their mmap storage owners. Mixed dense banks cannot coexist with legacy runtime/stacked overlays.

Malformed or unsupported custom layouts fail rather than falling back to ordinary weights. Full `switch1` JANGH has source and bounded fixture coverage but has **not** been qualified with a complete actual-model artifact.

## Supported configuration and inputs

K2 is text-only. Image, audio, video, and native MTP requests are unsupported. MoE/MoVA variants, query/key normalization, attention gates or biases, partial rotary, non-default rotary scaling, and sliding attention are rejected. A non-null `sliding_window` is rejected even if `use_sliding_window` is false. Missing `hidden_act` defaults to the native `silu`; other declared activations are rejected.

Generation settings come from the bundle's generation configuration and explicit caller settings. The runtime does not introduce temperature, top-p, repetition penalties, forced reasoning tags, or output corrections to compensate for numerical or model behavior.

## Reasoning, history, and tools

The native IFM parser handles `<ifm|think>` reasoning, including the fast/faster tag aliases, separately from visible answer content. Tool parsing supports native XML, typed XML, and JSON bodies inside IFM tool-call envelopes. Literal argument strings are preserved; malformed or truncated groups do not produce partial executable calls.

An explicitly declared fixed reasoning mode is validated from `jang_config.json`, not inferred solely from the model family name. The declaration must disable the reasoning toggle, expose no selectable efforts, and identify one default mode whose template kwargs contain its supported `reasoning_effort`. An explicit different effort, malformed effort value, or `enable_thinking` value other than `true` raises `K2HorizonTemplateContract.ContractError.unsupportedReasoningMode`. Validation occurs before and after merging request and bundle context; incompatible requests are not silently rewritten. Without that metadata declaration, this fixed-mode validation does not impose a family-wide effort default.

Existing assistant thinking/reasoning fields are retained. When every supported thinking field is absent, an empty `reasoning_content` string satisfies the native history template. Callers must retain actual reasoning history when available; this adapter cannot reconstruct omitted reasoning. Template errors for invalid historical field types remain errors.

## Cache topology

K2 uses attention KV state per decoder layer; it has no recurrent SSM/PLE companion state. Its model cache factory selects `KVCacheSimple` by default and `RotatingKVCache` when an explicit maximum KV size is provided. Paged RAM caching is not enabled by the model. Effective cache topology and any KV quantization must be reported from runtime telemetry rather than inferred from weight format.

The normal coordinator supports prefix reuse and typed SSD persistence. Native chat-template boundaries and request identity govern safe reuse. Actual affine and mixed-layout runtime smoke exercised multi-turn continuation and fresh-coordinator SSD restores with paged RAM disabled and no rejected restores. Those rows do not establish every rotating-cache, quantized-KV, cancellation, or long-context configuration.

## Qualification limits

Actual affine and mixed-layout bundles have completed runtime multi-turn smoke; mixed-layout smoke also exercised tool exchanges. An isolated same-Metal-library comparison matched all captured full-model logits exactly against the Python reference in both cold and chunked execution. The reference and Swift share the same cold-versus-chunked rounding differences on that fixture. Different compiled backend libraries produce small numerical differences; this control does not imply bit identity across backend builds or every prompt.

There is no matched-weight speed comparison across the three formats, no low-RAM claim, and no actual full-JANGH runtime qualification. Dev-built Osaurus application/document workflows and the consuming engine pin require their own evidence. Runtime support and successful smoke tests alone do not establish production or release readiness.
