# Speculative decoding defaults

## Product contract

The default is bundle-aware. Qwen Flash-Next with a usable native MTP head starts On (Adaptive), across affine and JANGH quantizations. A bundle without a head remains autoregressive. Qwen 27B uses DFlash2 automatically only when a compatible drafter is present in its bundle. A display name, quantization label or folder name alone does not prove capability.

Settings expose Off, Default and On (Adaptive). The model picker exposes Off (AR) and On (Adaptive), resolving Default for the selected bundle before weights load. Adaptive chooses draft depth or verification width internally; fixed D1/D2/D3 buttons are not product controls. Explicit Off disables speculation, including a bundled or selected external drafter. Preserve saved explicit choices; an old Off without trustworthy provenance cannot safely be reinterpreted as an untouched default.

Bundle capability discovery must inspect configuration and actual weight inventory. Native heads can live in main shards or sidecars and use different tensor layouts. A bundled DFlash2 directory must contain usable weights and compatible target metadata; a config-only directory does not qualify. Missing or incompatible optional draft support must not prevent the target from running AR.

## Request and cache contract

Chat, API, included evaluations and benchmarks use the same bundle-aware resolution unless the caller explicitly requests an override. Benchmark reports must record the effective strategy, rather than infer activation from an On setting. Keep model generation parameters and explicit sampling overrides unchanged.

Media requests (images, video, and every later request whose context still contains media) are prefilled through the target's VLM path and then speculate on lanes that support it; other targets use their normal AR media path. A text-only subsequent request must not inherit incompatible speculative companion or position state. Schema-constrained requests remain AR until speculative grammar rollback is qualified.

SSD remains the prefix-cache tier. Preserve full precision, logical prompt identity, media salts, target cache boundaries and architecture-specific companion state. Store each tool-call prefix before its continuation. A target KV hit alone does not prove a usable MTP or DFlash2 restore. Native head priming and drafter context must be aligned to the restored target boundary; otherwise perform the correct rederivation or cold path.

No runtime files may be written into model bundles. Sparse PLE n-gram tables remain file-backed; warming must respect memory availability. Do not quantize cache or alter the sampler to make a performance result look better.

## Qualification

This document defines the intended contract, not a blanket claim of completed verification. The integration review must cover absent/malformed heads, mixed JANGH layouts, explicit Off on a loaded model, preference persistence, included eval/bench defaults, schema/media fallback, multi-turn tool and disk-restored continuations, cancellation and model switching. Record output correctness, actual token/s, context, cache topology, footprint and benchmark clocks. Historical D1/D2/D3 and default-Off documents retain their original measurements but no longer define product defaults.
