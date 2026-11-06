# Preserve activation precision during optional tied-head quantization

Optional load-time tied-head quantization converted an FP16 embedding to a quantized embedding, then the preserved-affine loader policy forced its output to BF16. In Gemma 4 QAT, FP16 router and layer scales mixed with these activations and promoted arithmetic to FP32. Quantizing storage therefore changed the model activation precision and reduced throughput.

The conversion now records the original floating embedding dtype. The later inferred alignment policy respects an explicit dtype while retaining the existing BF16 alignment for checkpoint-quantized embeddings without an explicit contract.

Live app follow-up testing exposed a second precision transition: unmarked FP16 full-attention disk entries were restored as BF16, then promoted to FP32 when the next FP16 rows were appended. Cold Gemma caches contained 60 FP16 tensors; after restore, the five global-attention layers contained 10 FP32 tensors while the sliding layers retained 50 FP16 tensors. This also rounded restored values unnecessarily.

Both Gemma model entrypoints now declare a native cache dtype identity. ModelContainer applies the existing storage-preservation mechanism and scopes the cache key to this versioned identity through both synchronous and asynchronous configuration. Older entries use a different namespace. This preserves whatever floating dtype the model actually produced; it does not force FP16 or alter attention accumulation precision. Legacy behavior for other models remains unchanged.

## Validation

- Release production and test targets built successfully.
- Two Metal regression tests passed: source FP16/BF16/FP32 lookup, layer-scale multiplication and tied projection retain their activation dtype with bounded embedding quantization error; checkpoint-quantized FP16 metadata still receives the existing BF16 alignment.
- On M5 Max, Gemma 4 26B-A4B QAT JANG_4M with a Q6 tied head, a 2,588-token prompt and bundle-default sampling measured 81.4,81.4,81.2 tok/s, versus 33.9,33.9,33.9 before the fix. One warmup preceded each three-run measurement. All measured answers were identical between the Q6 arms and stopped naturally. Reported physical footprint fell from 13,821 to 11,764 MiB. This is an engine-harness result, not a universal chip or application claim.
- Two concurrent requests returned the independent expected answers and stopped naturally with both slots overlapping. These short rows are correctness checks, not steady-state throughput measurements.
- Real image replay restored 297/298 prefix tokens; the actual-history follow-up restored 298/351. Changed media missed the cache and produced the changed image description. All four responses stopped naturally at approximately94–96 tok/s with coherent visible output.

## Follow-up cache validation

The original tied-head fix was exercised through a consuming Release app: the same 5,655-token prompt completed at 74.4 tok/s over 505 output tokens, versus an earlier Q6 app row at 33.4 tok/s. A grounded follow-up completed at 76.4 tok/s, and two same-model children overlapped and returned a correct parent continuation. These rows exposed the restore dtype issue above; they do not validate its subsequent fix.

Both new Metal tests passed: disk reopen/append covers both Gemma entrypoints, sliding-window wrap and FP16/BF16/FP32 storage; container configuration covers synchronous/asynchronous cache policy and namespace. The two original tied-head tests also passed on this source. A fresh consuming-app run remains pending.

## Outstanding app gate

Repin and rebuild after the restore-policy change, then prove Q6 with normal prefix/disk caching, reload/restore, visible multi-turn output, concurrent children, media and physical footprint through the actual GUI. The app gate remains open; no merge readiness or release claim.
