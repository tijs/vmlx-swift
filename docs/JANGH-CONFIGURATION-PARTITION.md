# JANGH configuration admission and partition

`JANGHConfigurationPartition` reconciles root `quantization` and
`quantization_config` aliases after flattening `per_tensor` maps. Contradictory
owners and duplicate JSON keys are rejected. Conflicting ordinary/custom
module spellings under the generic model/language_model wrapper aliases are
also rejected, preventing an ordinary exact match from outranking a custom skip. Config and optional sidecar
`jangtq` headers must agree; the sidecar additionally requires format
`jangtq2` and format version 2. A legacy format label alone cannot authorize
odd-cubic JANGH decoding. The existing strict format contract checks codebooks,
packing, rotations and complete gate/up/down triples.

The output retains two distinct views. The custom view and validated contract
own the custom projections. The ordinary view replaces each custom override
with explicit `false` under both quantization aliases, so generic quantization
cannot apply its global affine default to those modules. Ordinary affine/MX
and unknown modes are preserved. The strict `BaseConfiguration` decoder must
still validate that ordinary view; this partition does not accept an unknown
mode on its behalf. Using only the ordinary view to construct a model is not a
supported loader path: all custom modules must be constructed without dense
placeholders and their packed banks must satisfy the index and mapping gates.

This first adapter admits custom banks only under the audited canonical
`model.layers.N.mlp.switch_mlp.{gate,up,down}_proj` namespace. Other complete
triples (including shared experts) are explicitly unsupported, preventing
generic shared-expert/attention aliases from broadening bank admission.

This first adapter supports one root metadata owner. Nested text-config
quantization owners and sidecar quantization plans are explicitly unsupported,
not silently preferred or rewritten. `model_type` is retained unchanged;
configuration admission does not register GLM or Naive model support.

Six XCTest methods are prepared for alias equivalence, flat/nested plans,
sidecar-only headers, contradictory owners, custom exclusions, unknown mode
preservation, legacy labels and duplicate keys. These new tests and source have
not been compiled or executed. Previously executed header/index tests do not
prove this new partition. Factory loading, architecture modules, full-model
memory, coherent multi-turn execution, prefill and performance remain gated.
