# Packed JANGH prefill kernels

Adds sorted packed codebook matrix multiplication, fused gate/up activation and
optional H32 output rotation. NAX handles FP16/BF16; the steel path works without
NAX and handles FP32 without relaxed operand precision. No complete model factory
or performance claim follows from these primitives.

The tile loader reads row scales only for valid output rows. Invalid expert IDs
produce NaNs without reading outside mapped banks. Small buffer inputs are padded
for MLX custom-kernel device-pointer rules. Packed banks must already be available
and row contiguous; the entrypoint does not dequantize or copy full expert banks.

Native MLX headers are generated from the checked-out Cmlx revision by
`tools/jangh/generate-prefill-headers.py`, with per-source SHA256 provenance. The
JANG tile/activation code preserves the reference backend rounding boundaries:
steel casts projection outputs before activation, while NAX applies activation
to its F32 accumulators before the final cast. Cross-backend bit identity is not
claimed.

## Bounded proof

Six XCTest methods passed on an M5 Max using these exact candidate primitive
sources and actual production MLX objects. Coverage includes 2/3/4-bit mixed
projections, FP16/BF16/FP32, input/output rotations, token counts1/3/7/8/16/33/65,
ragged columns, tiny device buffers and invalid routes. Independent scalar
bit-unpacking and backend-specific rounding form the numerical reference.
Additional coverage executes36 standalone H32 input/output dtype/shape combinations and96 mapped routed calls spanning56/64/72 route assignments, duplicates, unequal weights and B2→B1 shapes. H32 input and steel output rotation now fuse the F32 butterfly and output cast into one launch.
Both native NAX and forced steel execute; FP32 requests explicitly resolve to
steel. Full-package integration and full-model cache/tool/performance proof remain.

Failed experiments are retained: fused FP32 NAX required34KiB threadgroup memory
against a32KiB limit; split FP32 NAX differed from full-precision reference due to
relaxed operands; blindly switching the native NAX descriptor to strict precision
broke the fragment-layout contract. None of those candidates is promoted, and no
numerical threshold was relaxed. Default FP16/BF16 fusion remains enabled in this
primitive; actual model activation dtype must be verified at integration.

Next gates: strict loader admission, coherent multi-turn real-model execution, physical
footprint, and matched prefill/decode measurements against same-size affine.
