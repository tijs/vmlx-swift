# JANGH decode composition regression

Status: test source prepared; not compiled or executed. No loader is enabled and
no performance or model-family claim is made.

`JANGHDecodeCompositionTests` joins the actual primitives in this order:

1. Prepare the input using the gate/up input basis.
2. Run fused gate/up, raw projection clamps and SwiGLU, and optional output H32.
3. Pass those F32 route rows, already in the down input basis, directly to the
   weighted-down primitive, with no second rotation.
4. Cast only the accumulated token output to the requested output dtype.

The bounded matrix uses six distinct mixed gate/up/down bit triples spanning
2/3/4/6/8 bits. Input and down basis choices vary independently across cases;
all four combinations occur. Three input dtypes and three output dtypes yield
54 output comparisons, rather than a full combinatorial shader sweep. Shapes
are two tokens, eight routes, three experts, 96 input channels, 64 hidden
channels, and a nine-channel output tail. Routes include duplicates, changed
ordering, zero weights, negative weights and non-normalized positive weights.
Both clamped and unclamped activations are included.

The independent reference reads the actual packed words one bit at a time and
uses Float64 dot products, stable sigmoid, direct Sylvester H32 sums and router
reduction. It casts at the documented F32 hidden boundaries and starts from the
actual quantized input dtype. Assertions compare the intermediate hidden rows
as well as the final weighted outputs, so a wrong basis cannot be hidden by a
later reduction. This is a numerical reference, not expected bitwise equality
across differing reduction orders. Tolerances distinguish F32 computation from
final F16/BF16 output quantization.

Pending: Swift typecheck and actual Metal execution alongside both primitive
suites. Passing this test would establish only the exercised primitive
composition, not model loading, low physical memory, prefill, cache reuse,
multiturn coherence, or end-to-end speedup.
