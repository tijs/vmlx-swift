# Naive asymmetric decode attention

The candidate routes Q/K head width192 and V width128 through SDPA only when
query length is at most8, query length times the GQA ratio is at most32, and
queries do not outnumber keys. These match the pinned Metal vector kernel's
admission conditions. Larger queries retain the explicit reference implementation.

Value scaling remains in the activation dtype before attention. Sinks are cast
to the query dtype, matching the reference and the core's promotion contract.
Entirely masked query rows explicitly produce zero, including on the generic
SDPA fallback where a bool mask otherwise uses a finite minimum.

Three prepared tests cover the cutoff on both sides, analytical uniform scores
with sinks and fully padded rows, three activation dtypes, strided storage,
batch2, and offset/window masks against the retained reference. Accuracy budgets
are fixed in source before execution. Source authoring and diff checks alone do
not establish numerical correctness or speed; GPU, full-model and cache proofs
remain required before promotion.
