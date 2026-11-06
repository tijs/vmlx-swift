# JANGH decode word unpack

The fused gate/up and weighted-down kernels now generate a lane-local packed-word unpack at construction time. Each lane consumes16 coefficients from one32-bit word (2-bit), two32-bit words with a0/16-bit starting offset (3-bit), two32-bit words (4-bit), three32-bit words (6-bit), or four32-bit words (8-bit). The shader explicitly reuses these loads across16 coefficient FMAs.

K/H are already required to be multiples of32, so a16-coefficient lane chunk is wholly valid or wholly outside the input. Packed row alignment and cross-word fields remain unchanged. Mixed gate/up widths retain separate generated unpack logic. This changes no codebook values, input/output dtype, clamp, Hadamard transform, expert validation or launch geometry. Gate/up retains per-block partial accumulation; down retains per-row sequential coefficient accumulation.

Status: independently source reviewed; numerical and performance validation pending. Existing mixed-bit, tail, invalid-route, dtype and full routed composition tests must pass before admission. The synthetic small-bank benchmark motivated investigation but does not establish model throughput or large-bank physical footprint. No loader or factory behavior changes.
