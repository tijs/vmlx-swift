# JANGH tensor index admission

This experimental Foundation-only component validates bank locations before any
payload read or module construction. It does not enable a model factory or select
an architecture. Seven exact-source XCTest methods passed with zero failures in a standalone
Foundation-only module and test bundle. Production and test files were compiled
unchanged; the harness supplies a minimal module rather than the full package.
Full-package integration remains pending. A separate Foundation-only probe also admitted
real GLM metadata (126 projections, 252 banks) and Naive metadata (141 projections,
282 banks), with dimensions derived from architecture configuration and audited
projection definitions rather than packed shapes. Only config, index, and shard
headers were read; this was not a model load.

The caller supplies a validated `JANGHFormatContract`, an exact architecture-owned
projection dimension map, the safetensors index, and complete header summaries for
all indexed shards. Summaries exclude the safetensors `__metadata__` entry. Header
parsing and collection are separate responsibilities; this component cannot prove
that a caller supplied authentic or complete file contents. A future loader must
obtain summaries from open files and preserve file identity through mapping.

Admission checks exact custom-projection coverage, shard ownership and inventory,
relative shard filenames, bounded nonoverlapping tensor ranges, checked byte counts,
dtype alignment, U32/F16 shapes, and conflicting legacy representations. Unknown
custom tensors and missing banks fail explicitly. Ordinary tensors participate in
range-overlap checks but their architecture, dtype and index consistency remain the
generic loader's responsibility. Returned locations retain shard name, tensor name,
absolute file offset and byte count. No tensor is allocated or evaluated.

The bounded fixtures include mixed 2/3/6-bit banks, 96-channel rows, output tails,
missing/duplicate/unowned banks, unsafe paths, wrong dtype/geometry, integer overflow,
alignment, overlapping ranges, and legacy representation collisions.

Remaining work: reconcile config and sidecar declaration owners and both quantization
aliases; parse authentic index/header summaries; supply audited GLM dimensions and
canonical names; construct mapped custom modules without dense placeholders or
repacking; establish prefill and decode precision contracts; validate real model
memory, cache, multi-turn behavior and performance. Naive requires its own model
architecture. Configuration or location validation is not model execution support.

## File-backed metadata adapter (source prepared)

`JANGHHeaderAdapter` reads the actual index and length-prefixed shard headers into
planner summaries. The bundle root may be a user symlink; it is resolved and opened
once, then index/shard files are opened relative to that directory descriptor with
no-follow semantics. Each must be a regular file. Prefix/header reads have separate
per-file JSON and aggregate metadata caps (including shard prefixes); tensor payloads are never read. Descriptor
identity includes device, inode, file size, modification time and change time and
must remain unchanged across each read. Exact metadata bytes are retained for
provenance hashing. Typed descriptor decoding rejects nonintegral dimensions and
malformed offsets; a bounded scan rejects duplicate JSON keys, including escaped
aliases, before dictionary-derived metadata can silently discard declarations.

This adapter and its five file-fixture tests are newly authored and have not been
compiled or executed. Earlier planner evidence does not cover this adapter.
Tests prepare ordinary files, duplicate/escaped keys, malformed dimensions,
metadata size bounds, missing files, path traversal, forbidden file symlinks,
allowed bundle-root symlinks, and atomic file replacement identities.

A header snapshot is not a mapping lease. The future bank owner must preserve open
file identity through mapping or revalidate immediately when opening the mapped
bank; path identity alone is insufficient. Ordinary tensor semantic validation
remains the generic loader's responsibility. Config/sidecar alias reconciliation
and architecture descriptor construction are still separate missing adapters.
GLM dimensions use `hidden_size`, `moe_intermediate_size`, `n_routed_experts`, and
authoritative `mlp_layer_types`. Naive uses the same projection widths with its
own `moe_layer_freq` schedule; this does not supply its missing attention runtime.

The first adapter deliberately rejects symlinked shard/index files, including
legitimate Hugging Face snapshot links to cached blobs. Direct-file bundles and
symlinked bundle roots are the currently supported storage subset; this is not a
general Hugging Face loader. Before factory integration, legitimate blob links
need an audited resolution policy tied to retained descriptor identities rather
than treating symlink layout as invalid model format.
