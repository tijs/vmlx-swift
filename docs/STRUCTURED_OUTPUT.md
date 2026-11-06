# JSON Schema constrained output

This proposed runtime feature applies a compiled JSON grammar to each next-token distribution when a request explicitly supplies `GenerateParameters.jsonSchema`. It does not add closing-token bias, whitespace bias, prompt instructions, thinking tags, or sampler defaults. The existing request/bundle sampler selects among the allowed tokens.

## Request contract

```swift
var parameters = GenerateParameters()
parameters.jsonSchema = #"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}"#
// Pass parameters to the ordinary prepared-input generation API.
```

Schema requests use autoregressive decoding only. Speculative/native-MTP draft strategies are disabled for that request; ordinary requests keep their existing behavior. Model KV/prefix/SSD caching, including full-precision disk state, remains independent of grammar state. Each request owns a fresh matcher. Processor copies replay accepted token IDs into an independent matcher; matcher history is not restored from the model's SSD prefix cache.

The request must have an exact supported tokenizer vocabulary contract. The current bundle adapter accepts BPE with a plain ByteLevel decoder, empty continuation/end-of-word suffixes, and explicitly disabled tokenization-space cleanup. It maps added tokens using their literal bytes and excludes non-stop special IDs and vocabulary gaps. Decoder sequences, byte fallback, Metaspace and unknown tokenizer metadata are not qualified by this adapter. The lower-level grammar vocabulary enum is broader than the bundle adapter's admitted coverage.

Active reasoning envelopes and tool-call envelopes are not qualified and are explicitly rejected at request preparation. The implementation does not silently change a model's reasoning mode. A caller may explicitly choose the model's native non-reasoning mode, where available. Block-diffusion generation is also rejected. A completed object is not sufficient for success: the matcher must admit and consume a configured stop token. Length limits, cancellation, processor failures and incomplete schemas are not schema-success receipts. Consumers must inspect completion/failure status and must not treat partial streamed text as a validated result.

## Schema subset

`JSONSchemaGrammar.validateSupportedSchema(_:)` rejects unsupported keywords and combinations before compilation, using `JSONSchemaGrammarError`. This is a deliberately limited contract, not general JSON Schema support.

Admitted structures:

- Primitive `type`, or a nonempty distinct array of primitive type names; `true`/empty schemas.
- Object `properties`, `required` and boolean/schema `additionalProperties`. Required names must be declared properties. Nonempty named `properties` require explicit `additionalProperties:false`. Object constraints require an explicit object type.
- Array schema `items`, `minItems` and `maxItems` (integer bounds 0...1024). Array constraints require an explicit array type.
- `enum` or `const`, optionally with a consistent type, without other constraint siblings. Enums contain 1...1024 values.
- `anyOf` with 1...128 branches and no other constraint siblings.
- `$defs`/`definitions` and acyclic local `$ref` (simple `#/...` schema paths), without constraint siblings. Escaped/percent-encoded pointers, external references, cycles, and targets inside annotations or instance data are rejected. All reference chains are resolved against validated schema locations.
- Annotation fields `title`, `description`, `$comment`, `default`, `examples`; these do not constrain output. Explicit `$schema` dialect declarations are rejected until a dialect is qualified.

Generic objects without named properties retain the permissive default when `additionalProperties` is omitted. The bridge explicitly compiles with `strict_mode=false`; it does not silently insert `additionalProperties:false`.

Nonempty named-property schemas with omitted, true or schema-valued `additionalProperties` are rejected. The pinned donor can otherwise admit a repeated named key through its additional-property rule with the wrong value type; a CPU matcher reproduction confirmed this bypass. The runtime does not silently close such schemas.

Rejected features include `oneOf`, `allOf`, `not`, conditionals, dependencies, numeric bounds/`multipleOf`, `uniqueItems`, `contains`, `pattern`, `format`, `minLength`, `maxLength`, and unknown keywords. `false` schemas are rejected rather than compiling an empty language. Schema input is limited to 1 MiB, nesting depth 64 and 10,000 visited nodes.

String-length constraints are rejected: the pinned compiler's constrained-string branch does not implement the full JSON escaped-character language. Ordinary strings use the separate JSON string grammar. Recursive references and explicit dialect declarations are rejected pending qualification.

Low-level use with an exact vocabulary:

```swift
let vocabulary = try JSONSchemaGrammarTokenizer(
    vocabulary: rawPieces, vocabularyType: .byteLevel,
    stopTokenIDs: explicitStopIDs)
let grammar = try JSONSchemaGrammar(tokenizer: vocabulary, schema: schemaJSON)
let mask = try grammar.nextTokenMask() // packed UInt32 bits indexed by token ID
// Apply the actual mask, then sample using the existing sampler.
try grammar.accept(tokenID: sampledID)
let complete = try grammar.isTerminated()
let independent = try grammar.independentCopy()
```

The engine processor handles mask application and fail-closed errors. Application code should normally use the request parameter instead of implementing a second decode loop.

## Architect integration boundary

An Architect `generate` or `decide` node can supply a schema to this same generation request seam and consume the completed typed result. This change does not implement a workflow graph, input validation nodes, cloud provider nodes, agent planning, tool execution or a separate agentic runtime. Schema validation of arbitrary incoming user/workflow data remains separate from constrained model generation.

## Provenance and proof limits

The C shim and XGrammar sources are derived from `ml-explore/mlx-swift-lm` commit `22157fc397b59acfb03e91c370bcbf2cfb10970e`. Embedded XGrammar is v0.1.30, commit `d476a48dcd8fa3b5afeddbe850e73bb3b1dcf505`. Licenses/notices and local compatibility changes are recorded in `Libraries/MLXCXGrammar/PROVENANCE.md`. No upstream guided loop, completion reserve, closing/whitespace biases or heuristic tokenizer extractor is imported.

CPU grammar tests exercise masks, explicit stop IDs, independent copies, defaults and fail-closed schema admission. Separately, actual Release app HTTP qualification `live-run002` passed 15 bounded cases with engine `6e41dc752071868f2433bcc02c2e49da416096ca` and app `db5e11ab53c534dad2f4386286c221d6d2158a32`. The local Raptor 4B JANG_6M bundle used explicit native non-thinking selection and no sampler overrides. Coverage included Chat/Responses stream and nonstream schema output, changed-schema same-prefix requests, multi-turn, literal Unicode/protocol text, root string enum, explicit token-limit failure, unsupported pattern/length rejection, ordinary chat/tool-call/follow-up controls, disconnect drain and a subsequent fresh schema request. The four primary schema endpoint rows reported 95.30–97.23 tokens/s; this is a bounded receipt, not a general speed claim.

The successful generated outputs were checked with an independent JSON Schema validator. The 15 cases include expected failure/cancellation controls and are not 15 successful generations. Same-prefix requests alone do not prove SSD hits: disk/cache reuse requires separate before/after telemetry and server evidence. This run does not qualify all models/tokenizers, active reasoning or tool envelopes under schema constraints, arbitrary schema keywords, UI workflows, or universal cancellation/cache behavior. Source formatting changes after the recorded revision require their own build identity; no fresh live run is implied by this paragraph.

### Exact numeric constants
Numeric values inside `const` or `enum`, including nested instance objects/arrays, currently require plain integer literals in the inclusive range -9007199254740991 through 9007199254740991. Fractional and exponent spellings are rejected before parsing can round them. Ordinary `type:number` generation remains supported. This restriction prevents the donor compiler from compiling a different numeric constant than requested. JSON property names are serialized as JSON strings before EBNF escaping, including controls, quotes, backslashes and Unicode.

## Deterministic JSON formatting

Schema grammar compilation uses deterministic structural formatting: no indentation and explicit comma/colon separators, with `any_whitespace=false`. This removes unbounded structural whitespace choices that can otherwise consume the token budget. String spaces, escaped newlines/tabs, Unicode, enum/const values and property names retain their data semantics. This is a serialization grammar policy, not a whitespace logit bias or forced EOS. Generic unconstrained collection rules in the pinned donor retain a fixed comma-space separator; that bounded formatting does not introduce a whitespace loop. Natural grammar-authorized completion is still required; formatting alone is not a successful-generation proof.
