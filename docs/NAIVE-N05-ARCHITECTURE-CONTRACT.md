# Naive N0.5 architecture metadata

`NaiveN05ArchitectureContract` decodes the model's architecture without registering or loading an executable model. It does not parse or override sampling, reasoning, tool, or generation settings.

The source contract is NaiveAI/Naive-N0.5-Flash revision `0235b3b5ff27422b1f57cdc2acddfaf643e08356`, specifically `configuration_naive_n05_flash.py`. Missing fields use that configuration's defaults. Explicit null geometry is rejected. Nullable or zero routed scaling resolves to one, preserving both the requested nullable value and whether the field was present. Empty or null schedules use the vendor-derived schedules; explicit schedules take precedence. The vendor itself regenerates `layer_types` from `hybrid_layer_pattern` during validation.

This implementation supports a bounded subset: split Q/K/V, one indexer KV head, enabled DSA, sigmoid/noaux_tc routing, one expert group, SiLU, untied embeddings and no shared experts. A vendor-valid variant outside that subset produces `ContractError.unsupported`, rather than being described as corrupt. Derived schedule allocation is capped at 4,096 layers; this is an implementation limit, not a vendor model limit.

Five Foundation XCTest methods cover default schedules, asymmetric attention and rotary dimensions, nullable/falsy values, explicit schedules, unsupported variants and malformed types. A standalone XCTest bundle ran these five methods successfully; an independent metadata probe parsed a local 48-layer bundle and confirmed its 39 sliding / 9 sparse attention layers and 47 routed layers. Full-package compilation, model implementation, GPU execution and live generation remain separate gates. No factory support or speed claim follows from these metadata tests.
