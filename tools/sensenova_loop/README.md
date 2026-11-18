# SenseNova release integration tools

These tools prepare and test the experimental Mei text profile.
The authored release is pinned to
`sensenova/SenseNova-Loop-9B@329a65f701ba8c48b73c30d0cbaa6f0449733673`.
The Python implementation SHA256 is
`26e11d3522aa730c0ba031340ce42632bc2834da8cc0b6804006bb207d979a5b`.
Plans and progress live in Kiem project `proj/mei`.

Use an isolated Python 3.12 environment with Torch 2.11.0, Transformers 5.18.0,
NumPy 2.5.5, safetensors and MLX 0.32.4. The Swift candidate uses its repository's
own pinned MLX/Cmlx graph; do not replace that graph with the Python package.

* `reference.py --source MODELING_PY --out NEW_DIRECTORY [--dtype bfloat16]`
  creates deterministic tiny authored fixtures. It loads no release weights.
* `convert.py --source ORIGINAL_DIRECTORY --out NEW_DIRECTORY
  --hub-manifest PINNED_HUB_JSON --bits 6 --float32-activation` verifies original shard digests,
  exact text shapes and finite values, exports text weights with FP32 routers
  and shifted FP32 norms, and records exclusions, transformations and alignment
  receipts. Every output directory must be new. Original shards are immutable.
  Header alignment preserves payload bytes; the current Swift public loader
  uses resident storage, so alignment is not a residency fix.
  Conversion requires Python MLX. The Torch and Transformers dependencies
  are for the independent reference diagnostics.
* `capture_reference.py --source ORIGINAL_DIRECTORY --cases CASES_JSON
  --out NEW_DIRECTORY [--prepare-only] [--quantized-weights Q8_DIRECTORY]
  [--generate] [--trace]` renders the original tokenizer/template and captures
  teacher logits using the authored CPU forward. The Q8 diagnostic replaces
  only the executing module's weight with an independently unpacked affine
  weight, then restores its original mmap-backed parameter. It does not allocate
  a second full dequantized model. Its BF16 rounding differs from some fused
  backend implementations; report this boundary in numerical comparisons.
  For the native checkpoint-preserving control, add `--preserve-source-fp32`:
  Transformers' BF16 loader otherwise rounds the release's 48 FP32 `A_log` and
  `dt_bias` parameters. Add `--float32-activation` to match the experimental
  Q8/FP32 native profile; all floating affine metadata/small parameters use
  FP32 without creating a resident full dequantized model. The separate
  `--affine-linear-fp32` diagnostic retains BF16 activations while avoiding
  an early BF16 rounding of affine linear weights. Name these arms separately.
  `--frozen-token-ids` accepts previously frozen integer IDs/positions for
  following-token checks; detokenized HTTP bytes may retokenize differently,
  so they are not an authoritative substitute for emitted token IDs.
* `SenseNovaProbe.swift` is a standalone single-process native admission probe,
  not the Mei HTTP server. It requires `MODEL_DIRECTORY CASES_JSON NEW_OUTPUT`
  and optionally `--trace`. Build it in a disposable Swift package against the
  exact local candidate, with `MLX`, `MLXNN`, `MLXLLM` and `MLXLMCommon` products
  and byte-verified Metal resources from the matching Cmlx build.
  `--check-chunked-prefill` compares full forward with 64-token prefix chunks
  plus TokenIterator's held-back final token. It preserves logits and all
  56 cache offsets before checking the existing frozen absolute/RMS/top-1 gates.
  Admission failure before model loading does not establish long-forward
  memory requirements or chunk correctness.
* `compare.py --reference REFERENCE_DIRECTORY --native PROBE_DIRECTORY
  --tokenizer ORIGINAL_DIRECTORY --out NEW_JSON` compares the frozen teacher
  logits and grades actual complete native outputs. A comparison across
  different weight precisions cannot isolate implementation error.

Run each real model process through
`guard_run.py --out NEW_GUARD_DIRECTORY --deadline SECONDS -- COMMAND ...`.
It records host pressure, swap and the owned process's physical footprint;
stops on critical/sustained warning pressure, 256 MiB swap growth, 24 GiB
footprint, the 20 GiB disk floor, or deadline; and owns only that direct process
group. It is not the formal benchmark supervisor. Keep one inference owner,
CoCore off, and builds separate from timed runs. Preserve failed and partial
attempts; neither qualifies as a pass.

CPU converter/decoder checks:

```sh
SENSENOVA_FIXTURE_DIR=/absolute/tiny-fixture python -m unittest discover \
  -s tools/sensenova_loop -p 'test_*.py'
```

Swift `SenseNovaLoopTests` require `SENSENOVA_FIXTURE_DIR` and
`SENSENOVA_BF16_FIXTURE_DIR` for full validation. Without them, fixture tests
skip, so a green default run does not establish parity.
