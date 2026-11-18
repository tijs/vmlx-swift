"""Bounded authored BF16 CPU reference on frozen release prompt token IDs."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--quantized-weights", type=Path)
    p.add_argument("--generate", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--affine-linear-fp32", action="store_true")
    p.add_argument("--float32-activation", action="store_true")
    p.add_argument("--preserve-source-fp32", action="store_true")
    p.add_argument("--frozen-token-ids", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    original = json.loads(args.cases.read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        args.source, local_files_only=True, trust_remote_code=True
    )
    cases = []
    for case in original["cases"]:
        prompt = tokenizer.apply_chat_template(
            case["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if args.frozen_token_ids:
            ids = case["ids"]
            assert ids and all(
                type(value) is int and 0 <= value < len(tokenizer) for value in ids
            )
            positions = case["positions"]
            assert positions == sorted(set(positions)) and all(
                0 <= value < len(ids) for value in positions
            )
            prompt = case["prompt"]
        else:
            positions = list(range(max(0, len(ids) - 4), len(ids)))
        cases.append(
            {
                **case,
                "prompt": prompt,
                "ids": ids,
                "positions": positions,
            }
        )
    config = json.loads((args.source / "generation_config.json").read_text())
    eos = config.get("eos_token_id", tokenizer.eos_token_id)
    protocol = {
        "cases": cases,
        "eos_ids": eos if isinstance(eos, list) else [eos],
        "sampling": {"temperature": 0, "thinking": False},
        "reference_dtype": "float32" if args.float32_activation else "bfloat16",
        "logit_abs_gate": 0.25,
        "logit_rms_gate": 0.03,
        "kl_gate": 0.005,
        "top1_rule": "exact at every selected teacher-forced position; ties reported, not silently passed",
        "quantized_weight_directory": str(args.quantized_weights)
        if args.quantized_weights
        else None,
        "affine_linear_fp32": args.affine_linear_fp32,
        "preserve_source_fp32": args.preserve_source_fp32,
        "frozen_token_ids": args.frozen_token_ids,
    }
    (args.out / "cases.json").write_text(json.dumps(protocol, indent=2))
    if args.prepare_only:
        print("Frozen release-native rendering and token IDs prepared.")
        return
    torch.set_num_threads(2)
    start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.source,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    if args.preserve_source_fp32:
        # from_pretrained(dtype=BF16) rounds these original FP32 parameters.
        # Restore them from the independently pinned original shards so the
        # native preserve-checkpoint-dtypes arm has a matched reference.
        index = json.loads((args.source / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        keys = [key for key in index if key.endswith((".A_log", ".dt_bias"))]
        assert len(keys) == 48
        for key in keys:
            with safe_open(args.source / index[key], framework="pt", device="cpu") as f:
                value = f.get_tensor(key)
            assert value.dtype == torch.float32
            parent, name = key.rsplit(".", 1)
            module = model.get_submodule(parent)
            assert getattr(module, name).shape == value.shape
            setattr(module, name, torch.nn.Parameter(value, requires_grad=False))
        print(json.dumps({"restored_original_fp32_parameters": keys}), flush=True)
    if args.quantized_weights:
        from quantized_reference import install

        count = install(
            model,
            args.quantized_weights,
            affine_linear_fp32=args.affine_linear_fp32,
            float32_activation=args.float32_activation,
        )
        print(json.dumps({"candidate_quantized_modules": count}), flush=True)
    print(
        json.dumps({"stage": "loaded", "seconds": time.monotonic() - start}), flush=True
    )
    output = {}
    rows = []
    with torch.inference_mode():
        for case in cases:
            ids = torch.tensor([case["ids"]], dtype=torch.long)
            begin = time.monotonic()
            result = model(
                input_ids=ids, use_cache=False, output_hidden_states=args.trace
            )
            if args.trace:
                save_file(
                    {
                        f"hidden_{i}": h[:, -1:, :].float().contiguous()
                        for i, h in enumerate(result.hidden_states)
                    },
                    args.out / f"trace-{case['name']}.safetensors",
                )
            result = result.logits
            seconds = time.monotonic() - begin
            selected = result[:, case["positions"], :].float().contiguous()
            output[case["name"]] = selected
            row = {
                "case": case["name"],
                "prompt_tokens": len(case["ids"]),
                "seconds": seconds,
                "teacher_forced_tokens_per_second": len(case["ids"]) / seconds,
                "selected_top1": selected.argmax(-1).tolist(),
                "logits_sha256": hashlib.sha256(selected.numpy().tobytes()).hexdigest(),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            # Preserve each completed teacher case even if later generation
            # hits a resource/deadline guard. Partial captures never rank.
            save_file(output, args.out / "expected.safetensors")
            (args.out / "reference.partial.json").write_text(
                json.dumps({"rows": rows, "complete": False}, indent=2)
            )
            if args.generate:
                generated = model.generate(
                    ids,
                    do_sample=False,
                    max_new_tokens=case["max_tokens"],
                    eos_token_id=protocol["eos_ids"],
                    pad_token_id=tokenizer.pad_token_id,
                )[0, len(case["ids"]) :].tolist()
                row["generated_ids"] = generated
                row["text"] = tokenizer.decode(generated, skip_special_tokens=True)
                print(
                    json.dumps({"case": case["name"], "generated": row["text"]}),
                    flush=True,
                )
    save_file(output, args.out / "expected.safetensors")
    (args.out / "reference.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "torch": torch.__version__,
                "total_seconds": time.monotonic() - start,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
