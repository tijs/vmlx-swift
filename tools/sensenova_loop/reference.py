"""Generate tiny fixtures using the pinned author's implementation, CPU only.

No release weights are loaded. NumPy router calculations are independent of the
Torch and Swift implementations. The fixture retains all 32 physical layers and
56 visits, including nonzero input-dependent routers.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


def numpy_router(residual, weight, bias):
    flat = residual.reshape(*residual.shape[:-2], -1).astype(np.float64)
    norm = flat / np.sqrt(np.mean(flat * flat, axis=-1, keepdims=True) + 1e-6)
    logits = norm @ weight.astype(np.float64).T + bias
    pre = 1 / (1 + np.exp(-logits[..., :2]))
    post = 2 / (1 + np.exp(-logits[..., 2:4]))
    mix = logits[..., 4:].reshape(*logits.shape[:-1], 2, 2)
    for _ in range(20):
        for axis in (-1, -2):
            top = mix.max(axis=axis, keepdims=True)
            mix -= top + np.log(np.exp(mix - top).sum(axis=axis, keepdims=True))
    mix = np.exp(mix)
    branch = (pre[..., None, :] @ residual).squeeze(-2)
    return branch, post, mix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.source.read_bytes()
    assert hashlib.sha256(source).hexdigest() == (
        "26e11d3522aa730c0ba031340ce42632bc2834da8cc0b6804006bb207d979a5b"
    ), "reference implementation identity changed"
    spec = importlib.util.spec_from_file_location(
        "authored_sensenova_reference", args.source
    )
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    torch.set_num_threads(2)
    torch.manual_seed(17917)
    config = {
        "model_type": "sensenova_loop",
        "hidden_size": 64,
        "tie_word_embeddings": False,
        "eos_token_id": 1,
        "pad_token_id": 0,
        "loop_settings": {"start": 4, "end": 28, "eval_count": 2, "counts": [2]},
        "mhc": {
            "start": 4,
            "end": 28,
            "streams": 2,
            "iterations": 20,
            "pre_scale": 1.0,
            "fixed_identity": False,
        },
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 128,
            "max_position_embeddings": 524288,
            "rms_norm_eps": 1e-6,
            "full_attention_interval": 4,
            "attn_output_gate": True,
            "attention_bias": False,
            "tie_word_embeddings": False,
            "hidden_act": "silu",
            "dtype": "float32",
            "mtp_num_hidden_layers": 0,
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 262144,
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [1, 1, 0],
                "mrope_interleaved": True,
            },
            "partial_rotary_factor": 0.25,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "layer_types": [
                "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                for i in range(32)
            ],
        },
        "vision_config": {
            "model_type": "qwen3_5_vision",
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 64,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
    }
    config["dtype"] = args.dtype
    config["text_config"]["dtype"] = args.dtype
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    plan = ref.execution_plan(config)
    assert len(plan) == 56 and sum((i + 1) % 4 == 0 for i in plan) == 14
    model = ref.SenseNovaLoopForConditionalGeneration(ref.SenseNovaLoopConfig(**config))
    model = model.float().eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".mhc_" in name:
                parameter.normal_(0, 0.06)
            elif (
                "language_model" in name
                and name.endswith("weight")
                and (
                    "layernorm" in name
                    or ".norm." in name
                    or "q_norm" in name
                    or "k_norm" in name
                )
            ):
                parameter.normal_(0, 0.08)
    if args.dtype == "bfloat16":
        model = model.to(torch.bfloat16)
    weights = {
        k: v.detach().contiguous()
        for k, v in model.state_dict().items()
        if k.startswith(("model.language_model.", "lm_head."))
    }
    save_file(weights, args.out / "weights.safetensors")
    ids = torch.tensor([[4, 17, 9, 25, 6, 11]], dtype=torch.long)
    router = model.model.language_model.layers[4].mhc_attn
    residual = torch.linspace(-1.1, 1.3, 3 * 2 * 64).reshape(1, 3, 2, 64)
    with torch.no_grad():
        actual = router(residual)
        oracle = numpy_router(
            residual.numpy(), router.weight.numpy(), router.bias.numpy()
        )
        errors = [float(np.max(np.abs(a.numpy() - b))) for a, b in zip(actual, oracle)]
        assert max(errors) < 1e-5, errors
        outputs = model(input_ids=ids, use_cache=False, output_hidden_states=True)
        prefix = model(input_ids=ids[:, :3], use_cache=True)
        suffix = model(
            input_ids=ids[:, 3:], past_key_values=prefix.past_key_values, use_cache=True
        )
        cached_error = (outputs.logits[:, 3:] - suffix.logits).abs().max().item()
        assert cached_error < (1e-4 if args.dtype == "float32" else 0.03), cached_error
        fixture = {
            "ids": ids.to(torch.int32),
            "logits": outputs.logits,
            "cached_logits": suffix.logits,
            "router_residual": residual,
            "router_branch": actual[0],
            "router_post": actual[1],
            "router_mix": actual[2],
        }
        for i, hidden in enumerate(outputs.hidden_states):
            fixture[f"hidden_{i}"] = hidden.contiguous()
        save_file(
            {k: v.detach().contiguous() for k, v in fixture.items()},
            args.out / "expected.safetensors",
        )
        # A one-pass mutation must actually change this fixture.
        original = model.model.language_model._loop_plan
        model.model.language_model._loop_plan = tuple(range(32))
        one_pass = model(input_ids=ids, use_cache=False).logits
        model.model.language_model._loop_plan = original
        mutation_delta = (one_pass - outputs.logits).abs().max().item()
        assert mutation_delta > 1e-3
    report = {
        "reference_sha256": hashlib.sha256(source).hexdigest(),
        "seed": 17917,
        "dtype": args.dtype,
        "physical_layers": 32,
        "visits": len(plan),
        "router_numpy_max_abs": errors,
        "torch_cached_max_abs": cached_error,
        "one_pass_mutation_max_abs": mutation_delta,
        "swift_tolerances": {"router_abs": 1e-5, "model_abs": 1e-3 if args.dtype == "float32" else 0.03},
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
    }
    (args.out / "reference-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
