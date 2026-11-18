"""Strict, shard-streamed SenseNova release -> text-only MLX export.

Original shards are immutable. Run on the CPU to avoid overlapping GPU/model
work; FP32 routers and the authored FP32 norm shift are preserved. Optional
affine quantization is a separate, recorded weight arm (not KV quantization).
"""

import argparse
import gc
import hashlib
import json
import shutil
import struct
from pathlib import Path

import mlx.core as mx

REVISION = "329a65f701ba8c48b73c30d0cbaa6f0449733673"
NORMS = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def align_shard(source, destination):
    """Header-only repack; independently reread/hash the unchanged payload."""
    sizes = {
        "F16": 2,
        "BF16": 2,
        "F32": 4,
        "F64": 8,
        "U32": 4,
        "I32": 4,
        "I64": 8,
        "U8": 1,
        "I8": 1,
        "BOOL": 1,
    }
    with source.open("rb") as fin:
        length = struct.unpack("<Q", fin.read(8))[0]
        assert length <= 16 * 1024**2
        header = fin.read(length)
        assert len(header) == length
        metadata = json.loads(header)
        padding = (-(8 + length)) % 8
        aligned = header + b" " * padding
        assert json.loads(aligned) == metadata
        for key, item in metadata.items():
            if key != "__metadata__":
                assert (8 + len(aligned) + item["data_offsets"][0]) % sizes[
                    item["dtype"]
                ] == 0, key
        digest = hashlib.sha256()
        with destination.open("xb") as fout:
            fout.write(struct.pack("<Q", len(aligned)))
            fout.write(aligned)
            while block := fin.read(1024**2):
                digest.update(block)
                fout.write(block)
    with destination.open("rb") as verify:
        verify.seek(8 + len(aligned))
        actual = hashlib.file_digest(verify, "sha256").hexdigest()
    assert actual == digest.hexdigest(), "alignment changed tensor payload"
    assert destination.stat().st_size == source.stat().st_size + padding
    return {"padding_bytes": padding, "payload_sha256": actual, "unaligned_after": 0}


def expected_text_shapes(config):
    t = config["text_config"]
    h, intermediate = t["hidden_size"], t["intermediate_size"]
    heads, kv, dim = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    kh, vh = t["linear_num_key_heads"], t["linear_num_value_heads"]
    dk, dv = t["linear_key_head_dim"], t["linear_value_head_dim"]
    prefix = "model.language_model."
    shapes = {
        prefix + "embed_tokens.weight": [t["vocab_size"], h],
        prefix + "norm.weight": [h],
        "lm_head.weight": [t["vocab_size"], h],
    }
    for i in range(t["num_hidden_layers"]):
        base = f"{prefix}layers.{i}."
        shapes.update(
            {
                base + "input_layernorm.weight": [h],
                base + "post_attention_layernorm.weight": [h],
                base + "mlp.gate_proj.weight": [intermediate, h],
                base + "mlp.up_proj.weight": [intermediate, h],
                base + "mlp.down_proj.weight": [h, intermediate],
            }
        )
        if (i + 1) % 4:
            conv_dim = 2 * kh * dk + vh * dv
            shapes.update(
                {
                    base + "linear_attn.in_proj_qkv.weight": [conv_dim, h],
                    base + "linear_attn.in_proj_z.weight": [vh * dv, h],
                    base + "linear_attn.in_proj_a.weight": [vh, h],
                    base + "linear_attn.in_proj_b.weight": [vh, h],
                    base + "linear_attn.out_proj.weight": [h, vh * dv],
                    base + "linear_attn.conv1d.weight": [
                        conv_dim,
                        1,
                        t["linear_conv_kernel_dim"],
                    ],
                    base + "linear_attn.A_log": [vh],
                    base + "linear_attn.dt_bias": [vh],
                    base + "linear_attn.norm.weight": [dv],
                }
            )
        else:
            shapes.update(
                {
                    base + "self_attn.q_proj.weight": [2 * heads * dim, h],
                    base + "self_attn.k_proj.weight": [kv * dim, h],
                    base + "self_attn.v_proj.weight": [kv * dim, h],
                    base + "self_attn.o_proj.weight": [h, heads * dim],
                    base + "self_attn.q_norm.weight": [dim],
                    base + "self_attn.k_norm.weight": [dim],
                }
            )
        if config["mhc"]["start"] <= i < config["mhc"]["end"]:
            for kind in ["mhc_attn", "mhc_ffn"]:
                shapes[base + kind + ".weight"] = [8, 2 * h]
                shapes[base + kind + ".bias"] = [8]
    return shapes


def destination(key):
    if key == "model.language_model.loop_gate.logits":
        return None, "authored-unused-loop-gate"
    if key.startswith("model.visual."):
        return None, "text-only-export-vision-excluded"
    if key.startswith("model.language_model."):
        return key.replace("model.language_model.", "model.", 1), None
    if key == "lm_head.weight":
        return key, None
    raise ValueError(f"unrecognized source tensor: {key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hub-manifest", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=[4, 6, 8])
    parser.add_argument("--float32-activation", action="store_true")
    args = parser.parse_args()
    if not __debug__:
        parser.error("run without Python optimization so checkpoint validation stays enabled")
    if args.float32_activation and args.bits not in {6, 8}:
        parser.error("--float32-activation requires --bits 6 or --bits 8")
    args.out.mkdir(parents=True, exist_ok=False)
    mx.set_default_device(mx.cpu)
    config = json.loads((args.source / "config.json").read_text())
    assert config["model_type"] == "sensenova_loop"
    assert config["loop_settings"] == {
        "start": 4,
        "end": 28,
        "eval_count": 2,
        "counts": [2],
    }
    assert config["text_config"]["mtp_num_hidden_layers"] == 0
    index = json.loads((args.source / "model.safetensors.index.json").read_text())
    shapes = expected_text_shapes(config)
    declared = set(index["weight_map"])
    allowed_extras = {"model.language_model.loop_gate.logits"}
    extras = declared - set(shapes) - allowed_extras
    assert not (set(shapes) - declared), "missing text tensors in source index"
    assert all(k.startswith("model.visual.") for k in extras), (
        "unexpected text/MTP source keys"
    )
    manifest = json.loads(args.hub_manifest.read_text())
    assert manifest["sha"] == REVISION
    blobs = {f["rfilename"]: f for f in manifest["siblings"]}
    output_map, seen, inventory, receipts = {}, set(), [], []
    alignment_receipts = {}
    source_shards = sorted(set(index["weight_map"].values()))
    for shard_index, shard in enumerate(source_shards, 1):
        path = args.source / shard
        blob = blobs[shard]
        assert path.stat().st_size == blob["size"], shard
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        assert digest == blob["lfs"]["sha256"], f"source digest mismatch: {shard}"
        weights = mx.load(str(path))
        expected = {k for k, v in index["weight_map"].items() if v == shard}
        assert set(weights) == expected, f"shard/index disagreement: {shard}"
        output = {}
        for key, tensor in weights.items():
            assert key not in seen, key
            seen.add(key)
            if key in shapes:
                assert list(tensor.shape) == shapes[key], f"wrong source shape: {key}"
            elif key == "model.language_model.loop_gate.logits":
                assert list(tensor.shape) == [config["text_config"]["hidden_size"]]
            name, reason = destination(key)
            record = {
                "source_key": key,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "destination_key": name,
                "exclusion": reason,
            }
            inventory.append(record)
            if name is None:
                continue
            assert name not in output_map and name not in output, name
            if ".mhc_attn." in name or ".mhc_ffn." in name:
                assert tensor.dtype == mx.float32, (
                    f"router source precision changed: {key}"
                )
            if name.endswith("conv1d.weight"):
                assert tensor.ndim == 3 and tensor.shape[1] == 1
                tensor = mx.transpose(tensor, (0, 2, 1))
            if name.endswith(NORMS):
                assert tensor.ndim == 1
                tensor = tensor.astype(mx.float32) + 1
                record["transform"] = "raw_norm.float32+1"
            # Validate source values before lossy quantization or serialization.
            assert bool(mx.all(mx.isfinite(tensor)).item()), f"non-finite source: {key}"
            quantizable = (
                args.bits is not None
                and tensor.ndim == 2
                and name.endswith(".weight")
                and tensor.shape[-1] % 64 == 0
                and ".mhc_" not in name
            )
            if quantizable:
                q, scales, biases = mx.quantize(tensor, group_size=64, bits=args.bits)
                prefix = name.removesuffix(".weight")
                output[name] = q
                output[prefix + ".scales"] = scales
                output[prefix + ".biases"] = biases
                record["quantization"] = {"bits": args.bits, "group_size": 64}
            else:
                output[name] = tensor
        if output:
            filename = f"model-{shard_index:05d}.safetensors"
            staging = args.out / (".writing-" + filename)
            mx.save_safetensors(
                str(staging),
                output,
                metadata={
                    "format": "mlx",
                    "norm_convention": "mlx_plus_one",
                    "source_revision": REVISION,
                },
            )
            alignment = align_shard(staging, args.out / filename)
            staging.unlink()
            alignment_receipts[filename] = alignment
            for name in output:
                output_map[name] = filename
        receipts.append({"file": shard, "sha256": digest, "bytes": path.stat().st_size})
        del weights, output
        gc.collect()
        mx.clear_cache()
        print(
            f"converted shard {shard_index}/{len(source_shards)}: {shard}", flush=True
        )
    assert seen == set(index["weight_map"])
    # Do not let the generic host select a vision factory for this text export.
    for name in list(config):
        if name.startswith("vision_") or name in {
            "vision_config",
            "image_token_id",
            "video_token_id",
            "auto_map",
            "patch_conv3d",
        }:
            del config[name]
    config["architectures"] = ["SenseNovaLoopModel"]
    config["norm_convention"] = "mlx_plus_one"
    config["source_model"] = "sensenova/SenseNova-Loop-9B"
    config["source_revision"] = REVISION
    config["mei_text_only_export"] = True
    if args.bits:
        config["quantization"] = {"group_size": 64, "bits": args.bits}
    if args.float32_activation:
        config["activation_dtype"] = "float32"
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
    ]:
        shutil.copyfile(args.source / name, args.out / name)
    (args.out / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": output_map}, indent=2)
    )
    report = {
        "source_revision": REVISION,
        "weight_bits": args.bits,
        "source_receipts": receipts,
        "output_alignment": alignment_receipts,
        "tensor_inventory": inventory,
        "output_keys": len(output_map),
        "bytes": sum(p.stat().st_size for p in args.out.iterdir() if p.is_file()),
    }
    (args.out / "conversion-report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "tensor_inventory"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
