"""Bounded CPU reference: authored forward with candidate affine weights.

Swap one module's quantized weight into the mmap-backed authored model for its
forward, then restore the original parameter. This avoids a second resident
dequantized model. Supports affine exports with group size 64.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def dequantize_affine(packed, scales, biases, bits=8, rows=512, dtype=torch.bfloat16):
    """Independent packed integer decoder with an explicit output rounding."""
    assert packed.dtype == torch.uint32
    assert scales.dtype == biases.dtype == torch.bfloat16
    assert packed.ndim == scales.ndim == biases.ndim == 2
    assert scales.shape == biases.shape
    assert bits in (4, 6, 8)
    height, words_per_row = packed.shape
    width = words_per_row * 32 // bits
    assert words_per_row * 32 == width * bits
    assert width == scales.shape[1] * 64
    assert dtype in (torch.bfloat16, torch.float32)
    result = torch.empty((height, width), dtype=dtype)
    for start in range(0, height, rows):
        end = min(start + rows, height)
        octets = packed[start:end].numpy().astype("<u4", copy=False).view(np.uint8)
        if bits == 8:
            values = octets
        elif bits == 4:
            values = np.stack((octets & 15, octets >> 4), axis=-1)
        else:
            # Four consecutive 6-bit codes fill three little-endian bytes.
            a, b, c = octets.reshape(end - start, -1, 3).transpose(2, 0, 1)
            values = np.stack(
                (
                    a & 63,
                    ((a >> 6) | (b << 2)) & 63,
                    ((b >> 4) | (c << 4)) & 63,
                    c >> 2,
                ),
                axis=-1,
            )
        grouped = values.astype(np.float32).reshape(end - start, -1, 64)
        scale = scales[start:end].float().numpy()[..., None]
        bias = biases[start:end].float().numpy()[..., None]
        result[start:end] = torch.from_numpy(
            (grouped * scale + bias).reshape(end - start, width)
        )
    return result


def install(model, directory, affine_linear_fp32=False, float32_activation=False):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    bits = config["quantization"]["bits"]
    assert bits in (4, 6, 8)
    assert config["quantization"] == {"group_size": 64, "bits": bits}
    index = json.loads((directory / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    handles, originals, hooks, input_dtypes = {}, {}, [], {}
    for filename in set(index.values()):
        handles[filename] = safe_open(
            directory / filename, framework="pt", device="cpu"
        )

    def get(key):
        return handles[index[key]].get_tensor(key)

    count = 0
    for name, module in model.named_modules():
        canonical = name.replace("model.language_model.", "model.", 1)
        quantized = canonical + ".scales" in index
        parameters = dict(module.named_parameters(recurse=False))
        if not parameters or (not quantized and not float32_activation):
            continue
        if quantized:
            assert isinstance(module, (torch.nn.Linear, torch.nn.Embedding)), name
        originals[name] = parameters

        def before(mod, inputs, name=name, key=canonical, quantized=quantized):
            for param, original in originals[name].items():
                assert getattr(mod, param) is original, name
                if (
                    float32_activation
                    and original.dtype != torch.float32
                    and not (quantized and param == "weight")
                ):
                    setattr(
                        mod,
                        param,
                        torch.nn.Parameter(original.float(), requires_grad=False),
                    )
            linear_fp32 = affine_linear_fp32 and isinstance(mod, torch.nn.Linear)
            if quantized:
                weight = dequantize_affine(
                    get(key + ".weight"),
                    get(key + ".scales"),
                    get(key + ".biases"),
                    bits=bits,
                    dtype=torch.float32
                    if linear_fp32 or float32_activation
                    else torch.bfloat16,
                )
                assert weight.shape == mod.weight.shape, name
                mod.weight = torch.nn.Parameter(weight, requires_grad=False)
            if linear_fp32 and not float32_activation:
                assert len(inputs) == 1 and mod.bias is None
                input_dtypes[name] = inputs[0].dtype
                return (inputs[0].float(),)

        def after(mod, inputs, output, name=name):
            for param, original in originals[name].items():
                setattr(mod, param, original)
            if name in input_dtypes and output is not None:
                return output.to(input_dtypes.pop(name))

        hooks.append(module.register_forward_pre_hook(before))
        hooks.append(module.register_forward_hook(after, always_call=True))
        count += int(quantized)
    expected = sum(key.endswith(".scales") for key in index)
    assert count == expected, (count, expected)
    # Keep mmap handles and hook closures alive through all repeated visits.
    model._candidate_reference = (handles, originals, hooks)
    return count
