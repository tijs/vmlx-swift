"""Independent NumPy FP32/BF16 fixtures for authored Inkling primitives."""
import json
from pathlib import Path
import numpy as np


def bf16(x):
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def cast(x, dtype):
    return bf16(x) if dtype == "bfloat16" else np.asarray(x, dtype=np.float32)


def sigmoid(x):
    return np.float32(1) / (np.float32(1) + np.exp(-x))


def main():
    fixture = dict(provenance="Independent NumPy FP32 arithmetic and BF16 RNE; no Swift oracle.",
                   numpy_version=np.__version__, clips=[], activations=[], combinations=[])
    for dtype in ("float32", "bfloat16"):
        x = cast(np.sin(np.arange(24, dtype=np.float32) * np.float32(.71)).reshape(1, 3, 8) * np.float32(3), dtype)
        for scaled in (False, True):
            scale = np.linspace(.3, 1.7, 8, dtype=np.float32) if scaled else None
            y = x.astype(np.float32) * (scale if scaled else np.float32(1))
            norm = np.sqrt(np.sum(y * y, axis=-1, keepdims=True, dtype=np.float32))
            assert np.all(norm > np.float32(1.6))  # half-threshold reentry oracle
            expected = cast(y * np.minimum(np.float32(.8) / (norm + np.float32(1e-6)), np.float32(1)), dtype)
            fixture["clips"].append(dict(dtype=dtype, x=x.tolist(), threshold=.8,
                                         scale=scale.tolist() if scaled else None, expected=expected.tolist()))
        gate = cast(np.linspace(-18, 18, 24, dtype=np.float32).reshape(1, 3, 8), dtype)
        up = cast(np.cos(np.arange(24, dtype=np.float32)).reshape(1, 3, 8) * np.float32(80), dtype)
        g, u = gate.astype(np.float32), up.astype(np.float32)
        for gate_cap, up_cap in ((4, 25), (2, 7)):
            gc, uc = np.float32(gate_cap), np.float32(up_cap)
            expected = np.tanh(g / gc) * gc * sigmoid(g) * np.tanh(u / uc) * uc
            fixture["activations"].append(dict(dtype=dtype, gate=gate.tolist(), up=up.tolist(),
                                               gate_cap=gate_cap, up_cap=up_cap, expected=cast(expected, dtype).tolist()))
        # Dyadic values make all FP32 products/sums exact, while premature
        # BF16 products still lose precision. Reduction order is immaterial.
        out = cast(((np.arange(64, dtype=np.float32) % 31) - 15).reshape(1, 2, 4, 8) / np.float32(16), dtype)
        weights = np.array([[[.125, .34375, .5625, .96875], [.96875, .125, .34375, .5625]]], dtype=np.float32)
        expected = cast(np.sum(out.astype(np.float32) * weights[..., None], axis=-2, dtype=np.float32), dtype)
        fixture["combinations"].append(dict(dtype=dtype, out=out.tolist(), weights=weights.tolist(), expected=expected.tolist()))
    x = bf16(np.sin(np.arange(48, dtype=np.float32) * np.float32(.8)).reshape(1, 3, 16))
    router = bf16(np.cos(np.arange(112, dtype=np.float32) * np.float32(.37)).reshape(7, 16) * np.float32(.3))
    bias = np.linspace(-.1, .1, 7, dtype=np.float32)
    scores = sigmoid(x @ router.T)
    indices = np.argsort(-(scores + bias), axis=-1)[..., :3]
    selected = np.take_along_axis(scores, indices, axis=-1)
    weights = selected / (np.sum(selected, axis=-1, keepdims=True, dtype=np.float32) + np.float32(1e-20)) * np.float32(8)
    fixture["route"] = dict(x=x.tolist(), router=router.tolist(), bias=bias.tolist(),
                            top_k=3, route_scale=8, indices=indices.tolist(), weights=weights.tolist())
    path = Path(__file__).resolve().parents[1] / "Tests/Inkling2FocusedTests/Resources/InklingCompiledPrimitives.json"
    with path.open("x") as stream:
        json.dump(fixture, stream, indent=2)
        stream.write("\n")
    print(path)


if __name__ == "__main__":
    main()
