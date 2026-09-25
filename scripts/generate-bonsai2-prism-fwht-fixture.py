#!/usr/bin/env python3
"""Generate the deterministic Prism-Hadamard FWHT parity fixture for the
vMLX Bonsai 2 portability seam.

The fixture is computed by the PINNED bundled pack runtime itself: this
script imports ``runtime/runtime.py`` (and reads ``hadamard.json``) from the
immutable revision of ``prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`` and calls the
exact ``fwht(x, block, signs, inverse=False)`` function on deterministic
synthetic activations. The Swift side
(``Tests/MLXLMTests/PrismBonsaiHadamardPinnedParityTests.swift``) replays the
same calls through ``Source/MLXNN/PrismBonsaiHadamard.swift``
``hadamardFWHT(_:block:signs:inverse:)`` and compares against this fixture —
an external-reference parity gate, not a Swift self-reference.

Usage (needs the pinned checkout + an mlx python environment, e.g. the
local-model-bench bonsai2 mlx venv):

    python scripts/generate-bonsai2-prism-fwht-fixture.py \
        --pinned-dir /path/to/Ternary-Bonsai-2-27B-mlx-2bit \
        --out Tests/MLXLMTests/Resources/PrismBonsaiPinnedFWHTFixture.json

Pinned pack: prism-ml/Ternary-Bonsai-2-27B-mlx-2bit @
3f926b415992eaa2ae9dd7b573706494d6bbf787 (runtime/runtime.py `fwht`,
hadamard.json block_size 1024 / explicit signs).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

PIN_REVISION = "3f926b415992eaa2ae9dd7b573706494d6bbf787"
PIN_REPO = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
PIN_RUNTIME = "runtime/runtime.py"
PIN_HADAMARD = "hadamard.json"
PIN_BLOCK_INFO = "pack manifest block_size 1024, explicit ±1 signs"
FIXTURE_SCHEMA = "prism-bonsai-pinned-fwht-fixture"
DETERMINISTIC_SEED = 0xB0A51A2  # fixed; do not change (fixture is frozen)


def load_pinned_runtime(pinned_dir: Path):
    """Import the pinned runtime/runtime.py verbatim as the reference.

    Only ``fwht`` is exercised; the pack's `codec` dependency is stubbed so
    the module imports without the model payload or any install.
    """
    runtime_path = pinned_dir / "runtime" / "runtime.py"
    if not runtime_path.exists():
        sys.exit(f"pinned runtime not found: {runtime_path}")

    codec = types.ModuleType("codec")
    def _transcode(*_args, **_kwargs):  # pragma: no cover - never called
        raise RuntimeError("codec.transcode must not be needed for fwht fixtures")
    codec.transcode = _transcode
    sys.modules.setdefault("codec", codec)

    spec = importlib.util.spec_from_file_location("pinned_runtime", runtime_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_pinned_manifest(pinned_dir: Path):
    hadamard = json.loads((pinned_dir / PIN_HADAMARD).read_text())
    widths = list(hadamard["prism.hadamard.sign_widths"])
    values = list(hadamard["prism.hadamard.sign_values"])
    assert sum(widths) == len(values), "sign widths must pack sign values"
    assert hadamard["prism.hadamard.block_size"] == 1024
    assert hadamard["prism.hadamard.transform"] == "normalized-sylvester-walsh-hadamard"
    assert hadamard["prism.hadamard.sign_mode"] == "explicit"
    slices, offset = [], 0
    for w in widths:
        slices.append(values[offset:offset + w])
        offset += w
    return hadamard, slices


def deterministic_input(width: int, rows: int, index: int) -> list:
    """Fixed-seed synthetic activations, reproducible across machines.

    Values are signed f32 draws from a small LCG, scaled to stay well inside
    the float16 normal range so the f16 staging round-trip is meaningful.
    """
    state = (DETERMINISTIC_SEED + index * 0x9E3779B9) & 0xFFFFFFFF
    out = []
    def rand():
        nonlocal state
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        return ((state >> 8) / float(1 << 24)) - 0.5  # in [-0.5, 0.5)
    for _ in range(rows * width):
        out.append(rand() * 0.5)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinned-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import numpy as np
    import mlx.core as mx

    runtime = load_pinned_runtime(args.pinned_dir)
    fwht = runtime.fwht
    manifest, sign_slices = load_pinned_manifest(args.pinned_dir)

    cases = []
    # Real pack widths (hidden / mlp / vocabulary), each a multiple of 1024,
    # with the REAL per-width sign vectors from the pinned hadamard.json.
    for index, width in enumerate([5120, 6144, 17408]):
        rows = 2 if width == 5120 else 1
        signs = [float(v) for v in sign_slices[index]]
        assert len(signs) == width
        x = mx.array(deterministic_input(width, rows, index), dtype=mx.float16)
        x = x.reshape(rows, width)
        forward = fwht(x, 1024, mx.array(signs), inverse=False)
        inverse = fwht(x, 1024, mx.array(signs), inverse=True)
        mx.eval(forward, inverse)
        case = {
            "name": f"pack-width-{width}",
            "width": width,
            "shape": [rows, width],
            "dtype": "float16",
            "signs": signs,
            "x": [float(v) for v in np.asarray(x, dtype=np.float16).reshape(-1)],
            "forward": [float(v) for v in np.asarray(forward, dtype=np.float16).reshape(-1)],
            "inverse": [float(v) for v in np.asarray(inverse, dtype=np.float16).reshape(-1)],
        }
        if width == 5120:  # round-trip reference for the head case
            rt = fwht(fwht(x, 1024, mx.array(signs), inverse=False), 1024,
                      mx.array(signs), inverse=True)
            mx.eval(rt)
            case["roundtrip"] = [float(v) for v in np.asarray(rt, dtype=np.float16).reshape(-1)]
        cases.append(case)
        print(f"case {case['name']}: forward/inverse computed ({width * rows * 2} values)")

    # Normalization spike: H_1024/32 of all-ones under unit signs is
    # [32, 0, ..., 0] per block (proves scale 1/sqrt(1024)).
    ones = mx.ones((1, 1024), dtype=mx.float16)
    spike = fwht(ones, 1024, mx.ones(1024), inverse=False)
    mx.eval(spike)
    spike_values = [float(v) for v in np.asarray(spike, dtype=np.float16).reshape(-1)]
    cases.append({
        "name": "normalization-spike-1024",
        "width": 1024,
        "shape": [1, 1024],
        "dtype": "float16",
        "signs": [1.0] * 1024,
        "x": [1.0] * 1024,
        "forward": spike_values,
    })
    print("case normalization-spike-1024: computed")

    fixture = {
        "schema": FIXTURE_SCHEMA,
        "pin": {
            "repo": PIN_REPO,
            "revision": PIN_REVISION,
            "runtime_source": PIN_RUNTIME,
            "hadamard_json": PIN_HADAMARD,
            "reference": "fwht(x, block, signs, inverse=False) (verbatim import)",
            "block": int(manifest["prism.hadamard.block_size"]),
            "transform": manifest["prism.hadamard.transform"],
            "sign_mode": manifest["prism.hadamard.sign_mode"],
            "sign_widths": manifest["prism.hadamard.sign_widths"],
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "python_version": sys.version.split()[0],
            "generator": "scripts/generate-bonsai2-prism-fwht-fixture.py",
            "seed": DETERMINISTIC_SEED,
            "note": PIN_BLOCK_INFO,
        },
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()