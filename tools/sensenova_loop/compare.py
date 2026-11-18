"""Compare frozen teacher logits and grade complete native smoke outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file
from transformers import AutoTokenizer


def log_softmax(value):
    shifted = value - value.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def compare(expected, actual, protocol):
    assert set(expected) == set(actual) == {c["name"] for c in protocol["cases"]}
    rows = []
    for name, value in expected.items():
        x, y = value.astype(np.float64), actual[name].astype(np.float64)
        assert x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
        lx, ly = log_softmax(x), log_softmax(y)
        row = {
            "case": name,
            "max_abs": float(abs(x - y).max()),
            "rms": float(np.sqrt(np.mean((x - y) ** 2))),
            "kl_max": float((np.exp(lx) * (lx - ly)).sum(-1).max()),
            "reference_top1": x.argmax(-1).tolist(),
            "native_top1": y.argmax(-1).tolist(),
        }
        row["pass"] = (
            row["max_abs"] <= protocol["logit_abs_gate"]
            and row["rms"] <= protocol["logit_rms_gate"]
            and row["kl_max"] <= protocol["kl_gate"]
            and row["reference_top1"] == row["native_top1"]
        )
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads((args.native / "cases.json").read_text())
    rows = compare(
        load_file(args.reference / "expected.safetensors"),
        load_file(args.native / "selected-logits.safetensors"),
        protocol,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=True
    )
    cases = {case["name"]: case for case in protocol["cases"]}
    results = json.loads((args.native / "results.json").read_text())["rows"]
    for row in results:
        row["text"] = tokenizer.decode(row["content_ids"])
        expected = cases[row["case"]]["expected"]
        try:
            value = (
                json.loads(row["text"]) if isinstance(expected, dict) else row["text"]
            )
        except json.JSONDecodeError:
            value = None
        row["task_pass"] = row["finish"] == "eos" and value == expected
    report = {
        "reference": str(args.reference),
        "native": str(args.native),
        "logits": rows,
        "results": results,
        "fidelity_pass": all(row["pass"] for row in rows),
        "smoke_pass": all(row["task_pass"] for row in results),
    }
    with args.out.open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
