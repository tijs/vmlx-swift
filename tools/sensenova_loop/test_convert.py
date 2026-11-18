"""Converter contract tests use only the authored tiny synthetic fixture."""

import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("sensenova_convert", HERE / "convert.py")
convert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert)


class ConversionContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            prefix="sensenova-synthetic-conversion-"
        )
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        fixture = Path(os.environ["SENSENOVA_FIXTURE_DIR"])
        self.config = json.loads((fixture / "config.json").read_text())
        self.weights = mx.load(str(fixture / "weights.safetensors"))
        (self.source / "config.json").write_text(json.dumps(self.config))
        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
        ]:
            (self.source / name).write_text("{}")
        (self.source / "chat_template.jinja").write_text("synthetic fixture only")

    def run_conversion(self, *, bits=None, mutate_index=None, wrong_digest=False):
        mx.save_safetensors(str(self.source / "model.safetensors"), self.weights)
        index = {"weight_map": dict.fromkeys(self.weights, "model.safetensors")}
        if mutate_index:
            mutate_index(index["weight_map"])
        (self.source / "model.safetensors.index.json").write_text(json.dumps(index))
        with (self.source / "model.safetensors").open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        manifest = {
            "sha": convert.REVISION,
            "siblings": [
                {
                    "rfilename": "model.safetensors",
                    "size": (self.source / "model.safetensors").stat().st_size,
                    "lfs": {"sha256": "wrong" if wrong_digest else digest},
                }
            ],
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        args = [
            sys.executable,
            str(HERE / "convert.py"),
            "--source",
            str(self.source),
            "--out",
            str(self.root / "output"),
            "--hub-manifest",
            str(self.root / "manifest.json"),
        ]
        if bits:
            args += ["--bits", str(bits)]
        return subprocess.run(
            args, check=False, capture_output=True, text=True, timeout=60
        )

    def test_positive_mapping_preserves_router_precision_and_norm_math(self):
        result = self.run_conversion()
        self.assertEqual(result.returncode, 0, result.stderr)
        output = mx.load(str(self.root / "output/model-00001.safetensors"))
        key = "model.layers.4.mhc_attn.weight"
        self.assertEqual(output[key].dtype, mx.float32)
        self.assertTrue(
            bool(
                mx.all(
                    output[key]
                    == self.weights[key.replace("model.", "model.language_model.", 1)]
                ).item()
            )
        )
        norm = "model.layers.0.input_layernorm.weight"
        expected = (
            self.weights[norm.replace("model.", "model.language_model.", 1)].astype(
                mx.float32
            )
            + 1
        )
        self.assertEqual(output[norm].dtype, mx.float32)
        self.assertTrue(bool(mx.all(output[norm] == expected).item()))
        config = json.loads((self.root / "output/config.json").read_text())
        self.assertEqual(config["norm_convention"], "mlx_plus_one")
        self.assertNotIn("vision_config", config)
        with (self.root / "output/model-00001.safetensors").open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(length))
        self.assertEqual((8 + length) % 8, 0)
        for name, value in header.items():
            if name != "__metadata__":
                self.assertEqual((8 + length + value["data_offsets"][0]) % 4, 0)

    def test_quantization_never_quantizes_routers(self):
        result = self.run_conversion(bits=8)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = mx.load(str(self.root / "output/model-00001.safetensors"))
        self.assertIn("lm_head.scales", output)
        self.assertEqual(output["model.layers.4.mhc_attn.weight"].dtype, mx.float32)
        self.assertNotIn("model.layers.4.mhc_attn.scales", output)

    def test_wrong_source_digest_rejects(self):
        self.assertNotEqual(self.run_conversion(wrong_digest=True).returncode, 0)

    def test_missing_index_tensor_rejects(self):
        self.assertNotEqual(
            self.run_conversion(
                mutate_index=lambda d: d.pop("lm_head.weight")
            ).returncode,
            0,
        )

    def test_unknown_text_key_rejects(self):
        self.weights["model.language_model.unrecognized.weight"] = mx.ones([1])
        self.assertNotEqual(self.run_conversion().returncode, 0)

    def test_wrong_shape_rejects(self):
        self.weights["model.language_model.norm.weight"] = mx.ones([63])
        self.assertNotEqual(self.run_conversion().returncode, 0)

    def test_wrong_router_dtype_rejects(self):
        key = "model.language_model.layers.4.mhc_attn.weight"
        self.weights[key] = self.weights[key].astype(mx.bfloat16)
        self.assertNotEqual(self.run_conversion().returncode, 0)

    def test_nonfinite_source_rejects(self):
        self.weights["model.language_model.norm.weight"] = mx.full([64], float("nan"))
        self.assertNotEqual(self.run_conversion().returncode, 0)

    def test_deliberately_unaligned_header_keeps_exact_payload(self):
        metadata = {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
        header = json.dumps(metadata).encode()
        while (8 + len(header)) % 8 != 3:
            header += b" "
        payload = struct.pack("<ff", 1.25, -2.5)
        source = self.root / "unaligned.safetensors"
        original = struct.pack("<Q", len(header)) + header + payload
        source.write_bytes(original)
        target = self.root / "aligned.safetensors"
        receipt = convert.align_shard(source, target)
        with target.open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            self.assertEqual((8 + length) % 8, 0)
            self.assertEqual(json.loads(handle.read(length)), metadata)
            self.assertEqual(handle.read(), payload)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(receipt["payload_sha256"], hashlib.sha256(payload).hexdigest())

    def test_alignment_refuses_existing_destination(self):
        source = self.root / "source.safetensors"
        mx.save_safetensors(str(source), {"w": mx.ones([4])})
        target = self.root / "exists.safetensors"
        target.write_bytes(b"historical evidence")
        with self.assertRaises(FileExistsError):
            convert.align_shard(source, target)
        self.assertEqual(target.read_bytes(), b"historical evidence")


if __name__ == "__main__":
    unittest.main()
