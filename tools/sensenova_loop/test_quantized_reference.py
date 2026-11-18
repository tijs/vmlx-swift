import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from quantized_reference import dequantize_affine, install
from safetensors.torch import save_file


class QuantizedReferenceTests(unittest.TestCase):
    def test_fp32_reference_swaps_one_module_and_restores_original_parameters(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(3, 64).bfloat16()
                self.head = torch.nn.Linear(64, 3, bias=False).bfloat16()

            def forward(self, ids):
                return self.head(self.embed(ids))

        model = Toy()
        original = {name: param for name, param in model.named_parameters()}
        packed = torch.full((3, 16), 0x03020100, dtype=torch.uint32)
        scales = torch.full((3, 1), 0.003, dtype=torch.bfloat16)
        biases = torch.full((3, 1), -0.13, dtype=torch.bfloat16)
        tensors = {
            f"{name}.{kind}": value
            for name in ("embed", "head")
            for kind, value in (
                ("weight", packed),
                ("scales", scales),
                ("biases", biases),
            )
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            save_file(
                {key: value.clone() for key, value in tensors.items()},
                path / "weights.safetensors",
            )
            (path / "config.json").write_text(
                json.dumps({"quantization": {"bits": 8, "group_size": 64}})
            )
            (path / "model.safetensors.index.json").write_text(
                json.dumps(
                    {"weight_map": dict.fromkeys(tensors, "weights.safetensors")}
                )
            )
            self.assertEqual(install(model, path, float32_activation=True), 2)
            weight = dequantize_affine(packed, scales, biases, dtype=torch.float32)
            ids = torch.tensor([0, 2])
            expected = weight[ids] @ weight.T
            with torch.inference_mode():
                actual = model(ids)
                again = model(ids)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(again, expected, rtol=0, atol=0)
            self.assertEqual(actual.dtype, torch.float32)
            for name, param in model.named_parameters():
                self.assertIs(param, original[name])

    def test_packing_groups_and_rounding_against_scalar_oracle(self):
        values = np.arange(256, dtype=np.uint32).reshape(2, 128)
        words = sum(values[:, shift::4] << (8 * shift) for shift in range(4))
        scales = torch.tensor([[0.125, 0.25], [0.5, 1]], dtype=torch.bfloat16)
        biases = torch.tensor([[-1, 2], [3, -4]], dtype=torch.bfloat16)
        result = dequantize_affine(torch.from_numpy(words), scales, biases, rows=1)
        expected = torch.tensor(
            [
                [
                    float(scales[row, col // 64]) * int(values[row, col])
                    + float(biases[row, col // 64])
                    for col in range(128)
                ]
                for row in range(2)
            ],
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_rejects_incompatible_group_geometry(self):
        with self.assertRaises(AssertionError):
            dequantize_affine(
                torch.zeros((2, 16), dtype=torch.uint32),
                torch.ones((2, 2), dtype=torch.bfloat16),
                torch.zeros((2, 2), dtype=torch.bfloat16),
            )

    def test_fp32_affine_weights_do_not_round_to_bf16_early(self):
        packed = torch.tensor([[0x03020100] * 16], dtype=torch.uint32)
        scale = torch.tensor([[0.003]], dtype=torch.bfloat16)
        bias = torch.tensor([[-0.13]], dtype=torch.bfloat16)
        precise = dequantize_affine(packed, scale, bias, dtype=torch.float32)
        rounded = dequantize_affine(packed, scale, bias)
        expected = torch.tensor(
            [[float(scale) * value + float(bias) for value in [0, 1, 2, 3] * 16]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(precise, expected, rtol=0, atol=0)
        torch.testing.assert_close(rounded, expected.bfloat16(), rtol=0, atol=0)
        self.assertGreater((precise - rounded.float()).abs().max().item(), 0)

    def test_six_bit_codes_cross_byte_and_word_boundaries(self):
        values = list(range(64)) + list(reversed(range(64)))
        scalar_packed = sum(value << (6 * index) for index, value in enumerate(values))
        words = np.frombuffer(scalar_packed.to_bytes(96, "little"), dtype="<u4").copy()
        scales = torch.tensor([[0.5, 0.25]], dtype=torch.bfloat16)
        biases = torch.tensor([[-2, 10]], dtype=torch.bfloat16)
        actual = dequantize_affine(
            torch.from_numpy(words.reshape(1, -1)),
            scales,
            biases,
            bits=6,
            dtype=torch.float32,
        )
        expected = torch.tensor(
            [
                [
                    v * float(scales[0, i // 64]) + float(biases[0, i // 64])
                    for i, v in enumerate(values)
                ]
            ]
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_four_bit_codes_use_both_nibbles(self):
        packed = torch.tensor([[0x76543210, 0xFEDCBA98] * 4], dtype=torch.uint32)
        scale = torch.ones((1, 1), dtype=torch.bfloat16)
        bias = torch.zeros((1, 1), dtype=torch.bfloat16)
        actual = dequantize_affine(packed, scale, bias, bits=4, dtype=torch.float32)
        torch.testing.assert_close(actual, torch.arange(16).repeat(4).float()[None, :])


if __name__ == "__main__":
    unittest.main()
