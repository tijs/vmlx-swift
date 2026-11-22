# Inkling 2 Mini native text port provenance

Architecture ported from Apache-2.0 HuggingFace/MLX-VLM Inkling2 reference
https://github.com/Blaizzy/mlx-vlm/tree/392052b4439cf027ab42674d9977f62bb3dfa056/mlx_vlm/models/inkling2
and its Inkling common helpers. Copyright 2026 the HuggingFace Team. All rights reserved.

Mini loop additions follow the published Apache-2.0 conversion patch at
https://huggingface.co/nativ-community/Inkling-2-Mini-nvfp4/blob/26f71eef259a1dd7fc476778d67020351e0dfb8c/runtime/inkling2-mini.patch
SHA256 e504b0e41d86939b1b1753abfb752151c64ceb9911df09eef83f8a1fabb42546.

Source model: thinkingmachines/Inkling-2-Mini@6e7c8248f75265bdd4fbab5273304243b5d8c2cb.
Converted bundle: nativ-community/Inkling-2-Mini-nvfp4@26f71eef259a1dd7fc476778d67020351e0dfb8c.

Licensed under the Apache License, Version 2.0 (the License); you may not use
these adapted sources except in compliance with the License. Obtain a copy at
http://www.apache.org/licenses/LICENSE-2.0 . Unless required by applicable law or
agreed to in writing, software is distributed on an AS IS BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License
for the specific language governing permissions and limitations.

Initial envelope: native single-request text AR, 4K request cap enforced by Mei.
Native packed experts, FP32-sensitive gates and chronological per-visit KV with
four convolution histories. No claim of reference activation quantization,
vision, DSpark, batching, prefix/disk persistence, trimming or compiled decode.

`InklingResidualRounding.safetensors` retains an independent reference golden
for attention-residual mixing at the published model's 2304-channel BF16
boundary. Inputs are the first-pass physical-layer-2 states for the public
prompt “List the integers from 1 to 100, separated by commas:\n”. Expected
mixing and post-normalization values come from the pinned patched MLX-VLM
reference above (MLX 0.32.4, Metal). Original parameter values and dtypes are
preserved; no benchmark task or private application data is included. The
negative control demonstrates that division by 48 changes a BF16 routing
input compared with multiplication by the authored rounded reciprocal.

The opt-in `VMLX_INKLING_FUSED_SHORT_CONV` experiment adapts the reference
`inkling2/kernels.py` virtual-concatenation kernel (Safe math mode) for the
single-request, unmasked text path. Its initial admission matches BF16 inputs,
four taps, at most eight positions and channels divisible by 128. The ordinary
caller continues to own and update the FP32 convolution history. The separate
NumPy `InklingShortConvolution.json` fixture uses ordered FP32 multiply/add,
residual/scaling and explicit nearest-even BF16 rounding for lengths 1, 3 and 8,
including nonzero prior histories; tests also exercise strided kernel inputs.
This experiment remains disabled until its numerical, cache and speed checks
have completed.

`VMLX_INKLING_SORT_EXPERT_PREFILL` is a separate disabled experiment for prefill
blocks longer than eight positions with at least 64 selected routes. It follows
the authored gather-sort / packed GatherQMM / scatter-unsort boundary using the
existing MLXLMCommon permutation helpers. Expert IDs and router scores stay
unchanged, and original route order is restored before FP32 route weighting.
Actual NVFP4 and MXFP8 matrix-dispatch parity and end-to-end latency must be
qualified before enabling this option.
