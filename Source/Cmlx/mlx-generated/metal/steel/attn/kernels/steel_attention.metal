// Copyright © 2024-25 Apple Inc.

// clang-format off
#include "../../../utils.h"

#include "../../../steel/attn/kernels/steel_attention.h"

#define instantiate_attn(tname, dtype, bq, bk, bd, wm, wn, mname, mtype) \
  instantiate_kernel(                                                    \
      "steel_attention_" #tname "_bq" #bq "_bk" #bk "_bd" #bd            \
      "_wm" #wm "_wn" #wn "_mask" #mname,                                \
  attention, dtype, bq, bk, bd, wm, wn, mtype, float)

#define instantiate_attn_shapes_helper(iname, itype, mname, mtype)  \
    instantiate_attn(iname, itype, 32, 16, 256, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 16, 128, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  80, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  64, 4, 1, mname, mtype)

#define instantiate_attn_mask_helper(iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, bool_, bool)

instantiate_attn_mask_helper(float16, half);
instantiate_attn_mask_helper(bfloat16, bfloat16_t);

instantiate_attn_mask_helper(float32, float);

// float32 bd=256 experiment remedy tile (bq=16/bk=8/wm=2/wn=1), dispatched
// only by the regular full arm under VMLX_BONSAI2_SDPA_FULL_HD256=1
// (see mlx/backend/metal/scaled_dot_product_attention.h). The historical
// bq32/bk16/bd256/wm4/wn1 tile needs 53,760 B threadgroup memory versus the
// applegpu_g13s 32,768 B maximum; this derived candidate is 28,928 B. Only
// float32 needs instantiations: the remedy never fires for other dtypes.
instantiate_attn(float32, float, 16, 8, 256, 2, 1, float32, float)
instantiate_attn(float32, float, 16, 8, 256, 2, 1, bool_, bool)
// clang-format on
