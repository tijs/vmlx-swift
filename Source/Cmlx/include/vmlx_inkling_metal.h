#pragma once

#include <stdbool.h>
#include "mlx/c/array.h"
#include "mlx/c/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Borrows a cached Metal pipeline from the active MLX device.
void* vmlx_inkling_paged_prefill_pipeline(bool align_query, bool align_keys);

/// Reuses the resident Metal vector reduction on paged first-pass state.
int vmlx_inkling_paged_vector_merge(
    mlx_array* result, mlx_array partials, mlx_array sums, mlx_array maxima,
    mlx_stream stream);

#ifdef __cplusplus
}
#endif
