// Copyright © 2026. Apache-2.0.
// Adapted from the pinned MLX-VLM inkling2/kernels.py; see INKLING2_SOURCE.md.
import Foundation
import MLX
import MLXFast

/// Small stateless inference kernels. Visit-local state remains owned by the
/// caller; none of these kernels retains or mutates a cache object.
enum Inkling2InferenceKernels {
    static let sortedExpertPrefillEnabled =
        ProcessInfo.processInfo.environment["VMLX_INKLING_SORT_EXPERT_PREFILL"] == "1"

    static let fusedShortConvolutionEnabled =
        ProcessInfo.processInfo.environment["VMLX_INKLING_FUSED_SHORT_CONV"] == "1"

    static let fusedMaskEnabled =
        ProcessInfo.processInfo.environment["VMLX_INKLING_FUSED_MASK"] == "1"

    // Keep the projection matmul intact. Map each query/key pair to its
    // trained bias in one dispatch without materializing an index tensor.
    private static let maskKernel = MLXFast.metalKernel(
        name: "vmlx_inkling2_banded_mask",
        inputNames: ["projected", "parameters"], outputNames: ["out"],
        source: """
            uint i = thread_position_in_grid.x;
            int count = parameters[0], offset = parameters[1], window = parameters[2];
            int n = projected_shape[2], heads = projected_shape[1];
            int extent = projected_shape[3];
            uint row = i / count;
            uint batchHead = row / n;
            if (batchHead >= uint(projected_shape[0] * heads)) return;
            int q = row % n;
            int distance = offset + q - int(i % count);
            if (distance < 0 || (window > 0 && distance >= window)) {
                out[i] = T(-1e30f);
            } else if (distance >= extent) {
                out[i] = T(0);
            } else {
                out[i] = projected[(batchHead / heads) * projected_strides[0]
                    + (batchHead % heads) * projected_strides[1]
                    + q * projected_strides[2] + distance * projected_strides[3]];
            }
            """, ensureRowContiguous: false)

    static func bandedMask(
        _ projected: MLXArray, count: Int, offset: Int, window: Int, extent: Int
    ) -> MLXArray? {
        // The shader uses 32-bit indices for the grid and the signed relative distance.
        guard projected.ndim == 4, extent > 0, projected.dim(3) == extent,
              count > 0, count <= Int(Int32.max),
              (0...Int(Int32.max)).contains(window), offset >= 0,
              offset <= Int(Int32.max) - projected.dim(2) else { return nil }
        let rows = projected.size / extent
        guard rows > 0, rows <= Int(Int32.max) / count else { return nil }
        let shape = Array(projected.shape.prefix(3)) + [count]
        return maskKernel(
            [projected, MLXArray([Int32(count), Int32(offset), Int32(window)])],
            template: [("T", projected.dtype)], grid: (rows * count, 1, 1),
            threadGroup: (256, 1, 1), outputShapes: [shape], outputDTypes: [projected.dtype])[0]
    }

    // Cmlx's custom-kernel CompileOptions default to MathMode::Safe, matching
    // the authored kernel's explicit math_mode=safe. Preserve tap order and
    // FP32 intermediates before the final cast to the input dtype.
    private static let shortConvolutionKernel = MLXFast.metalKernel(
        name: "vmlx_inkling2_virtual_short_convolution",
        inputNames: ["state", "weight", "x", "divisor"],
        outputNames: ["out"],
        source: """
            uint i = thread_position_in_grid.x;
            int B = x_shape[0], L = x_shape[1], C = x_shape[2];
            if (i >= B * L * C) return;
            int c = i % C, t = (i / C) % L, b = i / (C * L);
            float value = 0;
            for (int j = 0; j < W; ++j) {
                int p = t + j - W + 1;
                float xv = p < 0
                    ? float(state[b * state_strides[0] + (p + W - 1) * state_strides[1]
                                  + c * state_strides[2]])
                    : float(x[b * x_strides[0] + p * x_strides[1] + c * x_strides[2]]);
                float w = weight[c * weight_strides[0] + j * weight_strides[1]];
                value = value + xv * w;
            }
            out[i] = T((value + float(x[b * x_strides[0] + t * x_strides[1]
                                       + c * x_strides[2]])) / divisor);
            """,
        ensureRowContiguous: false)

    static func virtualShortConvolution(
        state: MLXArray, weight: MLXArray, input: MLXArray, width: Int
    ) -> MLXArray {
        shortConvolutionKernel(
            [state, weight, input, MLXArray(Float(sqrt(Double(width + 1))))],
            template: [("T", input.dtype), ("W", width)],
            grid: (input.size, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [input.shape], outputDTypes: [input.dtype])[0]
    }
}
