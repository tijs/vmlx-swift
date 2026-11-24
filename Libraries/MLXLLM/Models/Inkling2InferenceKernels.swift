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

    private static let depthMixEnabled = ProcessInfo.processInfo.environment["VMLX_INKLING_DEPTH_MIX"] == "1"

    // One group per depth state. All learned queries share its squared norm
    // and the same state reads. The number of queries is a small model constant.
    private static let depthScore = MLXFast.metalKernel(
        name: "mei_inkling_depth_scores",
        inputNames: ["states", "queries"], outputNames: ["scores", "norms"],
        source: """
            uint lane = thread_position_in_threadgroup.x;
            uint row = threadgroup_position_in_grid.x;
            float sums[Q + 1] = {};
            for (uint i = lane; i < H; i += 256) {
                float x = float(states[row * H + i]);
                sums[Q] += x * x;
                for (uint q = 0; q < Q; q++) {
                    sums[q] += x * float(queries[q * H + i]);
                }
            }
            threadgroup float scratch[(Q + 1) * 8];
            for (uint q = 0; q <= Q; q++) {
                float sum = simd_sum(sums[q]);
                if (lane % 32 == 0) scratch[q * 8 + lane / 32] = sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < 32) {
                float norm = simd_sum(lane < 8 ? scratch[Q * 8 + lane] : 0.0f);
                float inverse = rsqrt(norm / float(H) + 1e-6f);
                if (lane == 0) norms[row] = inverse;
                for (uint q = 0; q < Q; q++) {
                    float value = simd_sum(lane < 8 ? scratch[q * 8 + lane] : 0.0f);
                    if (lane == 0) scores[row * Q + q] = value * inverse * rsqrt(float(H));
                }
            }
            """)

    private static let depthMerge = MLXFast.metalKernel(
        name: "mei_inkling_depth_merge",
        inputNames: ["states", "scores", "norms"], outputNames: ["out"],
        source: """
            uint x = thread_position_in_grid.x;
            uint q = threadgroup_position_in_grid.y;
            uint lane = thread_position_in_threadgroup.x;
            threadgroup float factors[32];
            if (lane < 32) {
                float score = lane < S ? scores[lane * Q + q] : -INFINITY;
                float maximum = simd_max(score);
                float weight = exp(score - maximum);
                float total = simd_sum(weight);
                if (lane < S) factors[lane] = weight / total * (NORMALIZED ? norms[lane] : 1.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float sum = 0;
            for (uint i = 0; i < S; i++) sum += float(states[i * H + x]) * factors[i];
            out[q * H + x] = sum;
            """)

    static func depthMix(_ states: MLXArray, _ query: MLXArray, normalized: Bool) -> MLXArray? {
        guard depthMixEnabled, Device.defaultDevice().deviceType == .gpu,
              states.ndim == 4, states.dim(0) == 1, states.dim(1) == 1,
              (1...32).contains(states.dim(2)), states.dim(3) == 2304 else { return nil }
        let width = states.dim(3)
        let depth = states.dim(2)
        let queries = query.size / width
        guard [1, 3, 4].contains(queries), query.size == width * queries else { return nil }
        let partial = depthScore(
            [states, query], template: [("H", width), ("Q", queries)],
            grid: (256 * depth, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[depth, queries], [depth]], outputDTypes: [.float32, .float32])
        return depthMerge(
            [states, partial[0], partial[1]],
            template: [("H", width), ("Q", queries), ("S", depth), ("NORMALIZED", normalized ? 1 : 0)],
            grid: (width, queries, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, 1, queries, width]], outputDTypes: [.float32])[0]
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
