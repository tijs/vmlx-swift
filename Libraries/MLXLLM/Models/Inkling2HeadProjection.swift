// Copyright © 2026. Apache-2.0.
// Reduction layout adapted from MLX's Apache-2.0 GEMVKernel in
// Source/Cmlx/mlx/mlx/backend/metal/kernels/gemv.h (BM8/BN1/SM1/SN32/TM4/TN4).
import Foundation
import MLX
import MLXFast

/// Experimental mixed-storage GEMV. Only the checkpoint storage is BF16;
/// products, accumulators, reduction and output remain FP32. The caller must
/// supply the model's BF16-rounded final hidden vector, promoted to FP32.
enum Inkling2HeadProjection {
    static let enabled = ProcessInfo.processInfo.environment["VMLX_INKLING_MIXED_HEAD"] == "1"

    static func supports(_ input: MLXArray, weight: MLXArray) -> Bool {
        Device.defaultDevice().deviceType == .gpu && input.dtype == .float32
            && weight.dtype == .bfloat16 && input.ndim == 3
            && input.dim(0) == 1 && input.dim(1) == 1 && weight.ndim == 2
            && input.dim(2) == weight.dim(1) && weight.dim(1) >= 128
            && weight.dim(1) % 128 == 0
            && weight.dim(0) >= 4096 && weight.dim(0) % 32 == 0
    }

    private static let kernel = MLXFast.metalKernel(
        name: "vmlx_inkling2_bf16_storage_fp32_head",
        inputNames: ["x", "weight"], outputNames: ["out"],
        source: """
            uint linear = thread_position_in_grid.x;
            uint lane = linear % 32;
            uint row = (linear / 32) * 4;
            float result[4] = {0, 0, 0, 0};
            for (uint base = lane * 4; base < K; base += 128) {
                float coefficients[4];
                #pragma unroll
                for (uint j = 0; j < 4; ++j) coefficients[j] = x[base + j];
                #pragma unroll
                for (uint r = 0; r < 4; ++r) {
                    #pragma unroll
                    for (uint j = 0; j < 4; ++j) {
                        result[r] += float(weight[(row + r) * K + base + j]) * coefficients[j];
                    }
                }
            }
            #pragma unroll
            for (uint r = 0; r < 4; ++r) {
                #pragma unroll
                for (ushort shift = 16; shift >= 1; shift >>= 1) {
                    result[r] += simd_shuffle_down(result[r], shift);
                }
                if (lane == 0) out[row + r] = result[r];
            }
            """)

    static func project(_ input: MLXArray, weight: MLXArray) -> MLXArray {
        precondition(supports(input, weight: weight))
        return kernel([input, weight], template: [("K", weight.dim(1))],
            grid: (weight.dim(0) / 4 * 32, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, 1, weight.dim(0)]], outputDTypes: [.float32])[0]
    }
}
