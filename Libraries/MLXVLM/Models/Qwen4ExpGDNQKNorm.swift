// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// AR-only Q/K RMS normalization and scaling. The reduction follows MLX's
/// rms_single_row (four contiguous reads per thread, then two SIMD reductions).
/// The normalized value and each scalar MUST round to the activation dtype
/// before multiplication. This is not an algebraic rewrite of normalization.
///
/// Qualification currently covers half/bfloat16 and power-of-two head widths.
/// FP32/non-power-of-two diagnostics differed from the reference and are NOT
/// admitted. All unsupported shapes/types keep the original graph.
final class Qwen4ExpGDNQKNorm {
    let heads: Int
    let headDimension: Int
    private let scales: MLXArray
    private let width: MLXArray

    init(heads: Int, headDimension: Int) {
        self.heads = heads
        self.headDimension = headDimension
        let inverseScale = pow(Float(headDimension), -0.5)
        scales = MLXArray([pow(inverseScale, 2), inverseScale])
        width = MLXArray([UInt32(max(0, headDimension))])
    }

    static func eligible(shape: [Int], dtype: DType, heads: Int, headDimension: Int) -> Bool {
        shape.count == 3 && shape[0] == 1 && shape[1] == 1
            && heads > 0 && headDimension >= 32 && headDimension <= 4096
            && headDimension.nonzeroBitCount == 1
            && shape[2] >= 2 * heads * headDimension
            && (dtype == .bfloat16 || dtype == .float16)
    }

    #if canImport(Metal)
    private static let kernel = MLXFast.metalKernel(
        name: "vmlx_qwen4_gdn_qk_norm_scaled",
        inputNames: ["input", "scales", "width"], outputNames: ["output"],
        source: """
            uint row = threadgroup_position_in_grid.x;
            uint lid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            threadgroup float local_sums[32];
            threadgroup float local_inv_mean[1];
            float acc = 0.0f;
            uint base = row * HEAD_DIM + lid * 4;
            for (uint i = 0; i < 4; ++i) {
                if (lid * 4 + i < HEAD_DIM) {
                    float xi = float(input[base + i]);
                    acc += xi * xi;
                }
            }
            acc = simd_sum(acc);
            if (simd_group == 0) local_sums[lane] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) local_sums[simd_group] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                acc = simd_sum(local_sums[lane]);
                if (lane == 0) local_inv_mean[0] = metal::precise::rsqrt(acc / width[0] + 1e-6f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
            #pragma clang fp contract(off)
            #pragma clang fp reassociate(off)
            T scale = T(scales[row < HEADS ? 0 : 1]);
            for (uint i = 0; i < 4; ++i) {
                if (lid * 4 + i < HEAD_DIM) {
                    T normed = T(float(input[base + i]) * local_inv_mean[0]);
                    output[base + i] = T(float(normed) * float(scale));
                }
            }
            }
            """)
    #endif

    func callAsFunction(_ convolved: MLXArray) -> MLXArray? {
        #if canImport(Metal)
        guard Self.eligible(shape: convolved.shape, dtype: convolved.dtype,
                            heads: heads, headDimension: headDimension),
            !CompiledDecodeTrace.isActive, Device.defaultDevice().deviceType == .gpu
        else { return nil }
        let threads = ((headDimension + 127) / 128) * 32
        return Self.kernel(
            [convolved, scales, width],
            template: [("T", convolved.dtype), ("HEADS", heads), ("HEAD_DIM", headDimension)],
            grid: (2 * heads * threads, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[1, 1, 2 * heads, headDimension]],
            outputDTypes: [convolved.dtype])[0]
        #else
        return nil
        #endif
    }
}
