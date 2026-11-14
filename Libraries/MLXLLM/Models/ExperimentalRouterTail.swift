// SPDX-License-Identifier: Apache-2.0
// Adapted from raullenchai/Rapid-MLX qwen35_moe_router.py:
// https://github.com/raullenchai/Rapid-MLX/blob/7d2b20c7f7fe4707f0690d3dd720be0b7b4f7725/rapid_mlx/qwen35_moe_router.py
// The router projection and precise softmax remain unchanged. This experiment
// preserves the stock selected-expert order and input-dtype reduction.
import Foundation
import MLX
import MLXFast

enum ExperimentalRouterTail {
    static let enabled = ProcessInfo.processInfo.environment["MEI_EXPERIMENTAL_ROUTER_TAIL"] == "1"
    static func supports(_ probabilities: MLXArray, topK: Int) -> Bool {
        guard Device.defaultDevice().deviceType == .gpu, probabilities.ndim >= 2,
            probabilities.dtype == .bfloat16 || probabilities.dtype == .float16
        else { return false }
        let experts = probabilities.dim(-1)
        guard experts >= 32, experts <= 512, experts % 32 == 0,
            topK >= 1, topK <= min(16, experts)
        else { return false }
        let rows = probabilities.size / experts
        return rows > 0 && rows <= 8
    }

    static let kernel = MLXFast.metalKernel(
        name: "mei_experimental_router_topk",
        inputNames: ["probs"], outputNames: ["indices", "scores"],
        source: """
        constexpr uint PER = uint(NE) / 32;
        const uint lane = thread_position_in_threadgroup.x;
        const uint row = threadgroup_position_in_grid.y;
        const device T* g = probs + row * uint(NE);
        float vals[PER];
        bool taken[PER];
        for (uint i = 0; i < PER; ++i) {
            vals[i] = float(g[lane * PER + i]);
            taken[i] = false;
        }
        float sel_p[K];
        uint sel_i[K];
        for (uint j = 0; j < K; ++j) {
            float best = -INFINITY;
            uint best_i = uint(NE);
            bool best_nan = false;
            for (uint i = 0; i < PER; ++i) {
                // MLX sorts NaNs after finite values, with stable index ties.
                // Never allow an exhausted lane's sentinel to win a reduction.
                const bool nan = isnan(vals[i]);
                if (!taken[i] && (nan || (!best_nan && vals[i] >= best))) {
                    best = vals[i];
                    best_i = lane * PER + i;
                    best_nan = nan;
                }
            }
            const bool group_nan = simd_any(best_nan);
            const float group_best = simd_max(best_nan ? -INFINITY : best);
            const bool selected = best_i < uint(NE) && (group_nan ? best_nan : best == group_best);
            const uint candidate = selected ? best_i : 0u;
            const uint group_best_i = simd_max(candidate);
            if (selected && best_i == group_best_i) {
                taken[best_i - lane * PER] = true;
            }
            sel_p[j] = group_nan ? NAN : group_best;
            sel_i[j] = group_best_i;
        }
        if (lane == 0) {
            T total = T(0);
            for (int j = int(K) - 1; j >= 0; --j) { total += T(sel_p[uint(j)]); }
            for (uint j = 0; j < K; ++j) {
                const uint out = uint(K) - 1u - j;
                indices[row * uint(K) + out] = sel_i[j];
                scores[row * uint(K) + out] = NORM ? T(T(sel_p[j]) / total) : T(sel_p[j]);
            }
        }
        """)

    static func call(_ probabilities: MLXArray, topK: Int, normalize: Bool) -> (MLXArray, MLXArray)? {
        guard supports(probabilities, topK: topK) else { return nil }
        let experts = probabilities.dim(-1)
        let shape = Array(probabilities.shape.dropLast()) + [topK]
        let result = kernel(
            [probabilities],
            template: [("T", probabilities.dtype), ("NE", experts), ("K", topK), ("NORM", normalize)],
            grid: (32, probabilities.size / experts, 1), threadGroup: (32, 1, 1),
            outputShapes: [shape, shape], outputDTypes: [.uint32, probabilities.dtype])
        return (result[0], result[1])
    }
}
