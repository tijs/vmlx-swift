// SPDX-License-Identifier: Apache-2.0 AND MIT
// Prototype port from Rapid-MLX@7d2b20c7f7fe4707f0690d3dd720be0b7b4f7725,
// rapid_mlx/gdn_prefill.py; its design credits the Apache-2.0 oMLX project.
// Stock-order recurrence follows vmlx-swift e280bd2b GatedDelta.swift (MIT).
// Full notices: LICENSE-RAPID-MLX and LICENSE-VMLX-MIT at the package root.
// Only the stock-order row16 variant is retained. Final qualification pending.
import Foundation
import MLX
import MLXFast

enum ExperimentalBlockedGDN {
    // Experimental layout selection is frozen once per owned process.
    static let stockRows = 16
    static let forceStrict = ProcessInfo.processInfo.environment["VMLX_GDN_STRICT"] == "1"

    // Measured dispatch policy is separate from the kernel's numerical
    // eligibility. Small chunks retain the existing stock implementation.
    static func minimumServingTokens(for dtype: DType) -> Int {
        if dtype == .bfloat16 { return 4096 }
        if dtype == .float32 { return 1024 }
        return Int.max
    }

    static let enabled = ProcessInfo.processInfo.environment["MEI_EXPERIMENTAL_BLOCKED_GDN"] == "1"
    static let verify = ProcessInfo.processInfo.environment["MEI_EXPERIMENTAL_BLOCKED_GDN_VERIFY"] == "1"

    static let stockOrderSource = #"""
    constexpr int TB = 32;
    constexpr int DB = ROWS;
    constexpr int NT = 32 * DB;
    const int tid = thread_position_in_threadgroup.x;
    const int dv = tid / 32;
    const int lane = tid % 32;
    const int hv = threadgroup_position_in_grid.y;
    const int hk = hv / (Hv / Hk);
    const int dv0 = threadgroup_position_in_grid.x * DB;
    threadgroup InT qs[TB][Dk];
    threadgroup InT ks[TB][Dk];
    threadgroup InT vs[TB][DB];
    threadgroup float gs[TB];
    threadgroup float bs[TB];
    float st[4];
    for (int i = 0; i < 4; ++i)
      st[i] = state_in[((size_t)hv * Dv + dv0 + dv) * Dk + 4 * lane + i];
    for (int t0 = 0; t0 < T; t0 += TB) {
      const int tt = min(TB, T - t0);
      for (int p = tid; p < tt * Dk; p += NT) {
        const int t = p / Dk, d = p % Dk;
        qs[t][d] = q[((size_t)(t0 + t) * Hk + hk) * Dk + d];
        ks[t][d] = k[((size_t)(t0 + t) * Hk + hk) * Dk + d];
      }
      for (int p = tid; p < tt * DB; p += NT) {
        const int t = p / DB, d = p % DB;
        vs[t][d] = v[((size_t)(t0 + t) * Hv + hv) * Dv + dv0 + d];
      }
      for (int t = tid; t < tt; t += NT) {
        gs[t] = g[(size_t)(t0 + t) * Hv + hv];
        bs[t] = beta[(size_t)(t0 + t) * Hv + hv];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int t = 0; t < tt; ++t) {
        float kv = 0.0f;
        for (int i = 0; i < 4; ++i) {
          st[i] = st[i] * gs[t];
          kv += st[i] * ks[t][4 * lane + i];
        }
        kv = simd_sum(kv);
        float delta = (vs[t][dv] - kv) * bs[t];
        float out = 0.0f;
        for (int i = 0; i < 4; ++i) {
          st[i] = st[i] + ks[t][4 * lane + i] * delta;
          out += st[i] * qs[t][4 * lane + i];
        }
        out = simd_sum(out);
        if (lane == 0)
          y[((size_t)(t0 + t) * Hv + hv) * Dv + dv0 + dv] = static_cast<InT>(out);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < 4; ++i)
      state_out[((size_t)hv * Dv + dv0 + dv) * Dk + 4 * lane + i] = st[i];
"""#
    static let stockHalfKernel = makeKernel(block: 32, stockOrder: true)
    static let stockFloatKernel = makeKernel(block: 16, stockOrder: true)

    private static func makeKernel(block: Int, stockOrder: Bool) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
            name: "mei_experimental_gdn_blocked_tb\(block)_stock\(stockOrder)",
            inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
            outputNames: ["y", "state_out"],
            source: stockOrderSource.replacingOccurrences(of: "constexpr int TB = 32;", with: "constexpr int TB = \(block);"))
    }

    static func call(q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray,
                     beta: MLXArray, state: MLXArray?, masked: Bool = false,
                     roundStateEachStep: Bool = false, preserveReductionOrder: Bool) -> (MLXArray, MLXArray)? {
        guard Device.defaultDevice().deviceType == .gpu, !masked, !roundStateEachStep,
            preserveReductionOrder,
            !forceStrict,
            q.ndim == 4, q.shape == k.shape, q.dtype == k.dtype,
            [.bfloat16, .float32].contains(q.dtype), q.dim(0) == 1,
            q.dim(1) >= 64, q.dim(2) == 16, q.dim(3) == 128,
            v.shape == [1, q.dim(1), 32, 128], v.dtype == q.dtype,
            g.shape == [1, q.dim(1), 32], beta.shape == g.shape,
            g.dtype == .float32, beta.dtype == .float32
        else { return nil }
        let initial: MLXArray
        if let state {
            guard state.shape == [1, 32, 128, 128], state.dtype == .float32 else { return nil }
            initial = state
        } else { initial = MLXArray.zeros([1, 32, 128, 128], dtype: .float32) }
        let kernel = q.dtype == .float32 ? stockFloatKernel : stockHalfKernel
        let result = kernel(
            [q, k, v, g.asType(.float32), beta.asType(.float32), initial, MLXArray(Int32(q.dim(1)))],
            template: [("InT", q.dtype), ("Dk", 128), ("Dv", 128), ("Hk", 16), ("Hv", 32), ("ROWS", stockRows)],
            grid: (4096, 32, 1),
            threadGroup: (32 * stockRows, 1, 1),
            outputShapes: [v.shape, initial.shape], outputDTypes: [q.dtype, .float32])
        return (result[0], result[1])
    }
}
