// SPDX-License-Identifier: Apache-2.0 AND MIT
// Prototype adapted from Rapid-MLX@7d2b20c7f7fe4707f0690d3dd720be0b7b4f7725:
// rapid_mlx/kernels/qwen4_fused_gdn_decode.py.
// Metal precision/reduction structure originally adapted from mlx-vlm #2105,
// Copyright 2025 Prince Canuma, MIT. This preserves the upstream notices.
// Default-off Qwen35 precise-convolution candidate; final combination qualification pending.
import Foundation
import MLX
import MLXFast

struct ExperimentalGDNInputs {
    let qkv, z, beta, alpha, convState, convWeight: MLXArray
    let aLog, dtBias, recurrentState, normWeight: MLXArray
    let normEps: Float
}

enum ExperimentalFusedGDN {
    static let enabled = ProcessInfo.processInfo.environment["MEI_EXPERIMENTAL_GDN"] == "1"
    static let verify = ProcessInfo.processInfo.environment["MEI_EXPERIMENTAL_GDN_VERIFY"] == "1"
    static let forceStrict = ProcessInfo.processInfo.environment["VMLX_GDN_STRICT"] == "1"
    static let kernel = MLXFast.metalKernel(
        name: "mei_experimental_qwen35_fused_gdn",
        inputNames: ["qkv", "z", "beta", "alpha", "conv_state", "conv_weight", "A_log", "dt_bias", "recurrent_state", "norm_weight", "norm_eps"],
        outputNames: ["output", "conv_state_out", "recurrent_state_out"],
        source: #"""

  const uint hv = threadgroup_position_in_grid.z;
  const uint hk = hv / RATIO;
  const uint lane = thread_position_in_threadgroup.x;
  const uint ty = thread_position_in_threadgroup.y;
  const uint tid = thread_index_in_threadgroup;

  constexpr int NT = 32 * TY;
  constexpr int NDK = DK / 32;
  constexpr int NDV = DV / TY;
  constexpr uint KD = (uint)(HK * DK);
  constexpr uint VD = (uint)(HV * DV);
  constexpr uint CD = 2u * KD + VD;

  threadgroup float sq[DK];
  threadgroup float sk[DK];
  threadgroup T sq_squared[DK];
  threadgroup T sk_squared[DK];
  threadgroup float sv[DV];
  threadgroup float sy[DV];
  threadgroup float shr[4];

  device const float* si = recurrent_state + (size_t)hv * DV * DK;
  device float* so = recurrent_state_out + (size_t)hv * DV * DK;
  float st[NDV][NDK];
  for (int j = 0; j < NDV; ++j) {
    uint dv = ty + (uint)TY * (uint)j;
    for (int i = 0; i < NDK; ++i)
      st[j][i] = si[(size_t)dv * DK + NDK * lane + i];
  }

  for (uint idx = tid; idx < (uint)(2 * DK + DV); idx += NT) {
    uint part = idx / (uint)DK;
    uint d = idx - part * (uint)DK;
    uint c = part == 0u ? hk * DK + d
           : (part == 1u ? KD + hk * DK + d : 2u * KD + hv * DV + d);
    device const T* wc = conv_weight + (size_t)c * K;
    float acc = 0.0f;
    for (uint tap = 0; tap + 1 < (uint)K; ++tap)
      acc += float(conv_state[(size_t)tap * CD + c]) * float(wc[tap]);
    acc += float(qkv[c]) * float(wc[K - 1]);
    T xb = static_cast<T>(acc);
    // Qwen3.5 qualifies the convolution SiLU against the installed MLX.
    // Qwen4 retains its original fast law and separate admission gate.
    T sig = mlx_sigmoid_precise(xb);
    T sl = xb * sig;
    if (part == 0u) sq[d] = float(sl);
    else if (part == 1u) sk[d] = float(sl);
    else sv[d] = float(sl);
    if (part == 2u || (hv % RATIO) == 0u) {
      for (uint tap = 0; tap + 2 < (uint)K; ++tap)
        conv_state_out[(size_t)tap * CD + c] =
            conv_state[(size_t)(tap + 1) * CD + c];
      conv_state_out[(size_t)(K - 2) * CD + c] = qkv[c];
    }
  }

  if (tid == 0u) {
    T av = alpha[hv] + dt_bias[hv];
    T sp = mlx_softplus_fast(av);
    shr[2] = metal::precise::exp(
        -metal::precise::exp(float(A_log[hv])) * float(sp));
    // Exhaustive bf16 sweep: mlx_sigmoid_precise<T> equals mx.sigmoid on
    // every finite bf16 input; the fast form differs on one (x ~ -6.85).
    shr[3] = float(mlx_sigmoid_precise(beta[hv]));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if constexpr (Q35) {
    // Qwen3.5-family uses mx.fast.rms_norm: float32 mean-of-squares and
    // reciprocal sqrt, cast back to T, then separate q/k scaling in T.
    if (simdgroup_index_in_threadgroup == 0u) {
      float pq = 0.0f, pk = 0.0f;
      uint base = 4u * lane;
      for (int i = 0; i < 4; ++i) {
        pq += sq[base + i] * sq[base + i];
        pk += sk[base + i] * sk[base + i];
      }
      pq = simd_sum(pq);
      pk = simd_sum(pk);
      if (lane == 0u) {
        shr[0] = metal::precise::rsqrt(pq / float(DK) + 1.0e-6f);
        shr[1] = metal::precise::rsqrt(pk / float(DK) + 1.0e-6f);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    T qscale = static_cast<T>(1.0f / float(DK));
    T kscale = static_cast<T>(metal::precise::rsqrt(float(DK)));
    for (uint d = tid; d < (uint)DK; d += NT) {
      T q_normalized = static_cast<T>(sq[d] * shr[0]);
      T k_normalized = static_cast<T>(sk[d] * shr[1]);
      sq[d] = float(static_cast<T>(q_normalized * qscale));
      sk[d] = float(static_cast<T>(k_normalized * kscale));
    }
  } else {
    for (uint d = tid; d < (uint)DK; d += NT) {
      T qv = static_cast<T>(sq[d]);
      T kv = static_cast<T>(sk[d]);
      sq_squared[d] = static_cast<T>(qv * qv);
      sk_squared[d] = static_cast<T>(kv * kv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index_in_threadgroup == 0u) {
      T pq = static_cast<T>(0), pk = static_cast<T>(0);
      uint base = 4u * lane;
      for (int i = 0; i < 4; ++i) {
        pq = static_cast<T>(sq_squared[base + i] + pq);
        pk = static_cast<T>(sk_squared[base + i] + pk);
      }
      pq = static_cast<T>(simd_sum(float(pq)));
      pk = static_cast<T>(simd_sum(float(pk)));
      if (lane == 0u) {
        T eps = static_cast<T>(1.0e-6f);
        T qdenom = pq + eps;
        T kdenom = pk + eps;
        shr[0] = float(static_cast<T>(metal::precise::rsqrt(qdenom)));
        shr[1] = float(static_cast<T>(metal::precise::rsqrt(kdenom)));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    T qscale = static_cast<T>(0.08838834764831845f);
    for (uint d = tid; d < (uint)DK; d += NT) {
      T q_normalized = static_cast<T>(static_cast<T>(sq[d]) * static_cast<T>(shr[0]));
      T k_normalized = static_cast<T>(static_cast<T>(sk[d]) * static_cast<T>(shr[1]));
      sq[d] = float(static_cast<T>(q_normalized * qscale));
      sk[d] = float(k_normalized);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int j = 0; j < NDV; ++j) {
    uint dv = ty + (uint)TY * (uint)j;
    float kv = 0.0f;
    for (int i = 0; i < NDK; ++i) {
      uint s = NDK * lane + i;
      st[j][i] = st[j][i] * shr[2];
      kv += st[j][i] * sk[s];
    }
    kv = simd_sum(kv);
    float delta = (sv[dv] - kv) * shr[3];
    float out = 0.0f;
    for (int i = 0; i < NDK; ++i) {
      uint s = NDK * lane + i;
      st[j][i] = st[j][i] + sk[s] * delta;
      out += st[j][i] * sq[s];
    }
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0u)
      sy[dv] = float(static_cast<T>(out));
    for (int i = 0; i < NDK; ++i)
      so[(size_t)dv * DK + NDK * lane + i] = st[j][i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simdgroup_index_in_threadgroup == 0u) {
    float po = 0.0f;
    uint base = 4u * lane;
    for (int i = 0; i < 4; ++i) po += sy[base + i] * sy[base + i];
    po = simd_sum(po);
    if (lane == 0u)
      shr[0] = metal::precise::rsqrt(po / (float)DV + norm_eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = tid; d < (uint)DV; d += NT) {
    T normalized = static_cast<T>(sy[d] * shr[0]);
    normalized = norm_weight[d] * normalized;
    // Exhaustive bf16 sweep: the precise float32 sigmoid matches mx.sigmoid
    // on every finite bf16-valued gate; the fast form differs on ~1%.
    if constexpr (Q35) {
      // Match _precise_swiglu's operation order: form the float32 SiLU
      // gate first, then multiply it by the normalized value.
      float zg = float(z[hv * DV + d]);
      float gate = zg * mlx_sigmoid_precise<float>(zg);
      float x = gate * float(normalized);
      output[hv * DV + d] = static_cast<T>(x);
    } else {
      float x = float(normalized) * mlx_sigmoid_precise<float>(float(z[hv * DV + d]));
      output[hv * DV + d] = static_cast<T>(x);
    }
  }

"""#,
        header: #"""

#include <metal_atomic>
template <typename U>
inline U mlx_sigmoid_precise(U x) {
  U e = static_cast<U>(metal::precise::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}

template <typename U>
inline U mlx_sigmoid_fast(U x) {
  U e = static_cast<U>(metal::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}

template <typename U>
inline U mlx_log1p_fast(U x) {
  float xf = float(x);
  float xp1 = 1.0f + xf;
  float out = xp1 == 1.0f ? xf : xf * (metal::log(xp1) / (xp1 - 1.0f));
  return static_cast<U>(out);
}

template <typename U>
inline U mlx_softplus_fast(U x) {
  if (metal::isnan(x))
    return metal::numeric_limits<U>::quiet_NaN();
  constexpr U inf = metal::numeric_limits<U>::infinity();
  U zero = static_cast<U>(0);
  U hi = metal::max(x, zero);
  U lo = metal::min(x, zero);
  return (lo == -inf || hi == inf)
      ? hi
      : (hi + mlx_log1p_fast(static_cast<U>(metal::exp(lo - hi))));
}


"""#)

    static func recordFailure(input x: ExperimentalGDNInputs, reference: [MLXArray],
                              candidate: [MLXArray], layer: Int) {
        for index in reference.indices {
            let error = abs(reference[index].asType(.float32) - candidate[index].asType(.float32)).max().item(Float.self)
            FileHandle.standardError.write(Data("GDN_REAL_MISMATCH layer=\(layer) tensor=\(index) max_abs=\(error)\n".utf8))
        }
        guard let path = ProcessInfo.processInfo.environment["MEI_EXPERIMENTAL_GDN_FAILURE_DIR"] else { return }
        var arrays = ["qkv": x.qkv, "z": x.z, "beta": x.beta, "alpha": x.alpha,
                      "conv_state": x.convState, "conv_weight": x.convWeight,
                      "a_log": x.aLog, "dt_bias": x.dtBias, "state": x.recurrentState,
                      "norm_weight": x.normWeight, "norm_eps": MLXArray(x.normEps)]
        for index in reference.indices {
            arrays["reference_\(index)"] = reference[index]
            arrays["candidate_\(index)"] = candidate[index]
        }
        do {
            try save(arrays: arrays, url: URL(fileURLWithPath: path)
                .appendingPathComponent("gdn-layer-\(layer).safetensors"))
        } catch {
            FileHandle.standardError.write(Data("GDN fixture save failed: \(error)\n".utf8))
        }
    }

    static func supports(_ input: ExperimentalGDNInputs, threadgroupY: Int) -> Bool {
        guard !forceStrict, Device.defaultDevice().deviceType == .gpu,
            [4, 8, 16, 32].contains(threadgroupY),
            input.normEps.isFinite, input.normEps > 0 else { return false }
        let expected: [(MLXArray, [Int])] = [
            (input.qkv, [1, 1, 8192]), (input.z, [1, 1, 4096]),
            (input.beta, [1, 1, 32]), (input.alpha, [1, 1, 32]),
            (input.convState, [1, 3, 8192]), (input.convWeight, [8192, 4, 1]),
            (input.dtBias, [32]), (input.normWeight, [128]),
        ]
        return expected.allSatisfy { $0.0.dtype == .bfloat16 && $0.0.shape == $0.1 }
            && input.aLog.shape == [32] && [.bfloat16, .float32].contains(input.aLog.dtype)
            && input.recurrentState.shape == [1, 32, 128, 128]
            && input.recurrentState.dtype == .float32
    }

    static func call(_ input: ExperimentalGDNInputs, threadgroupY: Int, preciseConvolution: Bool) -> (MLXArray, MLXArray, MLXArray)? {
        guard preciseConvolution, supports(input, threadgroupY: threadgroupY) else { return nil }
        let result = kernel(
            [input.qkv, input.z, input.beta, input.alpha, input.convState, input.convWeight,
             input.aLog, input.dtBias, input.recurrentState, input.normWeight, MLXArray(input.normEps)],
            template: [("T", DType.bfloat16), ("HK", 16), ("HV", 32), ("DK", 128),
                       ("DV", 128), ("K", 4), ("TY", threadgroupY), ("RATIO", 2),
                       ("Q35", true)],
            grid: (32, threadgroupY, 32), threadGroup: (32, threadgroupY, 1),
            outputShapes: [[1, 1, 4096], [1, 3, 8192], [1, 32, 128, 128]],
            outputDTypes: [.bfloat16, .bfloat16, .float32])
        return (result[0], result[1], result[2])
    }
}
