import MLX
import MLXFast

/// A single launch performs H32 in F32 registers and the requested output cast.
/// No full-size intermediate F32 activation buffer is materialized.
final class JANGHRowRotation {
    private let kernel = MLXFast.metalKernel(
        name: "jangh_h32_rows_v1", inputNames: ["x"], outputNames: ["out"],
        source: """
            uint lane = thread_index_in_simdgroup;
            size_t offset = size_t(threadgroup_position_in_grid.z) * K
                + size_t(threadgroup_position_in_grid.y) * 32u + lane;
            float value = float(x[offset]);
            for (ushort stage = 1; stage < 32; stage <<= 1) {
                float other = simd_shuffle_xor(value, stage);
                value = (lane & stage) ? other - value : value + other;
            }
            out[offset] = OUT_T(value * 0.17677669529663687f);
            """, ensureRowContiguous: false)

    func callAsFunction(_ input: MLXArray, outputDType: DType? = nil) throws -> MLXArray {
        let dtype = outputDType ?? input.dtype
        guard input.ndim == 2, input.dim(0) > 0, input.dim(1) > 0,
            input.dim(1).isMultiple(of: 32),
            [.float16, .bfloat16, .float32].contains(input.dtype),
            [.float16, .bfloat16, .float32].contains(dtype)
        else { throw JANGHFormatContract.ValidationError.invalid("invalid H32 row geometry or dtype") }
        return kernel([contiguous(input)], template: [("K", input.dim(1)), ("OUT_T", dtype)],
                      grid: (32, input.dim(1) / 32, input.dim(0)), threadGroup: (32, 1, 1),
                      outputShapes: [input.shape], outputDTypes: [dtype])[0]
    }
}
