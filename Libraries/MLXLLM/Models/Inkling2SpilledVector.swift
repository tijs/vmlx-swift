#if canImport(Darwin)
import Foundation
import MLX
import MLXFast
import MLXLMCommon

/// Paged masked vector attention using the native M1 Max partition order.
/// Keep the float accumulators alive until the final page so native-dtype
/// rounding happens at the same boundary as MLX's first vector pass.
enum Inkling2SpilledVector {
    private static let usesDefaultPartitionCount =
        ProcessInfo.processInfo.environment["MLX_SDPA_BLOCKS"] == nil

    static func supports(queries: MLXArray, keys: MLXArray, values: MLXArray, count: Int) -> Bool {
        let rows = queries.dim(2)
        guard usesDefaultPartitionCount, count >= 1023, count < Int(Int32.max),
              (1...8).contains(rows),
              queries.shape == [1, 24, rows, 128], keys.shape == [1, 8, rows, 128],
              values.shape == keys.shape,
              queries.dtype == .bfloat16, keys.dtype == .bfloat16, values.dtype == .bfloat16,
              Device.defaultDevice().deviceType == .gpu,
              GPU.deviceInfo().architecture == "applegpu_g13s" else { return false }
        return true
    }

    private static let update = MLXFast.metalKernel(
        name: "inkling_spill_vector_update",
        inputNames: ["query", "keys", "values", "bias", "sinks", "prior", "page"],
        outputNames: ["next"],
        source: """
            int lane = thread_index_in_simdgroup;
            int head = 3 * threadgroup_position_in_grid.x + thread_position_in_threadgroup.y;
            int partition = threadgroup_position_in_grid.z / ROWS;
            int row = threadgroup_position_in_grid.z % ROWS;
            int line = head * ROWS + row;
            int kvHead = head / 3;
            int base = (line * PARTITIONS + partition) * 130;
            int length = keys_shape[2];
            int start = page[0], total = page[1];
            int end = start + length;
            int offset = (partition + PARTITIONS - start % PARTITIONS) % PARTITIONS;
            float q[4], numerator[4];
            for (int col = 0; col < 4; ++col) {
                q[col] = 0.08838834764831845f * float(query[line * 128 + lane * 4 + col]);
                numerator[col] = prior[base + lane * 4 + col];
            }
            float peak = prior[base + 128], sum = prior[base + 129];
            for (int key = start + offset; key < end + int(end == total); key += PARTITIONS) {
                bool sink = key == total;
                int distance = total - ROWS + row - key;
                float mask = 0.0f;
                if (sink) {
                    mask = float(sinks[line]);
                } else if (distance < 0) {
                    mask = float(bfloat(-1e30f));
                } else if (distance < bias_shape[3]) {
                    mask = float(bias[line * bias_shape[3] + distance]);
                }
                if (!(mask >= -3.38953139e+38f)) continue;
                int stride = (kvHead * length + key - start) * 128 + lane * 4;
                float score = 0.0f;
                for (int col = 0; col < 4; ++col) {
                    score = fma(q[col], sink ? 0.0f : float(keys[stride + col]), score);
                }
                score = simd_sum(score) + mask;
                float maximum = max(peak, score);
                float factor = fast::exp(peak - maximum);
                float weight = fast::exp(score - maximum);
                peak = maximum;
                sum = fma(sum, factor, weight);
                for (int col = 0; col < 4; ++col) {
                    float value = sink ? 0.0f : float(values[stride + col]);
                    // Match the contraction used by the prebuilt vector kernel.
                    numerator[col] = fma(weight, value, numerator[col] * factor);
                }
            }
            if (lane == 0) {
                next[base + 128] = peak;
                next[base + 129] = sum;
            }
            for (int col = 0; col < 4; ++col) {
                next[base + lane * 4 + col] = numerator[col];
            }
            """)

    // Match the M1 Max's two-pass dispatch in Metal's scaled dot product attention.
    private static func partitionCount(keysWithSink: Int, rows: Int) -> Int {
        guard rows > 1, keysWithSink > 1024 else { return 64 }
        switch keysWithSink {
        case ...8192: return 128
        case ...32768: return 256
        case ...65536: return 512
        default: return 1024
        }
    }

    static func attention(
        queries: MLXArray, projectedBias: MLXArray, sinks: MLXArray,
        history: ActiveKVSpillHistory
    ) throws -> MLXArray {
        let rows = queries.dim(2)
        precondition(queries.shape == [1, 24, rows, 128] && (1...8).contains(rows) && history.count >= 1023)
        let partitions = partitionCount(keysWithSink: history.count + 1, rows: rows)
        let templates: [(String, any KernelTemplateArg)] = [("ROWS", rows), ("PARTITIONS", partitions)]
        let shape = [24, rows, partitions, 130]
        var state = concatenated([
            MLXArray.zeros([24, rows, partitions, 128], dtype: .float32),
            MLXArray.full([24, rows, partitions, 1], values: MLXArray(-Float.greatestFiniteMagnitude)),
            MLXArray.zeros([24, rows, partitions, 1], dtype: .float32),
        ], axis: -1)
        try withError { error in
            try history.forEachTile { range, keys, values in
                state = update(
                    [queries, keys, values, projectedBias, sinks, state,
                     MLXArray([Int32(range.lowerBound), Int32(history.count)])],
                    template: templates, grid: (256, 3, rows * partitions),
                    threadGroup: (32, 3, 1), outputShapes: [shape],
                    outputDTypes: [.float32])[0]
                eval(state)
                try error.check()
            }
        }
        return try GPU.inklingPagedVectorMerge(
            partials: contiguous(state[.ellipsis, ..<128].asType(queries.dtype)),
            sums: contiguous(state[.ellipsis, 129]),
            maxima: contiguous(state[.ellipsis, 128]))
    }
}
#endif
