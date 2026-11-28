#if canImport(Darwin)
import Foundation
import Metal
import MLX
import MLXLMCommon

/// Resume the native STEEL attention kernel at SSD page boundaries.
enum Inkling2SpilledPrefill {
    private static let driver = Result { try Driver() }

    static func supports(
        queries: MLXArray, keys: MLXArray, values: MLXArray, pageTokens: Int
    ) -> Bool {
        pageTokens.isMultiple(of: 16) && queries.dim(0) == 1 && queries.dim(1) == 24
            && queries.dim(2) > 8 && queries.dim(3) == 128
            && keys.dim(1) == 8 && keys.shape == values.shape
            && keys.dtype == .bfloat16 && values.dtype == .bfloat16 && queries.dtype == .bfloat16
            && Device.defaultDevice().deviceType == .gpu
            && GPU.deviceInfo().architecture == "applegpu_g13s"
    }

    static func attention(
        queries: MLXArray, projected: MLXArray, sinks: MLXArray,
        history: ActiveKVSpillHistory, extent: Int
    ) throws -> MLXArray {
        guard history.pageTokens.isMultiple(of: 16) else {
            throw PrefillError.failed("The SSD page geometry must align with a Metal key tile")
        }
        return try driver.get().attention(
            queries: queries, projected: projected, sinks: sinks, history: history, extent: extent)
    }

    private enum PrefillError: LocalizedError {
        case failed(String)

        var errorDescription: String? {
            switch self {
            case .failed(let reason): reason
            }
        }
    }

    private final class Driver: @unchecked Sendable {
        let device: any MTLDevice
        let queue: any MTLCommandQueue
        let pipelines: [any MTLComputePipelineState]

        init() throws {
            let pipelines = try (0..<4).map { index in
                guard let pipeline = try GPU.inklingPagedPrefillPipeline(
                    alignedQueries: index & 2 != 0, alignedKeys: index & 1 != 0
                ) else {
                    throw PrefillError.failed("The bundled Inkling paging kernel is required")
                }
                return pipeline
            }
            let device = pipelines[0].device
            guard let queue = device.makeCommandQueue() else {
                throw PrefillError.failed("An active Metal device is required")
            }
            self.pipelines = pipelines
            self.device = device
            self.queue = queue
        }

        func attention(
            queries: MLXArray, projected: MLXArray, sinks: MLXArray,
            history: ActiveKVSpillHistory, extent: Int
        ) throws -> MLXArray {
            let rows = queries.dim(2)
            let heads = 24
            let kvHeads = 8
            let width = 128
            let total = history.count
            let output = try allocate(heads * rows * width * 2)
            let state = try (0..<2).map { _ in try allocate(heads * rows * (width + 8) * 4) }
            let query = try buffer(queries)
            let qPositions = MLXArray(Int32(total - rows)..<Int32(total)).expandedDimensions(axis: 1)
            let zero = MLXArray.zeros([1, kvHeads, 1, width], dtype: queries.dtype)
            var pageIndex = 0
            var visiblePageMask: (any MTLBuffer)?
            try history.forEachTile { range, pageKeys, pageValues in
                let isLast = range.upperBound == total
                let length = range.count + (isLast ? 1 : 0)
                let keys = isLast ? concatenated([pageKeys, zero], axis: 2) : pageKeys
                let values = isLast ? concatenated([pageValues, zero], axis: 2) : pageValues
                let maskBuffer: any MTLBuffer
                let maskStrides: [Int64]
                if total - rows - (range.upperBound - 1) >= extent {
                    // Fully visible keys outside the relative bias range share a zero row.
                    let zeros = try visiblePageMask ?? buffer(
                        .zeros([history.pageTokens], dtype: queries.dtype))
                    visiblePageMask = zeros
                    maskBuffer = zeros
                    maskStrides = [0, 0, 0]
                } else {
                    let pagePositions = MLXArray(Int32(range.lowerBound)..<Int32(range.upperBound))
                    let distance = qPositions - pagePositions
                    let maskShape = [1, 1, rows, range.count]
                    let indices = broadcast(clip(distance, min: 0, max: extent - 1).reshaped(maskShape),
                                            to: [1, heads, rows, range.count])
                    let bias = takeAlong(projected, indices, axis: -1)
                    let mask = MLX.where(
                        distance.reshaped(maskShape) .< 0, MLXArray(Float(-1e30)),
                        MLX.where(distance.reshaped(maskShape) .>= extent, MLXArray(Float(0)), bias)
                    ).asType(queries.dtype)
                    maskBuffer = try buffer(isLast ? concatenated([mask, sinks], axis: -1) : mask)
                    maskStrides = [Int64(heads * rows * length), Int64(rows * length), Int64(length)]
                }
                let queryTiles = (rows + 31) / 32
                let scale = Float(1) / Float(width).squareRoot()
                let flags = (rows % 32 == 0 ? 2 : 0) | ((total + 1) % 16 == 0 ? 1 : 0)
                // Match the vendored Metal AttnParams layout.
                let fields = [1, heads, width, rows, length, heads / kvHeads, Int(scale.bitPattern),
                              queryTiles, (length + 15) / 16, rows / 32, length / 16,
                              rows % 32, length % 16, length - rows]
                let strides = [heads * rows * width, rows * width, width,
                               kvHeads * length * width, length * width, width,
                               kvHeads * length * width, length * width, width,
                               heads * rows * width, width, heads * width]
                var args = fields.map { UInt32(bitPattern: Int32(truncatingIfNeeded: $0)) }
                    .withUnsafeBytes { Data($0) }
                strides.map(Int64.init).withUnsafeBytes { args.append(contentsOf: $0) }
                let keysBuffer = try buffer(keys)
                let valuesBuffer = try buffer(values)
                guard let command = queue.makeCommandBuffer(),
                      let encoder = command.makeComputeCommandEncoder() else {
                    throw PrefillError.failed("Unable to create a Metal command")
                }
                encoder.setComputePipelineState(pipelines[flags])
                encoder.setBuffer(query, offset: 0, index: 0)
                encoder.setBuffer(keysBuffer, offset: 0, index: 1)
                encoder.setBuffer(valuesBuffer, offset: 0, index: 2)
                encoder.setBuffer(output, offset: 0, index: 3)
                args.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 4) }
                maskStrides.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 5) }
                encoder.setBuffer(maskBuffer, offset: 0, index: 6)
                encoder.setBuffer(state[(pageIndex + 1) % 2], offset: 0, index: 9)
                encoder.setBuffer(state[pageIndex % 2], offset: 0, index: 10)
                var phase = UInt32((pageIndex == 0 ? 0 : 1) | (isLast ? 0 : 2))
                encoder.setBytes(&phase, length: MemoryLayout<UInt32>.size, index: 11)
                encoder.dispatchThreadgroups(MTLSize(width: queryTiles, height: heads, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
                encoder.endEncoding()
                command.commit()
                command.waitUntilCompleted()
                if let error = command.error { throw error }
                pageIndex += 1
            }
            return MLXArray(Data(bytes: output.contents(), count: output.length),
                            [1, rows, heads, width], dtype: queries.dtype).transposed(0, 2, 1, 3)
        }

        private func allocate(_ length: Int) throws -> any MTLBuffer {
            guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw PrefillError.failed("Unable to allocate a Metal continuation buffer")
            }
            return buffer
        }

        private func buffer(_ array: MLXArray) throws -> any MTLBuffer {
            guard let buffer = array.asMTLBuffer(device: device) else {
                throw PrefillError.failed("Unable to stage an attention operand")
            }
            return buffer
        }
    }
}
#endif
