import Cmlx
import Darwin
import Foundation
import MLX
import XCTest

final class JANGHDenseNamedMmapTests: XCTestCase {
    func testDenseNamesRetainMappedOwnershipAndRejectInvalidBanks() throws {
        try MLXMetalTestLock.withLock {
            let page = Int(getpagesize())
            let path = FileManager.default.temporaryDirectory.appendingPathComponent(
                "dense-bank-\(UUID().uuidString)")
            try Data(repeating: 0, count: 4 * page).write(to: path)
            defer { try? FileManager.default.removeItem(at: path) }
            let baseline = mlx_safetensors_mmap_tracked_buffer_bytes()
            let layer: Int32 = 1_900_000_001
            func map(_ name: String, _ shape: [Int32], _ dtype: DType, _ length: Int) throws
                -> MLXArray
            {
                var raw = mlx_array_new()
                var dimensions = shape
                do {
                    let result = try withError {
                        path.path.withCString { file in
                            name.withCString { tensor in
                                mlx_array_new_mmap_file_region_named(
                                    &raw, file, UInt64(page), length,
                                    &dimensions, Int32(dimensions.count), dtype.cmlxDtype, tensor)
                            }
                        }
                    }
                    XCTAssertEqual(result, 0)
                } catch {
                    mlx_array_free(raw)
                    throw error
                }
                return MLXArray(raw)
            }
            for packed in [true, false] {
                let shape: [Int32] = packed ? [1, Int32(page / 4), 1] : [1, Int32(page / 2)]
                let name =
                    "model.layers.\(layer).mlp.down_proj.tq2_" + (packed ? "packed" : "scales")
                var bank: MLXArray? = try map(name, shape, packed ? .uint32 : .float16, page)
                XCTAssertEqual(bank?.shape, shape.map(Int.init))
                XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(), baseline + Int64(page))
                XCTAssertEqual(mlx_safetensors_mmap_advise_layer(1, layer), Int64(page))
                var selectedLayer = layer
                var expert: Int32 = 0
                XCTAssertEqual(
                    mlx_safetensors_mmap_advise_experts(1, &selectedLayer, &expert, 1), Int64(page))
                expert = 1
                XCTAssertEqual(
                    mlx_safetensors_mmap_advise_experts(1, &selectedLayer, &expert, 1), 0)
                bank = nil
                XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(), baseline)
                XCTAssertEqual(mlx_safetensors_mmap_advise_layer(1, layer), 0)
            }
            let shape: [Int32] = [1, Int32(page / 4), 1]
            // Dense Qwen3.5-family JANGH (27B JANGH2) maps gate/up/down banks, with or without the VLM
            // `language_model.` prefix.
            for name in [
                "model.layers.0.mlp.gate_proj.tq2_packed",
                "model.layers.0.mlp.up_proj.tq2_packed",
                "language_model.model.layers.0.mlp.down_proj.tq2_packed",
                "language_model.model.layers.63.mlp.gate_proj.tq2_packed",
            ] {
                var bank: MLXArray? = try map(name, shape, .uint32, page)
                XCTAssertEqual(bank?.shape, shape.map(Int.init))
                bank = nil
                XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(), baseline)
            }
            for name in [
                "model.layers.0.mlp.experts.gate_proj.tq2_packed",
                "vision_tower.model.layers.0.mlp.up_proj.tq2_packed",
                "language_model.layers.0.mlp.up_proj.tq2_packed",
                "model.layers.00.mlp.down_proj.tq2_packed",
                "model.layers.2147483648.mlp.down_proj.tq2_packed",
            ] {
                XCTAssertThrowsError(try map(name, shape, .uint32, page))
            }
            let valid = "model.layers.0.mlp.down_proj.tq2_packed"
            XCTAssertThrowsError(try map(valid, [2, Int32(page / 4), 1], .uint32, 2 * page))
            XCTAssertThrowsError(try map(valid, shape, .float16, page))
            XCTAssertThrowsError(try map(valid, shape, .uint32, page - 1))
            XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(), baseline)
        }
    }
}
