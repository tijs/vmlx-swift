import Foundation
import MLX
import MLXNN
import XCTest
@testable import MLXLLM

final class InklingExpertPrefillProbeTests: XCTestCase {
    func testPublishedPackedExpertPrefillDump() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["VMLX_INKLING_EXPERT_PROBE"] == "1" else { throw XCTSkip("Explicit supervised packed-expert diagnostic") }
        let directory = URL(fileURLWithPath: try XCTUnwrap(env["VMLX_INKLING_PROBE_MODEL"]))
        let output = URL(fileURLWithPath: try XCTUnwrap(env["VMLX_INKLING_PROBE_OUTPUT"]))
        let fixture = URL(fileURLWithPath: try XCTUnwrap(env["VMLX_INKLING_EXPERT_FIXTURE"]))
        let physical = try XCTUnwrap(Int(try XCTUnwrap(env["VMLX_INKLING_EXPERT_LAYER"])))
        let repeats = try XCTUnwrap(Int(try XCTUnwrap(env["VMLX_INKLING_EXPERT_REPEATS"])))
        guard [2,20].contains(physical), [1,8].contains(repeats),
            !FileManager.default.fileExists(atPath: output.path) else { throw NSError(domain: "InklingExpertProbe", code: 1) }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let config = try JSONDecoder().decode(Inkling2LoopConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        try Device.withDefaultDevice(.gpu) {
            Memory.cacheLimit = 536_870_912
            Memory.memoryLimit = 4 * 1024 * 1024 * 1024
            var arrays = try loadArrays(url: fixture)
            let original = try XCTUnwrap(arrays.removeValue(forKey: "fixture.input")).asType(.bfloat16)
            let selected = try XCTUnwrap(arrays.removeValue(forKey: "fixture.indices")).asType(.uint32)
            let input = concatenated(Array(repeating: original, count: repeats), axis: 1)
            let indices = concatenated(Array(repeating: selected, count: repeats), axis: 1)
            let mlp = InklingSwitchMLP(config, layer: physical)
            try mlp.update(parameters: ModuleParameters.unflattened(arrays), verify: .all)
            eval(mlp)
            let routed = mlp.project(input, indices, sortPrefill: true)
            eval(routed)
            XCTAssertEqual(routed.shape, indices.shape + [config.text.hidden])
            XCTAssertTrue(all(isFinite(routed)).item(Bool.self))
            try save(arrays: ["routed": routed.asType(.float32), "input": input.asType(.float32),
                "indices": indices.asType(.float32)], url: output.appendingPathComponent("expert-output.safetensors"))
            try JSONSerialization.data(withJSONObject: ["status":"completed", "layer":physical,
                "shape":routed.shape,"input_positions":input.dim(1),"selected_routes":indices.size,
                "dtype":String(describing:routed.dtype),"scope":"Packed expert dump only; independent reference comparison separate"],
                options: [.prettyPrinted]).write(to: output.appendingPathComponent("record.json"))
        }
    }
}
