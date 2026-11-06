import Foundation
import XCTest
@testable import MLXVLM

final class Glm5NextJANGHDetectionTests: XCTestCase {
    private func config(_ quantization: [String: Any]?) throws -> Data {
        var root = try XCTUnwrap(JSONSerialization.jsonObject(
            with: Data(Glm5NextConstructionTests.tinyJSON.utf8)) as? [String: Any])
        if let quantization { root["quantization"] = quantization }
        return try JSONSerialization.data(withJSONObject: root)
    }

    func testOrdinaryFormatsAndNamesDoNotEnableCustomLoading() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("JANGH2-" + UUID().uuidString)
        for ordinary in [nil, ["mode": "affine", "bits": 4, "group_size": 64],
                         ["mode": "mxfp8", "bits": 8, "group_size": 32]] as [[String: Any]?] {
            XCTAssertNil(try Glm5NextJANGHPreparation.loadIfDeclared(
                directory: directory, configurationData: config(ordinary)))
        }
    }

    func testExplicitCustomDeclarationsCannotFallBackWhenMalformed() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let custom = ["model.layers.2.mlp.switch_mlp.gate_proj": ["mode": "jangtq2", "bits": 2]]
        XCTAssertThrowsError(try Glm5NextJANGHPreparation.loadIfDeclared(
            directory: directory, configurationData: config(custom)))
        try Data(#"{"format":"jangtq2","format_version":2}"#.utf8)
            .write(to: directory.appendingPathComponent("jang_config.json"))
        XCTAssertThrowsError(try Glm5NextJANGHPreparation.loadIfDeclared(
            directory: directory, configurationData: config(nil)))
    }
}
