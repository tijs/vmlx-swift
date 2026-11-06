import Foundation
import MLX
import MLXNN
import MLXVLM
import XCTest

@testable import MLXLMCommon

final class Glm5NextCustomRoutedConstructionTests: XCTestCase {
    private final class FixtureBank: Module, WeightedRoutedExpertLayer {
        @ParameterInfo(key: "tq2_fixture") var marker: MLXArray
        var calls = 0
        var seenScoreShape: [Int]?
        override init() {
            _marker.wrappedValue = MLXArray([UInt32(7)])
            super.init()
        }
        func callRouted(_ input: MLXArray, indices: MLXArray, scores: MLXArray) -> MLXArray {
            calls += 1
            seenScoreShape = scores.shape
            return MLXArray.zeros(input.shape, dtype: input.dtype)
        }
    }

    func testCustomConstructionOwnsCheckpointPathWithoutDenseExpertBanks() throws {
        try MLXMetalTestLock.withLock {
            let config = try JSONDecoder().decode(
                Glm5NextConfiguration.self,
                from: Data(Glm5NextConstructionTests.tinyJSON.replacingOccurrences(
                    of: "\"num_nextn_predict_layers\":1", with: "\"num_nextn_predict_layers\":0").utf8))
            let sparse = (0 ..< config.textConfig.numHiddenLayers).filter {
                config.textConfig.mlpLayerTypes[$0] == .sparse
            }
            var banks: [Int: any WeightedRoutedExpertLayer] = [:]
            for index in sparse { banks[index] = FixtureBank() }
            let model = try Glm5NextLanguageModel(config.textConfig, routedExperts: banks)
            let names = Set(model.parameters().flattened().map { $0.0 })
            for index in sparse {
                let moe = try XCTUnwrap(model.layers[index].moe)
                XCTAssertEqual(moe.routedAdviceLayerIndex, index)
                XCTAssertNil(moe.switchMLP)
                XCTAssertTrue(moe.routedExperts === (try XCTUnwrap(banks[index])))
                XCTAssertTrue(names.contains("layers.\(index).mlp.switch_mlp.tq2_fixture"))
                XCTAssertFalse(names.contains("layers.\(index).mlp.switch_mlp.gate_proj.weight"))
                XCTAssertFalse(names.contains("layers.\(index).mlp.switch_mlp.up_proj.weight"))
                XCTAssertFalse(names.contains("layers.\(index).mlp.switch_mlp.down_proj.weight"))
            }
            let moe = try XCTUnwrap(model.layers[sparse[0]].moe)
            let input = MLXArray.ones([1, 2, config.textConfig.hiddenSize])
            let actual = moe(input)
            let expected = moe.sharedExperts(input)
            eval(actual, expected)
            XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
            XCTAssertEqual((banks[sparse[0]] as? FixtureBank)?.calls, 1)
            XCTAssertEqual((banks[sparse[0]] as? FixtureBank)?.seenScoreShape,
                           [1, 2, config.textConfig.numExpertsPerTok])
        }
    }

    func testIncompleteAndDenseLayerSubstitutionAreRejectedBeforeConstruction() throws {
        try MLXMetalTestLock.withLock {
            let config = try JSONDecoder().decode(
                Glm5NextConfiguration.self,
                from: Data(Glm5NextConstructionTests.tinyJSON.replacingOccurrences(
                    of: "\"num_nextn_predict_layers\":1", with: "\"num_nextn_predict_layers\":0").utf8)).textConfig
            let sparse = (0 ..< config.numHiddenLayers).filter {
                config.mlpLayerTypes[$0] == .sparse
            }
            XCTAssertThrowsError(try Glm5NextLanguageModel(
                config, routedExperts: [sparse[0]: FixtureBank()]))
            var banks: [Int: any WeightedRoutedExpertLayer] = [:]
            for index in sparse { banks[index] = FixtureBank() }
            banks[0] = FixtureBank()
            XCTAssertThrowsError(try Glm5NextLanguageModel(config, routedExperts: banks))
        }
    }
    func testCustomBackboneRefusesUnimplementedActiveMTPBanks() throws {
        try MLXMetalTestLock.withLock {
            let config = try JSONDecoder().decode(Glm5NextConfiguration.self,
                from: Data(Glm5NextConstructionTests.tinyJSON.utf8)).textConfig
            var banks: [Int: any WeightedRoutedExpertLayer] = [:]
            for layer in 0 ..< config.numHiddenLayers where config.mlpLayerTypes[layer] == .sparse {
                banks[layer] = FixtureBank()
            }
            try NativeMTPActivation.$explicitRequestOverride.withValue(true) {
                XCTAssertThrowsError(try Glm5NextLanguageModel(config, routedExperts: banks))
            }
        }
    }

    func testGenericLoaderExcludesOnlyCompleteCustomBankKeys() throws {
        try MLXMetalTestLock.withLock {
            let config = try JSONDecoder().decode(Glm5NextConfiguration.self,
                from: Data(Glm5NextConstructionTests.tinyJSON.replacingOccurrences(
                    of: "\"num_nextn_predict_layers\":1", with: "\"num_nextn_predict_layers\":0").utf8))
            var banks: [Int: any WeightedRoutedExpertLayer] = [:]
            for layer in 0 ..< config.textConfig.numHiddenLayers
                where config.textConfig.mlpLayerTypes[layer] == .sparse {
                banks[layer] = FixtureBank()
            }
            let names = Set(banks.keys.flatMap { layer in
                ["gate_proj", "up_proj", "down_proj"].flatMap { role in
                    ["tq2_packed", "tq2_scales"].map { suffix in
                        "model.layers.\(layer).mlp.switch_mlp.\(role).\(suffix)"
                    }
                }
            })
            let custom = try Glm5Next(config, requesting: [.text], routedExperts: banks,
                                      customRoutedTensorNames: names)
            XCTAssertTrue(custom.requiresExactTensorMmapBuffers)
            for name in names { XCTAssertTrue(custom.excludeFromGenericSafetensorsLoad(key: name)) }
            XCTAssertFalse(custom.excludeFromGenericSafetensorsLoad(key: "model.embed_tokens.weight"))
            XCTAssertThrowsError(try Glm5Next(config, requesting: [.text], routedExperts: banks,
                customRoutedTensorNames: names.union(["model.embed_tokens.weight"])))
            let ordinary = try Glm5Next(config, requesting: [.text])
            XCTAssertFalse(ordinary.requiresExactTensorMmapBuffers)
            for name in names { XCTAssertFalse(ordinary.excludeFromGenericSafetensorsLoad(key: name)) }
        }
    }

}
