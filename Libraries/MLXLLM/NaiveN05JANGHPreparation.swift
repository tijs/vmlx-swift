import Foundation
import MLXLMCommon

/// Validates the architecture and both weight formats before constructing the
/// model. Custom banks never pass through the ordinary affine quantizer.
struct NaiveN05JANGHPreparation {
    let configuration: NaiveN05ArchitectureContract
    let banks: JANGHModelPreparation
    let baseConfiguration: BaseConfiguration

    static func loadIfDeclared(directory: URL, configurationData: Data) throws
        -> NaiveN05JANGHPreparation?
    {
        guard let root = try? JSONSerialization.jsonObject(with: configurationData) as? [String: Any],
            root["model_type"] as? String == "naive_n05_flash"
        else { return nil }
        let sidecarURL = directory.appendingPathComponent("jang_config.json")
        let sidecar = FileManager.default.fileExists(atPath: sidecarURL.path)
            ? try Data(contentsOf: sidecarURL) : nil
        guard JANGHModelPreparation.declaresCustomFormat(
            configuration: configurationData, sidecar: sidecar) else { return nil }
        return try Self(directory: directory, configurationData: configurationData, sidecar: sidecar)
    }

    init(directory: URL, configurationData: Data, sidecar: Data?) throws {
        let config = try JSONDecoder.json5().decode(
            NaiveN05ArchitectureContract.self, from: configurationData)
        guard !NativeMTPActivation.isExplicitlyRequested else {
            throw NaiveN05ArchitectureContract.ContractError.unsupported(
                "native MTP is not implemented for Naive-N0.5")
        }
        let prepared = try JANGHModelPreparation(
            directory: directory, configuration: configurationData, sidecar: sidecar,
            layout: .init(
                modelType: "naive_n05_flash", hiddenSize: config.hiddenDimensions,
                intermediateSize: config.expertDimensions, expertCount: config.expertCount,
                sparseLayers: Set(config.routedLayers.indices.filter { config.routedLayers[$0] })))
        baseConfiguration = try JSONDecoder.json5().decode(
            BaseConfiguration.self, from: prepared.ordinaryConfiguration)
        configuration = config
        banks = prepared
    }

    func construct(requesting: Set<ModelRuntimeRequestModality>?) throws -> NaiveN05FlashModel {
        guard requesting == nil || requesting == [.text] else {
            throw NaiveN05ArchitectureContract.ContractError.unsupported(
                "Naive-N0.5 accepts text input only")
        }
        let routed = try banks.makeRoutedExperts(activationLimit: nil)
        return try NaiveN05FlashModel(
            configuration,
            routedFactory: { layer, _ in
                guard let bank = routed[layer] else {
                    throw NaiveN05ArchitectureContract.ContractError.unsupported(
                        "missing admitted routed layer")
                }
                return bank
            },
            excludedSafetensorsKeys: banks.excludedTensorNames)
    }
}
