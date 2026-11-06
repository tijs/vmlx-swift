import Foundation
import MLXLMCommon

/// Qwen3.8-Flash-Next (qwen4_exp) JANGH routed experts: 48 sparse layers x 512 experts, top-10,
/// gate/up 2560->640, down 640->2560, hadamard32-rotated odd-cubic codebook banks
/// (`model.layers.L.mlp.switch_mlp.{gate,up,down}_proj.tq2_{packed,scales}`).
/// Everything else in the bundle (attention/GDN, shared experts, router, mHC, embeddings, head,
/// vision, MTP incl. its own experts) is ordinary affine and loads through the generic path;
/// the 51B PLE n-gram table stays file-backed (Qwen4Exp.configure(modelDirectory:)).
struct Qwen4ExpJANGHPreparation {
    let banks: JANGHModelPreparation

    /// Detect the explicit custom contract (config `jangtq` header / `mode: jangtq2`), never a
    /// folder-name substring. Once declared, malformed metadata is an error, not an affine fallback.
    static func loadIfDeclared(directory: URL, configurationData: Data) throws -> Self? {
        guard let root = try? JSONSerialization.jsonObject(with: configurationData) as? [String: Any],
              root["model_type"] as? String == "qwen4_exp" else { return nil }
        let sidecarURL = directory.appendingPathComponent("jang_config.json")
        let sidecar = FileManager.default.fileExists(atPath: sidecarURL.path)
            ? try Data(contentsOf: sidecarURL) : nil
        guard JANGHModelPreparation.declaresCustomFormat(configuration: configurationData, sidecar: sidecar)
        else { return nil }
        return try Self(directory: directory, configurationData: configurationData, sidecar: sidecar)
    }

    init(directory: URL, configurationData: Data, sidecar: Data?) throws {
        // Read the routed geometry from the raw JSON: Qwen4ExpConfiguration embeds the strict
        // quantization decoder, which (correctly) refuses `mode: jangtq2` before the partition
        // has replaced the custom modules with `false`.
        guard let root = try JSONSerialization.jsonObject(with: configurationData) as? [String: Any],
              let text = root["text_config"] as? [String: Any],
              let hidden = text["hidden_size"] as? Int, let inter = text["moe_intermediate_size"] as? Int,
              let experts = text["num_experts"] as? Int, let layers = text["num_hidden_layers"] as? Int,
              let topK = text["num_experts_per_tok"] as? Int,
              hidden > 0, inter > 0, experts > 0, layers > 0,
              topK > 0, topK <= experts
        else { throw Qwen4ExpJANGHError.invalidCustomTensorExclusions }
        banks = try JANGHModelPreparation(
            directory: directory, configuration: configurationData, sidecar: sidecar,
            layout: .init(modelType: "qwen4_exp", hiddenSize: hidden, intermediateSize: inter,
                          expertCount: experts, sparseLayers: Set(0 ..< layers), routesPerToken: topK))
    }

    /// `configurationData` is the factory's merged ordinary configuration (custom modules
    /// already set to `false`; inactive MTP already scrubbed).
    func construct(configurationData: Data, requesting: Set<ModelRuntimeRequestModality>?) throws -> Qwen4Exp {
        let config = try JSONDecoder.json5().decode(Qwen4ExpConfiguration.self, from: configurationData)
        _ = try Qwen4Exp.resolveConstruction(config, requesting: requesting)
        let routed = try banks.makeRoutedExperts(activationLimit: nil)   // no SwiGLU clamp in qwen4_exp
        return try Qwen4Exp(config, requesting: requesting, routedExperts: routed,
                            customRoutedTensorNames: banks.excludedTensorNames)
    }
}
