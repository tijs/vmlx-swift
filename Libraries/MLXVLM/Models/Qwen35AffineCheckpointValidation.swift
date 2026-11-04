// Copyright © 2026 Apple Inc.
import Foundation
import CryptoKit
import MLX
import MLXLMCommon

/// Research fixture validation only. No loading, serving, immutable checkpoint
/// binding or proof that the generic loader uses the declared policy is implied.
/// The caller must pass a fresh, pristine model and trusted, stable files.
/// `validateSelectedFiles` loads every tensor, so it is suitable only for tiny
/// fixtures. A single shard without an index bypasses strict manifest checks.
struct Qwen35AffineCheckpointValidation {
    struct ValidationError: Error, CustomStringConvertible {
        let description: String
        init(_ description: String) { self.description = description }
    }
    private let resident: Qwen35AffineResidentContract
    private let routed: Qwen35AffineRoutedContract
    private let routedKeys: Set<String>
    private let directory: URL
    private let shards: [String]
    private let normConvention: String?

    init(model: Qwen35, bits: Int, groupSize: Int, directory: URL, shardNames: Set<String>,
         metadataHashes: [String: String] = [:]) throws {
        let text = model.config.textConfiguration
        guard (type(of: model) == Qwen35.self || type(of: model) == Qwen35MoE.self),
            model.modalities == [.text], text.hiddenLayers > 0, text.numExperts > 0,
            text.mtpNumHiddenLayers == 0, text.weightFormat.isEmpty,
            !JANGTQStreamingExperts.isEnabled, !shardNames.isEmpty,
            Set(metadataHashes.keys).isSubset(of: ["config.json", "model.safetensors.index.json"]),
            shardNames.allSatisfy({ $0.range(of: #"\Amodel(-[0-9]{5}-of-[0-9]{5})?\.safetensors\z"#, options: .regularExpression) != nil }),
            shardNames == ["model.safetensors"] || metadataHashes["model.safetensors.index.json"] != nil,
            metadataHashes.values.allSatisfy({ $0.utf8.count == 64 && $0.utf8.allSatisfy {
                (48...57).contains($0) || (97...102).contains($0)
            } })
        else { throw ValidationError("unsupported model or research configuration") }
        let files = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        guard Set(files) == shardNames.union(metadataHashes.keys) else {
            throw ValidationError("unsupported sidecar, transform or shard layout")
        }
        for file in files {
            let url = directory.appendingPathComponent(file)
            let properties = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            guard properties.isRegularFile == true, properties.isSymbolicLink != true else {
                throw ValidationError("non-regular checkpoint file")
            }
        }
        for (name, expected) in metadataHashes {
            let handle = try FileHandle(forReadingFrom: directory.appendingPathComponent(name))
            defer { try? handle.close() }
            var digest = SHA256()
            while let chunk = try handle.read(upToCount: 1024 * 1024), !chunk.isEmpty {
                digest.update(data: chunk)
            }
            guard digest.finalize().map({ String(format: "%02x", $0) }).joined() == expected else {
                throw ValidationError("metadata SHA-256 mismatch: \(name)")
            }
        }
        if metadataHashes["config.json"] != nil {
            let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
            let decoded = try JSONDecoder().decode(Qwen35Configuration.self, from: data)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            guard try encoder.encode(decoded) == encoder.encode(model.config) else {
                throw ValidationError("checkpoint config differs from model")
            }
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                (json["quantization"] == nil) != (json["quantization_config"] == nil),
                let quant = (json["quantization"] ?? json["quantization_config"]) as? [String: Any],
                Set(quant.keys).isSubset(of: ["bits", "group_size", "mode"])
            else { throw ValidationError("unsupported config quantization declaration") }
            let quantData = try JSONSerialization.data(withJSONObject: quant)
            let declared = try JSONDecoder().decode(BaseConfiguration.Quantization.self, from: quantData)
            guard declared.bits == bits, declared.groupSize == groupSize, declared.mode == .affine else {
                throw ValidationError("config quantization differs from pinned policy")
            }
        }
        if metadataHashes["model.safetensors.index.json"] != nil {
            _ = try Qwen35AffineResearchManifest.validate(modelDirectory: directory,
                selectedShards: shardNames.sorted().map { directory.appendingPathComponent($0) },
                layerCount: text.hiddenLayers, numExperts: text.numExperts,
                inputDims: text.hiddenSize, hiddenDims: text.moeIntermediateSize)
        }
        var names = Set<String>()
        var policies: [String: Qwen35AffineRoutedContract.Policy] = [:]
        for layer in 0..<text.hiddenLayers {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let path = "language_model.model.layers.\(layer).mlp.switch_mlp.\(projection)"
                policies[path] = .init(bits: bits, groupSize: groupSize)
                for suffix in ["weight", "scales", "biases"] { names.insert(path + "." + suffix) }
            }
        }
        self.routedKeys = names
        self.directory = directory
        self.shards = shardNames.sorted()
        self.normConvention = text.normConvention
        self.resident = try Qwen35AffineResidentContract.capture(model: model, affine: [:],
            routed: Set(policies.keys.map { $0 + ".weight" }), tiedHead: text.tieWordEmbeddings,
            layerCount: text.hiddenLayers)
        self.routed = try Qwen35AffineRoutedContract.capture(model: model, layerCount: text.hiddenLayers,
            numExperts: text.numExperts, inputDims: text.hiddenSize, hiddenDims: text.moeIntermediateSize,
            policies: policies)
    }

    func validate(weights: [String: MLXArray], metadata: [String: String]) throws {
        // Only the MLX-native format, with no alternate metadata conventions,
        // is admitted. Exact captured key sets reject remapped aliases, router
        // quantization companions and any otherwise unsupported raw tensor.
        guard metadata == ["format": "mlx"] else { throw ValidationError("unsupported metadata") }
        var residentTensors: [String: Qwen35AffineResidentContract.Tensor] = [:]
        var routedTensors: [String: Qwen35AffineRoutedContract.Tensor] = [:]
        for (key, array) in weights {
            let dtype: String
            switch array.dtype {
            case .uint32: dtype = "U32"
            case .bfloat16: dtype = "BF16"
            case .float16: dtype = "F16"
            case .float32: dtype = "F32"
            default: throw ValidationError("unsupported dtype: \(key)")
            }
            if routedKeys.contains(key) {
                routedTensors[key] = .init(shape: array.shape, dtype: dtype)
            } else {
                residentTensors[key] = .init(shape: array.shape, dtype: dtype)
            }
        }
        try resident.validate(residentTensors)
        try routed.validate(routedTensors)
        guard weights.allSatisfy({ key, value in
            !key.contains("conv1d.weight") || (value.ndim == 3 && value.dim(-1) == 1)
        }) else { throw ValidationError("non-native convolution layout") }
        guard !NormConventionResolver.shouldApplyPlusOneShift(
            metadataConvention: nil, configConvention: normConvention, declaredConvention: nil,
            weights: weights, probeSuffixes: [".input_layernorm.weight", ".post_attention_layernorm.weight"],
            fallbackWhenNoProbe: { false }) else {
            throw ValidationError("checkpoint requires norm transformation")
        }
    }

    /// Reads each selected shard and validates its own metadata and all keys.
    /// Files must remain trusted and unchanged throughout and after this call.
    func validateSelectedFiles() throws {
        var arrays: [String: MLXArray] = [:]
        for name in shards {
            let (shard, metadata) = try MLX.loadArraysAndMetadata(url: directory.appendingPathComponent(name))
            guard metadata == ["format": "mlx"] else { throw ValidationError("unsupported shard metadata: \(name)") }
            for (key, value) in shard {
                guard arrays.updateValue(value, forKey: key) == nil else {
                    throw ValidationError("duplicate tensor across shards: \(key)")
                }
            }
        }
        try validate(weights: arrays, metadata: ["format": "mlx"])
    }
}
