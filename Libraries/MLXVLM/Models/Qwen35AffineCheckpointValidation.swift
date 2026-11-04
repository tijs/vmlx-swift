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
    private let geometry: (layers: Int, experts: Int, input: Int, hidden: Int)
    private let quantizedGates: Set<String>

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
            let properties = try directory.appendingPathComponent(name).resourceValues(forKeys: [.fileSizeKey])
            guard let size = properties.fileSize, size >= 0, size <= 64 * 1024 * 1024 else {
                throw ValidationError("metadata file exceeds size bound: \(name)")
            }
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
        var mixedOverrides: Set<String> = []
        if metadataHashes["config.json"] != nil {
            let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
            let decoded = try JSONDecoder().decode(Qwen35Configuration.self, from: data)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            guard try encoder.encode(decoded) == encoder.encode(model.config) else {
                throw ValidationError("checkpoint config differs from model")
            }
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                let quant = (json["quantization"] ?? json["quantization_config"]) as? [String: Any]
            else { throw ValidationError("unsupported config quantization declaration") }
            if json["quantization"] != nil && json["quantization_config"] != nil {
                guard let left = json["quantization"] as? [String: Any],
                    let right = json["quantization_config"] as? [String: Any],
                    (left as NSDictionary).isEqual(right as NSDictionary) else {
                    throw ValidationError("conflicting config quantization declarations")
                }
            }
            let defaults = quant.filter { ["bits", "group_size", "mode"].contains($0.key) }
            let quantData = try JSONSerialization.data(withJSONObject: defaults)
            let declared = try JSONDecoder().decode(BaseConfiguration.Quantization.self, from: quantData)
            guard declared.bits == bits, declared.groupSize == groupSize, declared.mode == .affine else {
                throw ValidationError("config quantization differs from pinned policy")
            }
            mixedOverrides = Set(quant.keys).subtracting(["bits", "group_size", "mode"])
            if !mixedOverrides.isEmpty {
                var expected = Set<String>()
                for layer in 0..<text.hiddenLayers {
                    expected.insert("language_model.model.layers.\(layer).mlp.gate")
                    expected.insert("language_model.model.layers.\(layer).mlp.shared_expert_gate")
                }
                guard bits == 4, groupSize == 64, mixedOverrides == expected else {
                    throw ValidationError("unsupported mixed affine policy coverage")
                }
                for path in expected {
                    guard let override = quant[path] as? [String: Any],
                        Set(override.keys).isSubset(of: ["bits", "group_size", "mode"]) else {
                        throw ValidationError("unsupported affine override: \(path)")
                    }
                    let overrideData = try JSONSerialization.data(withJSONObject: override)
                    let value = try JSONDecoder().decode(BaseConfiguration.Quantization.self, from: overrideData)
                    guard value.bits == 8, value.groupSize == 64, value.mode == .affine else {
                        throw ValidationError("unsupported affine override: \(path)")
                    }
                }
            }
        }
        var inspection: Qwen35AffineResearchManifest.Inspection?
        if metadataHashes["model.safetensors.index.json"] != nil {
            inspection = try Qwen35AffineResearchManifest.inspect(modelDirectory: directory,
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
        self.geometry = (text.hiddenLayers, text.numExperts, text.hiddenSize, text.moeIntermediateSize)
        self.quantizedGates = mixedOverrides
        var residentPolicies: [String: Qwen35AffineResidentContract.Affine] = [:]
        if !mixedOverrides.isEmpty {
            guard let inspection else { throw ValidationError("mixed policy requires indexed descriptors") }
            for key in inspection.tensors.keys where key.hasSuffix(".scales") && !names.contains(key) {
                let path = String(key.dropLast(".scales".count))
                residentPolicies[path] = .init(bits: mixedOverrides.contains(path) ? 8 : bits, groupSize: groupSize)
            }
            guard mixedOverrides.isSubset(of: Set(residentPolicies.keys)) else {
                throw ValidationError("missing quantized router or shared expert gate")
            }
        }
        self.resident = try Qwen35AffineResidentContract.capture(model: model, affine: residentPolicies,
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
        for path in quantizedGates {
            guard residentTensors[path + ".weight"]?.dtype == "U32",
                residentTensors[path + ".scales"]?.dtype == "BF16",
                residentTensors[path + ".biases"]?.dtype == "BF16" else {
                throw ValidationError("unsupported quantized gate dtype: \(path)")
            }
        }
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

    enum DescriptorStatus { case requiresNormValueCheck }

    /// Header-only, bounded inspection. It does not establish that norms need
    /// no transform: that requires independently bounded value reads. It must
    /// never be treated as model admission.
    func validateSelectedDescriptors() throws -> DescriptorStatus {
        _ = try checkedSelectedDescriptors(shouldCancel: { false })
        return .requiresNormValueCheck
    }

    private func checkedSelectedDescriptors(shouldCancel: () -> Bool) throws -> Qwen35AffineResearchManifest.Inspection {
        let manifest = try Qwen35AffineResearchManifest.inspect(modelDirectory: directory,
            selectedShards: shards.map { directory.appendingPathComponent($0) },
            layerCount: geometry.layers, numExperts: geometry.experts,
            inputDims: geometry.input, hiddenDims: geometry.hidden, shouldCancel: shouldCancel)
        guard manifest.metadata.values.allSatisfy({ $0 == ["format": "mlx"] }) else {
            throw ValidationError("unsupported shard metadata")
        }
        var residentTensors: [String: Qwen35AffineResidentContract.Tensor] = [:]
        var routedTensors: [String: Qwen35AffineRoutedContract.Tensor] = [:]
        for (key, value) in manifest.tensors {
            if shouldCancel() || Task<Never, Never>.isCancelled { throw CancellationError() }
            if routedKeys.contains(key) {
                routedTensors[key] = .init(shape: value.shape, dtype: value.dtype)
            } else {
                residentTensors[key] = .init(shape: value.shape, dtype: value.dtype)
            }
            if key.contains("conv1d.weight"),
                value.shape.count != 3 || value.shape.last != 1 {
                throw ValidationError("non-native convolution layout")
            }
        }
        try resident.validate(residentTensors)
        try routed.validate(routedTensors)
        for path in quantizedGates {
            guard residentTensors[path + ".weight"]?.dtype == "U32",
                residentTensors[path + ".scales"]?.dtype == "BF16",
                residentTensors[path + ".biases"]?.dtype == "BF16" else {
                throw ValidationError("unsupported quantized gate dtype: \(path)")
            }
        }
        return manifest
    }

    /// CPU-only bounded value reads, after descriptor validation. This remains
    /// research evidence, not a loader admission or file-immutability proof.
    func requiresNormShift(shouldCancel: () -> Bool = { false }) throws -> Bool {
        let manifest = try checkedSelectedDescriptors(shouldCancel: shouldCancel)
        return try Qwen35NormHeaderCheck.requiresShift(inspection: manifest,
            modelDirectory: directory, configConvention: normConvention, shouldCancel: shouldCancel)
    }
}
