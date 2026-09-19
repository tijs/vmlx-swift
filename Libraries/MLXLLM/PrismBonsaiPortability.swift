// Copyright © 2026

import Foundation
import MLXLMCommon
import MLXNN

/// Bonsai 2 Prism-Hadamard portability gate (vmlx LLM load path only).
///
/// Target pack: `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` at immutable revision
/// `3f926b415992eaa2ae9dd7b573706494d6bbf787`, `model_type`
/// `prism_hadamard_qwen35` (`base_model_type` `qwen3_5`, text decoder
/// `qwen3_5_text`). The pack folds a blockwise Hadamard rotation (block 1024,
/// ±1 signs) into its stored weights; matching activation + inverse-embedding
/// transforms must run at forward time (bundled `runtime/runtime.py`
/// `fwht`/`Packed`). This pin has no Swift equivalent for those transforms, so
/// the pack must NEVER be loaded through the generic registry or the
/// `text_config.model_type` fallback — the fallback builds a
/// default-parameter `Qwen35TextModel` and fails late and ambiguously
/// (wrong-model trap).
///
/// This gate makes the decision deterministic and explicit:
///
/// - OFF (default): `prism_hadamard_qwen35` in `model_type` (root or
///   `text_config.model_type`) → clean `unsupportedModelType` reject. The
///   text_config fallback is never attempted for this type, and no ordinary
///   model is ever constructed for it.
/// - ON: the pack must pass strict manifest + `hadamard.json` sign validation
///   (see `decide(configData:hadamardData:gateEnabled:)`); every validation
///   failure aborts early with a configuration error. A VALID pack enters the
///   isolated transform load path only when the root `model_type` matches the
///   pinned contract (`prism_hadamard_qwen35` with the `qwen3_5_text` text
///   decoder and a root config that self-describes the decoder size); the
///   `qwen3_5_text` decoder is constructed explicitly and the validated plan
///   is handed to `loadWeights(bonsaiTransform:)`, which installs the packed
///   transform modules and consumes the packed/signs keys transactionally
///   before the final `update(verify: [.noUnusedKeys])`.
///
/// The env gate follows the repo convention of `DSV4_FORCE_JANGTQ` and the
/// `VMLX_*` gates (process environment, exact string `"1"`): it is local to
/// the vmlx load path, default OFF, uses no compile flag, and changes no
/// public API default. Enabling it never makes an invalid/missing Bonsai
/// manifest load — it only changes WHERE the load fails (early, with a clear
/// reason).
///
/// SEAM STATUS (transform-slice commits):
/// 1. `MLXNN`: `HadamardActivation` + forward-FWHT packed Linear
///    (`HadamardPackedLinear`) and inverse-FWHT packed Embedding
///    (`HadamardPackedEmbedding`) over `MLX.hadamardTransform`
///    (`Source/MLX/Ops.swift`), mirroring the pack runtime
///    `runtime/runtime.py:16-70` semantics clean-room
///    (`Source/MLXNN/PrismBonsaiHadamard.swift`). DONE + pinned-runtime
///    FWHT parity gate (`PrismBonsaiHadamardPinnedParityTests`,
///    `tools/BonsaiPrismFWHTParity`).
/// 2. `MLXLMCommon/Load.swift`: `loadWeights(..., bonsaiTransform:)` installs
///    those modules via the affine-quantize module-swap machinery when a
///    validated plan is passed, consumes the per-module `.signs` tensors, and
///    bypasses the standard affine `QuantizedLinear` swap for packed paths so
///    the final `update(verify: [.noUnusedKeys])` passes. DONE.
/// 3. `LLMModelFactory` gate-on-manifest-valid wiring: the validated plan is
///    built in `_load` and handed to `loadWeights(bonsaiTransform:)` with an
///    explicitly constructed `Qwen35TextModel` — never the registry, never
///    the `text_config.model_type` fallback. DONE (loader/factory wiring only;
///    a real-pack load/benchmark is a later identity/parity gate).
enum PrismBonsaiPortability {

    /// Root model_type of the Bonsai 2 pack (config.json `model_type`).
    static let prismHadamardQwen35 = "prism_hadamard_qwen35"

    /// Nested text decoder of the pinned pack (config.json
    /// `text_config.model_type`). The transform load path constructs this
    /// decoder explicitly; any other nested value (or none) fails closed.
    static let requiredTextDecoderModelType = "qwen3_5_text"

    /// Environment variable that turns the gate ON. Absent or not exactly
    /// `"1"` → OFF.
    static let environmentVariableName = "VMLX_BONSAI_PRISM_HADAMARD"

    /// Default-off, load-path-local feature gate.
    static var isEnabled: Bool {
        ProcessInfo.processInfo.environment[environmentVariableName] == "1"
    }

    // MARK: - Manifest contract (verified against the pinned revision)

    /// Runtime stamp the packed modules require (config `requires_runtime`).
    static let requiredRuntime = "runtime/artifact.py"
    /// Config key naming the sign manifest (config `hadamard_config`).
    static let requiredHadamardConfigFilename = "hadamard.json"
    /// `schema_version` stamped by the artifact pipeline (config).
    static let requiredSchemaVersion = 2
    /// Base architecture (config `base_model_type`).
    static let requiredBaseModelType = "qwen3_5"
    /// Config quantization container; byte layout matches MLX affine 2-bit.
    static let requiredBits = 2
    static let requiredGroupSize = 128
    static let requiredQuantizationMode = "affine"
    /// Required GDN activation layout (config `gdn_activation_layout`).
    static let requiredGDNActivationLayout = "grouped"
    /// Allowed Hadamard block sizes (runtime `fwht` guard).
    static let allowedBlockSizes: Set<Int> = [512, 1024, 2048, 4096]
    /// Sign values must be exactly ±1.
    static let allowedSignValues: Set<Double> = [-1, 1]
    /// Hadamard manifest version (hadamard.json `prism.hadamard.version`).
    static let requiredHadamardVersion = 1
    /// Required forward transform (hadamard.json `prism.hadamard.transform`).
    static let requiredHadamardTransform = "normalized-sylvester-walsh-hadamard"
    /// Required sign mode (hadamard.json `prism.hadamard.sign_mode`).
    static let requiredHadamardSignMode = "explicit"
    /// Namespaced keys of hadamard.json at the pinned revision.
    static let hadamardVersionKey = "prism.hadamard.version"
    static let hadamardBlockSizeKey = "prism.hadamard.block_size"
    static let hadamardTransformKey = "prism.hadamard.transform"
    static let hadamardSignModeKey = "prism.hadamard.sign_mode"
    static let hadamardWeightNamesKey = "prism.hadamard.weight_names"
    static let hadamardInverseWeightNamesKey = "prism.hadamard.inverse_weight_names"
    static let hadamardSignWidthsKey = "prism.hadamard.sign_widths"
    static let hadamardSignValuesKey = "prism.hadamard.sign_values"
    static let hadamardGDNVGroupedKey = "prism.hadamard.gdn_v_grouped"

    // MARK: - Decision

    enum Decision: Equatable {
        /// Neither root nor text_config model_type is `prism_hadamard_qwen35`:
        /// the ordinary load path (including the text_config fallback) is
        /// unchanged.
        case notBonsai
        /// Prism-Hadamard identity detected and the gate is OFF: reject with
        /// `unsupportedModelType`, never the text_config fallback.
        case gateOffReject
        /// Gate ON and the pack manifest + hadamard signs validate. The
        /// transformed load path (MLXNN packed modules + the
        /// `MLXLMCommon/Load.swift` `bonsaiTransform:` seam + the factory
        /// handoff) is wired: the factory builds the validated plan and
        /// hands it to `loadWeights(bonsaiTransform:)` alongside an
        /// explicitly constructed `qwen3_5_text` decoder. Identity reached
        /// only through nested `text_config` (VLM-wrapped shape) is refused
        /// by the factory with a clear error — it is not this factory's
        /// load to absorb.
        case gateOnManifestValid
        /// Gate ON and validation failed; the attached error is the reason.
        case gateOnManifestInvalid(ManifestValidationError)
    }

    /// Structured validation failure; `errorDescription` is operator-facing.
    struct ManifestValidationError: LocalizedError, Equatable {
        let reason: String
        init(_ reason: String) {
            self.reason = reason
        }

        var errorDescription: String? {
            "\(prismHadamardQwen35): \(reason)"
        }
    }

    // MARK: - Config probes

    private struct ConfigProbe: Codable {
        let modelType: String?
        let schemaVersion: Int?
        let requiresRuntime: String?
        let hadamardConfig: String?
        let tensorNamespace: String?
        let gdnActivationLayout: String?
        let baseModelType: String?
        let quantization: QuantizationProbe?
        let modules: [ModuleProbe]?
        let textConfig: TextConfigProbe?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case schemaVersion = "schema_version"
            case requiresRuntime = "requires_runtime"
            case hadamardConfig = "hadamard_config"
            case tensorNamespace = "tensor_namespace"
            case gdnActivationLayout = "gdn_activation_layout"
            case baseModelType = "base_model_type"
            case quantization
            case modules
            case textConfig = "text_config"
        }
    }

    private struct QuantizationProbe: Codable {
        let bits: Int?
        let groupSize: Int?
        let mode: String?

        enum CodingKeys: String, CodingKey {
            case bits
            case groupSize = "group_size"
            case mode
        }
    }

    private struct ModuleProbe: Codable {
        let path: String?
        let block: Int?
        let embedding: Bool?

        enum CodingKeys: String, CodingKey {
            case path
            case block
            case embedding
        }
    }

    private struct TextConfigProbe: Codable {
        let modelType: String?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
        }
    }

    // MARK: - Decision logic

    /// Pure, deterministic decision over a config.json payload and an optional
    /// hadamard.json payload. `gateEnabled` is passed in explicitly so tests
    /// exercise both arms without touching the process environment.
    ///
    /// - Important: `.notBonsai` is returned when the identity cannot be
    ///   established (nil data or unreadable JSON) — the generic loader's own
    ///   decode error then handles the malformed config as it always has.
    static func decide(
        configData: Data?,
        hadamardData: Data?,
        gateEnabled: Bool
    ) -> Decision {
        guard let configData,
            let object = (try? JSONSerialization.jsonObject(with: configData))
                as? [String: Any]
        else {
            return .notBonsai
        }
        let rootType = object["model_type"] as? String
        let textType = (object["text_config"] as? [String: Any])?["model_type"] as? String
        let isPrism = rootType == prismHadamardQwen35 || textType == prismHadamardQwen35
        guard isPrism else { return .notBonsai }
        guard gateEnabled else { return .gateOffReject }

        // Gate ON: strict manifest + sign validation, first failure wins.
        guard let probe = try? JSONDecoder.json5().decode(ConfigProbe.self, from: configData)
        else {
            return invalid("config.json did not decode as a Bonsai manifest")
        }
        guard probe.schemaVersion == requiredSchemaVersion else {
            return invalid(
                "schema_version must be \(requiredSchemaVersion), got "
                    + "\(probe.schemaVersion.map { "\($0)" } ?? "nil")")
        }
        guard probe.requiresRuntime == requiredRuntime else {
            return invalid(
                "requires_runtime must be \"\(requiredRuntime)\", got "
                    + "\(probe.requiresRuntime.map { "\"\($0)\"" } ?? "nil")")
        }
        guard probe.hadamardConfig == requiredHadamardConfigFilename else {
            return invalid(
                "hadamard_config must be \"\(requiredHadamardConfigFilename)\", got "
                    + "\(probe.hadamardConfig.map { "\"\($0)\"" } ?? "nil")")
        }
        guard probe.baseModelType == requiredBaseModelType else {
            return invalid(
                "base_model_type must be \"\(requiredBaseModelType)\", got "
                    + "\(probe.baseModelType.map { "\"\($0)\"" } ?? "nil")")
        }
        // Pinned text-decoder contract: when the ROOT model_type is the
        // prism pack, the nested text_config.model_type must be exactly
        // `qwen3_5_text`. A different or missing decoder (e.g.
        // `qwen3_5_moe`, or a plain `qwen3_5_text` at the root with a
        // prism nested value) would either fail late in the transform path
        // or route to the wrong architecture — fail closed here instead.
        if rootType == prismHadamardQwen35, textType != requiredTextDecoderModelType {
            return invalid(
                "text_config.model_type must be \"\(requiredTextDecoderModelType)\" "
                    + "when model_type is \"\(prismHadamardQwen35)\", got "
                    + "\(textType.map { "\"\($0)\"" } ?? "nil")")
        }
        guard let tensorNamespace = probe.tensorNamespace, !tensorNamespace.isEmpty else {
            return invalid("tensor_namespace must be present and non-empty")
        }
        guard probe.gdnActivationLayout == requiredGDNActivationLayout else {
            return invalid(
                "gdn_activation_layout must be \"\(requiredGDNActivationLayout)\", got "
                    + "\(probe.gdnActivationLayout.map { "\"\($0)\"" } ?? "nil")")
        }
        guard let quantization = probe.quantization else {
            return invalid("quantization manifest is missing")
        }
        guard quantization.bits == requiredBits else {
            return invalid(
                "quantization.bits must be \(requiredBits), got "
                    + "\(quantization.bits.map { "\($0)" } ?? "nil")")
        }
        guard quantization.groupSize == requiredGroupSize else {
            return invalid(
                "quantization.group_size must be \(requiredGroupSize), got "
                    + "\(quantization.groupSize.map { "\($0)" } ?? "nil")")
        }
        guard quantization.mode == requiredQuantizationMode else {
            return invalid(
                "quantization.mode must be \"\(requiredQuantizationMode)\", got "
                    + "\(quantization.mode.map { "\"\($0)\"" } ?? "nil")")
        }
        guard let modules = probe.modules, !modules.isEmpty else {
            return invalid("modules manifest is missing or empty")
        }
        guard modules.allSatisfy({ !($0.path ?? "").isEmpty }) else {
            return invalid("every modules[] entry must have a non-empty path")
        }
        guard modules.allSatisfy({ $0.block.map(allowedBlockSizes.contains) ?? false }) else {
            return invalid(
                "every modules[] entry must have a block in \(allowedBlockSizes.sorted())")
        }
        guard let commonBlock = modules.first?.block,
            modules.allSatisfy({ $0.block == commonBlock })
        else {
            return invalid("all modules[] entries must share the same block size")
        }
        guard modules.contains(where: { $0.embedding == true }) else {
            return invalid("modules[] must contain an embedding module")
        }
        let modulePaths = modules.compactMap { $0.path }
        guard Set(modulePaths).count == modulePaths.count else {
            return invalid("modules[] paths must be unique")
        }
        guard modules.filter({ $0.embedding == true }).count == 1 else {
            return invalid("modules[] must contain exactly one embedding module")
        }

        // hadamard.json — the sign manifest.
        guard let hadamardData,
            let hadamard = (try? JSONSerialization.jsonObject(with: hadamardData))
                as? [String: Any]
        else {
            return invalid(
                "\(requiredHadamardConfigFilename) is missing or not JSON")
        }
        guard let version = (hadamard[hadamardVersionKey] as? NSNumber)?.intValue,
            version == requiredHadamardVersion
        else {
            return invalid(
                "\(hadamardVersionKey) must be \(requiredHadamardVersion), got "
                    + "\(String(describing: hadamard[hadamardVersionKey]))")
        }
        guard let blockSize = (hadamard[hadamardBlockSizeKey] as? NSNumber)?.intValue,
            allowedBlockSizes.contains(blockSize), blockSize == commonBlock
        else {
            return invalid(
                "\(hadamardBlockSizeKey) must match the modules[] block size "
                    + "(\(commonBlock)), got \(String(describing: hadamard[hadamardBlockSizeKey]))")
        }
        guard hadamard[hadamardTransformKey] as? String == requiredHadamardTransform else {
            return invalid(
                "\(hadamardTransformKey) must be \"\(requiredHadamardTransform)\", got "
                    + "\(String(describing: hadamard[hadamardTransformKey]))")
        }
        guard hadamard[hadamardSignModeKey] as? String == requiredHadamardSignMode else {
            return invalid(
                "\(hadamardSignModeKey) must be \"\(requiredHadamardSignMode)\", got "
                    + "\(String(describing: hadamard[hadamardSignModeKey]))")
        }
        guard let weightNames = hadamard[hadamardWeightNamesKey] as? [String],
            !weightNames.isEmpty
        else {
            return invalid("\(hadamardWeightNamesKey) must be non-empty")
        }
        guard let inverseWeightNames = hadamard[hadamardInverseWeightNamesKey] as? [String],
            !inverseWeightNames.isEmpty
        else {
            return invalid("\(hadamardInverseWeightNamesKey) must be non-empty")
        }
        // Mirror of the plan builder: the forward and inverse weight-name lists
        // must be disjoint (a folded tensor is either forward or inverse,
        // never both). The two lists use the pack's own checkpoint key
        // naming, which is deliberately NOT cross-checked against modules[]
        // paths — the pinned pack's modules[] paths and hadamard.json
        // weight_names use different namespaces/names for the same modules.
        let foldedBases = Set(
            weightNames.map(PrismBonsaiHadamardPlan.normalizedModuleBase))
        let inverseBases = Set(
            inverseWeightNames.map(PrismBonsaiHadamardPlan.normalizedModuleBase))
        guard foldedBases.isDisjoint(with: inverseBases) else {
            return invalid(
                hadamardWeightNamesKey + " and " + hadamardInverseWeightNamesKey
                    + " must not overlap")
        }
        let signWidths = (hadamard[hadamardSignWidthsKey] as? [Any])?
            .compactMap { ($0 as? NSNumber)?.intValue } ?? []
        let signValues = (hadamard[hadamardSignValuesKey] as? [Any])?
            .compactMap { ($0 as? NSNumber)?.doubleValue } ?? []
        guard !signWidths.isEmpty else {
            return invalid("\(hadamardSignWidthsKey) must be non-empty")
        }
        guard signWidths.allSatisfy({ $0 >= 1 }) else {
            return invalid(
                hadamardSignWidthsKey + " must contain only positive widths")
        }
        guard !signValues.isEmpty else {
            return invalid("\(hadamardSignValuesKey) must be non-empty")
        }
        guard signWidths.reduce(0, +) == signValues.count else {
            return invalid(
                "\(hadamardSignWidthsKey) must sum to the \(hadamardSignValuesKey) count "
                    + "(\(signWidths.reduce(0, +)) != \(signValues.count))")
        }
        guard signValues.allSatisfy({ allowedSignValues.contains($0) }) else {
            return invalid(
                "\(hadamardSignValuesKey) must contain only ±1 values")
        }
        guard (hadamard[hadamardGDNVGroupedKey] as? Bool) == true else {
            return invalid("\(hadamardGDNVGroupedKey) must be true")
        }
        return .gateOnManifestValid
    }

    private static func invalid(_ reason: String) -> Decision {
        .gateOnManifestInvalid(ManifestValidationError(reason))
    }

    /// True only when the prism identity is carried by the ROOT `model_type`
    /// — the pinned, LLM-factory-loadable shape. A prism identity that exists
    /// only under `text_config.model_type` is a VLM-wrapped pack that must
    /// route through the VLM factory; the LLM transform load refuses it.
    static func isRootPrismIdentity(rootModelType: String?) -> Bool {
        rootModelType == prismHadamardQwen35
    }

    // MARK: - Text-decoder size probe

    /// Probe the pack's ROOT config for the `qwen3_5_text` decoder's core
    /// size parameters.
    ///
    /// `Qwen35TextConfiguration` defaults every field when its key is absent
    /// (`hidden_size` → 4096, `num_hidden_layers` → 32, `vocab_size` →
    /// 151936), so decoding a bare Bonsai manifest (which carries only the
    /// prism contract fields) would silently construct a default-parameter
    /// `Qwen35TextModel` — the exact wrong-model trap the gate exists to
    /// close. The factory therefore requires the pack's ROOT config to
    /// explicitly carry positive `hidden_size`, `num_hidden_layers` and
    /// `vocab_size` before any decoder is constructed; absence (or a
    /// `text_config`-nested layout) fails closed with a configuration error
    /// instead of defaulting.
    static func textDecoderRootSizeParameters(
        configData: Data
    ) -> (hiddenSize: Int, hiddenLayers: Int, vocabSize: Int)? {
        struct Probe: Codable {
            let hiddenSize: Int?
            let hiddenLayers: Int?
            let vocabSize: Int?

            enum CodingKeys: String, CodingKey {
                case hiddenSize = "hidden_size"
                case hiddenLayers = "num_hidden_layers"
                case vocabSize = "vocab_size"
            }
        }
        guard let probe = try? JSONDecoder.json5().decode(Probe.self, from: configData),
            let hiddenSize = probe.hiddenSize, hiddenSize > 0,
            let hiddenLayers = probe.hiddenLayers, hiddenLayers > 0,
            let vocabSize = probe.vocabSize, vocabSize > 0
        else {
            return nil
        }
        return (hiddenSize, hiddenLayers, vocabSize)
    }
}
