// Copyright © 2026
//
// Bonsai 2 Prism-Hadamard transform modules (clean-room Swift/MLX portability
// slice for prism-ml/Ternary-Bonsai-2-27B-mlx-2bit @
// 3f926b415992eaa2ae9dd7b573706494d6bbf787, model_type
// prism_hadamard_qwen35).
//
// The pack folds a blockwise normalized Sylvester/Walsh-Hadamard rotation
// (block 1024, explicit ±1 per-module signs) into its stored 2-bit affine
// group-128 weights. At forward time the matching transform must run on
// activations: a forward FWHT before every packed projection (q/k/v/o,
// mlp gate/up/down, lm_head) and an inverse FWHT after the packed
// embedding lookup (`embed_tokens` rows are stored in the rotated basis).
//
// These types implement the runtime contract (pack `runtime/runtime.py`
// `fwht`/`Packed`, and `runtime/artifact.py` per-module `.signs` tensors)
// as clean-room semantics over the repository's own MLX primitives:
// `hadamardTransform(_:scale:)` (Source/MLX/Ops.swift), `quantizedMM`,
// `dequantized`. Nothing here copies the bundled Python runtime.
//
// The feature gate (`VMLX_BONSAI_PRISM_HADAMARD=1`, default OFF) and the
// strict manifest/hadamard.json validation live in
// `Libraries/MLXLLM/PrismBonsaiPortability.swift`; this file provides the
// transform math, the packed module types, the validated plan consumed by
// the `MLXLMCommon/Load.swift` install seam, and the pure install-plan
// resolution used to select modules and consume the per-module `.signs`
// tensors.

import Foundation
import MLX

// MARK: - FWHT / Hadamard activation

/// Blockwise normalized Sylvester/Walsh-Hadamard transform with explicit ±1
/// signs, matching the packed-pack activation contract:
///
///     forward: y = H_blockwise(x_f32 * signs)   scale = 1/sqrt(block)
///     inverse: y = H_blockwise(x_f32) * signs   scale = 1/sqrt(block)
///
/// Applied along the final activation axis; the block must divide the width.
/// Signs broadcast over the leading dimensions (one sign per activation
/// column, length == width). Staging is float32, the output is cast back to
/// the input dtype — mirroring the pack runtime `fwht` semantics.
///
/// - Parameters:
///   - x: activations, any leading shape, last dim divisible by `block`
///   - block: Hadamard block size (the pinned pack uses 1024)
///   - signs: explicit ±1 sign vector of length `x.dim(-1)` (required)
///   - inverse: `false` = forward transform (packed linear / LM-head path),
///     `true` = inverse transform (packed embedding path)
public func hadamardFWHT(
    _ x: MLXArray, block: Int, signs: MLXArray, inverse: Bool = false,
    stream: StreamOrDevice = .default
) -> MLXArray {
    precondition(block > 0, "hadamardFWHT: block must be positive")
    let width = x.shape[x.shape.count - 1]
    precondition(
        width % block == 0,
        "hadamardFWHT: block \(block) does not divide activation width \(width)")

    let shape = x.shape
    var y = x.asType(.float32)
    if !inverse {
        y = y * signs.asType(.float32)
    }
    y = hadamardTransform(
        y.reshaped([-1, block]), scale: 1 / sqrt(Float(block)), stream: stream)
        .reshaped(shape)
    if inverse {
        y = y * signs.asType(.float32)
    }
    return y.asType(x.dtype)
}

/// Forward-FWHT activation module: applies
/// ``hadamardFWHT(_:block:signs:inverse:)`` with `inverse == false` before a
/// packed projection. Also exposes `inverse(_:)` for the embedding path.
///
/// This is the named "Hadamard activation" building block of the portability
/// seam; the packed modules below use it internally at forward time.
open class HadamardActivation: Module, UnaryLayer {

    /// Hadamard block size (0 disables the transform — identity).
    public let block: Int
    /// Explicit ±1 sign vector, length == activation width.
    public let signs: MLXArray?

    /// - Parameters:
    ///   - block: Hadamard block size; 0 disables the transform
    ///   - signs: ±1 sign vector, length == activation width (nil when
    ///     `block == 0`)
    public init(block: Int, signs: MLXArray?) {
        self.block = block
        self.signs = signs
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard block > 0, let signs else { return x }
        return hadamardFWHT(x, block: block, signs: signs, inverse: false)
    }

    /// Inverse transform (packed embedding path).
    public func inverse(_ x: MLXArray) -> MLXArray {
        guard block > 0, let signs else { return x }
        return hadamardFWHT(x, block: block, signs: signs, inverse: true)
    }
}

// MARK: - Packed modules

/// Forward-FWHT packed linear: applies the Hadamard transform (+ per-module
/// signs) to the activations and then runs the standard MLX affine
/// `quantizedMM`. Storage layout is byte-identical to `QuantizedLinear`
/// (2-bit / group-128 for this pack); only the activation transform differs.
open class HadamardPackedLinear: Linear, Quantized {

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    /// Hadamard block size (0 disables the transform).
    public let block: Int
    /// Explicit ±1 sign vector; required when `block > 0`.
    public let signs: MLXArray?

    @ParameterInfo(key: "scales") public var scales: MLXArray
    @ParameterInfo(key: "biases") public var biases: MLXArray?

    open override var shape: (Int, Int) {
        let shape = weight.shape2
        return (shape.0, shape.1 * 32 / bits)
    }

    /// Init from already-packed checkpoint tensors plus the transform
    /// contract. `weight`/`scales`/`biases` are the pack's affine 2-bit
    /// arrays; `signs` is the per-module `.signs` vector.
    public init(
        weight: MLXArray, bias: MLXArray? = nil,
        scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int,
        block: Int, signs: MLXArray?,
        mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.block = block
        self.signs = signs
        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases
        super.init(weight: weight, bias: bias)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let inputDType = x.dtype
        var x = x
        if block > 0 {
            guard let signs else {
                preconditionFailure(
                    "HadamardPackedLinear with block \(block) requires signs")
            }
            x = hadamardFWHT(x, block: block, signs: signs, inverse: false)
        }
        var out = quantizedMM(
            x, weight, scales: scales, biases: biases, transpose: true,
            groupSize: groupSize, bits: bits, mode: mode)
        if let bias {
            out = out + bias
        }
        // Same output-dtype pin as QuantizedLinear: quantizedMM types its
        // output as promote_types(x, scales); f16/bf16 activations against
        // f16 scales promote to float32, which must not propagate model-wide.
        if !QuantizedRuntime.propagatePromotedOutput, out.dtype != inputDType,
            inputDType == .bfloat16 || inputDType == .float16
        {
            out = out.asType(inputDType)
        }
        return out
    }
}

/// Inverse-FWHT packed embedding: dequantized `embed_tokens` rows are stored
/// in the rotated basis and must be inverse-Hadamard-transformed after the
/// lookup. Storage layout matches `QuantizedEmbedding`.
open class HadamardPackedEmbedding: QuantizedEmbedding {

    /// Hadamard block size (0 disables the transform).
    public let block: Int
    /// Explicit ±1 sign vector; required when `block > 0`.
    public let signs: MLXArray?

    /// Init from already-packed checkpoint tensors plus the transform
    /// contract.
    public init(
        weight: MLXArray,
        scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int,
        block: Int, signs: MLXArray?
    ) {
        self.block = block
        self.signs = signs
        super.init(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, mode: .affine)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = super.callAsFunction(x)
        if block > 0 {
            guard let signs else {
                preconditionFailure(
                    "HadamardPackedEmbedding with block \(block) requires signs")
            }
            out = hadamardFWHT(out, block: block, signs: signs, inverse: true)
        }
        return out
    }

    // NB: `asLinear` (tied-head) semantics are intentionally NOT overridden.
    // A tied output head on a rotated-basis embedding is not defined without
    // materializing the true table; the pinned pack ships an explicit packed
    // `lm_head` and declares `tie_word_embeddings: false`. Tied-head handling
    // for this basis is a parity-gate concern, not a silent convenience.
}

/// Runtime validation for packed transform tensors (load-time guard, kept
/// separate so it is exercised deterministically without a model).
public enum HadamardPackedCheck {

    /// Validate the per-module `.signs` tensor against the packed weight's
    /// input width: exactly one sign per activation column, block divides the
    /// width, and every value is exactly ±1.
    ///
    /// - Parameters:
    ///   - signs: the `.signs` tensor being installed
    ///   - packedWeightWidth: stored (packed) width of the `.weight` tensor
    ///   - bits: quantization bits (`2` for this pack)
    ///   - block: Hadamard block size
    /// - Returns: the activation input width (`packedWeightWidth * 32 / bits`)
    @discardableResult
    public static func validateSigns(
        _ signs: MLXArray, packedWeightWidth: Int, bits: Int, block: Int
    ) throws -> Int {
        let inputWidth = packedWeightWidth * 32 / bits
        guard block > 0 else {
            throw PrismBonsaiInstall.Error(
                "Hadamard block must be positive, got \(block)")
        }
        guard inputWidth % block == 0 else {
            throw PrismBonsaiInstall.Error(
                "Hadamard block \(block) does not divide activation width \(inputWidth)")
        }
        guard signs.shape.count == 1, signs.shape[0] == inputWidth else {
            throw PrismBonsaiInstall.Error(
                "signs tensor shape \(signs.shape) does not match activation width \(inputWidth)")
        }
        let isUnit = MLX.all(
            (MLX.abs(signs) .== 1), stream: .default).item(Bool.self)
        guard isUnit else {
            throw PrismBonsaiInstall.Error(
                "signs tensor must contain only ±1 values")
        }
        return inputWidth
    }

    /// Validate a packed 2-bit affine tensor group (`.weight` + `.scales` +
    /// `.biases`) against the quantization contract before any module is
    /// swapped: the packed width must unpack to a positive activation width
    /// divisible by the group size and the Hadamard block, and the affine
    /// scales/biases shapes must match the packed layout exactly.
    ///
    /// - Parameters:
    ///   - weight: the packed `.weight` tensor (2-D `[out, in / (32/bits)]`)
    ///   - scales: the affine `.scales` tensor
    ///   - biases: the affine per-group `.biases` tensor (nil when absent)
    ///   - groupSize: quantization group size (`128` for this pack)
    ///   - bits: quantization bits (`2` for this pack)
    ///   - block: Hadamard block size
    /// - Returns: the activation input width (`packedWidth * 32 / bits`)
    @discardableResult
    public static func validatePackedTensors(
        weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, block: Int
    ) throws -> Int {
        guard weight.shape.count == 2 else {
            throw PrismBonsaiInstall.Error(
                "packed weight must be 2-D, got \(weight.shape)")
        }
        let out = weight.shape[0]
        let packedWidth = weight.shape[1]
        guard packedWidth > 0 else {
            throw PrismBonsaiInstall.Error(
                "packed weight width must be positive, got \(packedWidth)")
        }
        let inputWidth = packedWidth * 32 / bits
        guard inputWidth % groupSize == 0 else {
            throw PrismBonsaiInstall.Error(
                "packed input width \(inputWidth) must be divisible by group "
                    + "size \(groupSize)")
        }
        let expectedScalesShape = [out, inputWidth / groupSize]
        guard scales.shape.count == 2,
            scales.shape[0] == expectedScalesShape[0],
            scales.shape[1] == expectedScalesShape[1]
        else {
            throw PrismBonsaiInstall.Error(
                "packed scales shape \(scales.shape) does not match weight "
                    + "\(weight.shape) at group \(groupSize) "
                    + "(expected \(expectedScalesShape))")
        }
        if let biases {
            guard biases.shape == scales.shape else {
                throw PrismBonsaiInstall.Error(
                    "packed biases shape \(biases.shape) must equal scales "
                        + "shape \(scales.shape)")
            }
        }
        guard inputWidth % block == 0 else {
            throw PrismBonsaiInstall.Error(
                "Hadamard block \(block) does not divide activation width \(inputWidth)")
        }
        return inputWidth
    }
}

// MARK: - Manifest-derived plan

/// The validated Bonsai Prism-Hadamard transform plan consumed by the
/// `Load.swift` install seam.
///
/// A plan is produced only from a fully validated config.json + hadamard.json
/// pair (strict builder below mirrors the MLXLLM gate contract so a plan can
/// never be built from a malformed pack). Entries map checkpoint tensor bases
/// (e.g. `language_model.model.embed_tokens`) to a module role and the
/// transform block.
public struct PrismBonsaiHadamardPlan: Equatable, Sendable {

    public enum Role: String, Equatable, Sendable {
        /// Forward-FWHT packed linear (q/k/v/o/out projections, mlp
        /// gate/up/down, lm_head).
        case forwardPackedLinear
        /// Inverse-FWHT packed embedding (`embed_tokens`).
        case inversePackedEmbedding
    }

    public struct Entry: Equatable, Sendable {
        /// Checkpoint tensor base as published by the pack, e.g.
        /// `language_model.model.layers.0.self_attn.q_proj`.
        public let checkpointBase: String
        public let role: Role
        /// Hadamard block (uniform across entries; 1024 for this pack).
        public let block: Int
        /// `modules[]` dtype declaration (e.g. `float16`); drives the packed
        /// embedding output dtype.
        public let dtypeName: String?

        /// Public memberwise initializer so a validated plan can also be
        /// constructed/consumed across modules (the install seam and the
        /// parity gate) and so `resolve`'s assumptions can be exercised
        /// deterministically in tests.
        public init(
            checkpointBase: String, role: Role, block: Int, dtypeName: String?
        ) {
            self.checkpointBase = checkpointBase
            self.role = role
            self.block = block
            self.dtypeName = dtypeName
        }
    }

    // MARK: Contract constants (mirrors the MLXLLM gate — see
    // PrismBonsaiPortability for the factory-side decision).

    public static let prismHadamardQwen35 = "prism_hadamard_qwen35"
    public static let requiredRuntime = "runtime/artifact.py"
    public static let requiredHadamardConfigFilename = "hadamard.json"
    public static let requiredSchemaVersion = 2
    public static let requiredBaseModelType = "qwen3_5"
    public static let requiredBits = 2
    public static let requiredGroupSize = 128
    public static let requiredQuantizationMode = "affine"
    public static let requiredGDNActivationLayout = "grouped"
    public static let allowedBlockSizes: Set<Int> = [512, 1024, 2048, 4096]
    public static let requiredHadamardVersion = 1
    public static let requiredHadamardTransform =
        "normalized-sylvester-walsh-hadamard"
    public static let requiredHadamardSignMode = "explicit"

    public static let hadamardVersionKey = "prism.hadamard.version"
    public static let hadamardBlockSizeKey = "prism.hadamard.block_size"
    public static let hadamardTransformKey = "prism.hadamard.transform"
    public static let hadamardSignModeKey = "prism.hadamard.sign_mode"
    public static let hadamardWeightNamesKey = "prism.hadamard.weight_names"
    public static let hadamardInverseWeightNamesKey =
        "prism.hadamard.inverse_weight_names"
    public static let hadamardSignWidthsKey = "prism.hadamard.sign_widths"
    public static let hadamardSignValuesKey = "prism.hadamard.sign_values"
    public static let hadamardGDNVGroupedKey = "prism.hadamard.gdn_v_grouped"

    // MARK: Plan contents

    public let block: Int
    public let groupSize: Int
    public let bits: Int
    public let entries: [Entry]
    public let foldedWeightNames: [String]
    public let inverseWeightNames: [String]
    public let signWidths: [Int]
    public let signValuesCount: Int

    public init(
        block: Int, groupSize: Int, bits: Int, entries: [Entry],
        foldedWeightNames: [String], inverseWeightNames: [String],
        signWidths: [Int], signValuesCount: Int
    ) {
        self.block = block
        self.groupSize = groupSize
        self.bits = bits
        self.entries = entries
        self.foldedWeightNames = foldedWeightNames
        self.inverseWeightNames = inverseWeightNames
        self.signWidths = signWidths
        self.signValuesCount = signValuesCount
    }

    public func entry(forCheckpointBase base: String) -> Entry? {
        entries.first { $0.checkpointBase == base }
    }

    /// Normalize a module path or checkpoint weight base for manifest
    /// comparison: strips the pack's optional `language_model.` / `model.`
    /// namespace prefixes and a trailing `.weight` suffix so the
    /// `prism.hadamard.weight_names` and `inverse_weight_names` key lists
    /// can be compared on a common base.
    public static func normalizedModuleBase(_ path: String) -> String {
        var base = path
        for prefix in ["language_model.model.", "language_model.", "model."] {
            if base.hasPrefix(prefix) {
                base = String(base.dropFirst(prefix.count))
                break
            }
        }
        if base.hasSuffix(".weight") {
            base = String(base.dropLast(".weight".count))
        }
        return base
    }

    // MARK: Strict builder

    public struct BuildError: LocalizedError, Equatable {
        public let reason: String
        public init(_ reason: String) { self.reason = reason }
        public var errorDescription: String? {
            "\(prismHadamardQwen35): \(reason)"
        }
    }

    /// Strict deterministic build from the pack's `config.json` and
    /// `hadamard.json` payloads. Every contract check mirrors the factory
    /// gate; a failure throws `BuildError` and no plan is produced.
    public init(configData: Data, hadamardData: Data) throws {
        guard let config = (try? JSONSerialization.jsonObject(with: configData))
            as? [String: Any]
        else {
            throw BuildError("config.json did not decode as a Bonsai manifest")
        }

        func intValue(_ value: Any?) -> Int? {
            (value as? NSNumber)?.intValue
        }
        func stringValue(_ value: Any?) -> String? { value as? String }

        guard stringValue(config["model_type"]) == Self.prismHadamardQwen35
            || stringValue(
                (config["text_config"] as? [String: Any])?["model_type"])
                == Self.prismHadamardQwen35
        else {
            throw BuildError("model_type must be \(Self.prismHadamardQwen35)")
        }
        guard intValue(config["schema_version"]) == Self.requiredSchemaVersion
        else {
            throw BuildError(
                "schema_version must be \(Self.requiredSchemaVersion), got "
                    + "\(String(describing: config["schema_version"]))")
        }
        guard stringValue(config["requires_runtime"]) == Self.requiredRuntime
        else {
            throw BuildError(
                "requires_runtime must be \"\(Self.requiredRuntime)\", got "
                    + "\(String(describing: config["requires_runtime"]))")
        }
        guard stringValue(config["hadamard_config"])
            == Self.requiredHadamardConfigFilename
        else {
            throw BuildError(
                "hadamard_config must be \"\(Self.requiredHadamardConfigFilename)\", got "
                    + "\(String(describing: config["hadamard_config"]))")
        }
        guard stringValue(config["base_model_type"]) == Self.requiredBaseModelType
        else {
            throw BuildError(
                "base_model_type must be \"\(Self.requiredBaseModelType)\", got "
                    + "\(String(describing: config["base_model_type"]))")
        }
        guard let tensorNamespace = stringValue(config["tensor_namespace"]),
            !tensorNamespace.isEmpty
        else {
            throw BuildError("tensor_namespace must be present and non-empty")
        }
        guard stringValue(config["gdn_activation_layout"])
            == Self.requiredGDNActivationLayout
        else {
            throw BuildError(
                "gdn_activation_layout must be \"\(Self.requiredGDNActivationLayout)\", got "
                    + "\(String(describing: config["gdn_activation_layout"]))")
        }

        guard let quant = config["quantization"] as? [String: Any] else {
            throw BuildError("quantization manifest is missing")
        }
        guard intValue(quant["bits"]) == Self.requiredBits else {
            throw BuildError(
                "quantization.bits must be \(Self.requiredBits), got "
                    + "\(String(describing: quant["bits"]))")
        }
        guard intValue(quant["group_size"]) == Self.requiredGroupSize else {
            throw BuildError(
                "quantization.group_size must be \(Self.requiredGroupSize), got "
                    + "\(String(describing: quant["group_size"]))")
        }
        guard stringValue(quant["mode"]) == Self.requiredQuantizationMode else {
            throw BuildError(
                "quantization.mode must be \"\(Self.requiredQuantizationMode)\", got "
                    + "\(String(describing: quant["mode"]))")
        }

        guard let modules = config["modules"] as? [[String: Any]],
            !modules.isEmpty
        else {
            throw BuildError("modules manifest is missing or empty")
        }
        let modulePaths = modules.compactMap { stringValue($0["path"]) }
        guard modulePaths.count == modules.count,
            modulePaths.allSatisfy({ !$0.isEmpty })
        else {
            throw BuildError("every modules[] entry must have a non-empty path")
        }
        let moduleBlocks = modules.compactMap { intValue($0["block"]) }
        guard moduleBlocks.count == modules.count,
            moduleBlocks.allSatisfy(Self.allowedBlockSizes.contains)
        else {
            throw BuildError(
                "every modules[] entry must have a block in "
                    + "\(Self.allowedBlockSizes.sorted())")
        }
        guard let commonBlock = moduleBlocks.first,
            moduleBlocks.allSatisfy({ $0 == commonBlock })
        else {
            throw BuildError("all modules[] entries must share the same block size")
        }
        guard modules.contains(where: { ($0["embedding"] as? Bool) == true })
        else {
            throw BuildError("modules[] must contain an embedding module")
        }
        guard Set(modulePaths).count == modulePaths.count else {
            throw BuildError("modules[] paths must be unique")
        }
        guard modules.filter({ ($0["embedding"] as? Bool) == true }).count == 1
        else {
            throw BuildError("modules[] must contain exactly one embedding module")
        }

        guard let hadamard =
            (try? JSONSerialization.jsonObject(with: hadamardData))
            as? [String: Any]
        else {
            throw BuildError(
                "\(Self.requiredHadamardConfigFilename) is missing or not JSON")
        }
        guard intValue(hadamard[Self.hadamardVersionKey])
            == Self.requiredHadamardVersion
        else {
            throw BuildError(
                "\(Self.hadamardVersionKey) must be "
                    + "\(Self.requiredHadamardVersion), got "
                    + "\(String(describing: hadamard[Self.hadamardVersionKey]))")
        }
        guard intValue(hadamard[Self.hadamardBlockSizeKey]) == commonBlock else {
            throw BuildError(
                "\(Self.hadamardBlockSizeKey) must match the modules[] block "
                    + "size (\(commonBlock)), got "
                    + "\(String(describing: hadamard[Self.hadamardBlockSizeKey]))")
        }
        guard stringValue(hadamard[Self.hadamardTransformKey])
            == Self.requiredHadamardTransform
        else {
            throw BuildError(
                "\(Self.hadamardTransformKey) must be \""
                    + "\(Self.requiredHadamardTransform)\", got "
                    + "\(String(describing: hadamard[Self.hadamardTransformKey]))")
        }
        guard stringValue(hadamard[Self.hadamardSignModeKey])
            == Self.requiredHadamardSignMode
        else {
            throw BuildError(
                "\(Self.hadamardSignModeKey) must be \""
                    + "\(Self.requiredHadamardSignMode)\", got "
                    + "\(String(describing: hadamard[Self.hadamardSignModeKey]))")
        }
        guard let folded = hadamard[Self.hadamardWeightNamesKey] as? [String],
            !folded.isEmpty
        else {
            throw BuildError("\(Self.hadamardWeightNamesKey) must be non-empty")
        }
        guard let inverse =
            hadamard[Self.hadamardInverseWeightNamesKey] as? [String],
            !inverse.isEmpty
        else {
            throw BuildError(
                "\(Self.hadamardInverseWeightNamesKey) must be non-empty")
        }
        // The forward and inverse weight-name lists must be disjoint (a folded
        // tensor is either forward or inverse, never both). They are NOT
        // cross-checked against modules[] paths: the pinned pack's modules[]
        // paths and hadamard.json weight_names use different names/namespaces
        // for the same modules (e.g. modules[] `self_attn.q_proj` vs
        // weight_names `layers.0.linear_attn.in_proj_qkv.weight`), so an
        // exact partition check would reject the valid pack.
        let foldedBases = Set(folded.map(Self.normalizedModuleBase))
        let inverseBases = Set(inverse.map(Self.normalizedModuleBase))
        guard foldedBases.isDisjoint(with: inverseBases) else {
            throw BuildError(
                Self.hadamardWeightNamesKey + " and "
                    + Self.hadamardInverseWeightNamesKey + " must not overlap")
        }
        let widths = (hadamard[Self.hadamardSignWidthsKey] as? [Any])?
            .compactMap { ($0 as? NSNumber)?.intValue } ?? []
        let values = (hadamard[Self.hadamardSignValuesKey] as? [Any])?
            .compactMap { ($0 as? NSNumber)?.doubleValue } ?? []
        guard !widths.isEmpty else {
            throw BuildError("\(Self.hadamardSignWidthsKey) must be non-empty")
        }
        guard widths.allSatisfy({ $0 >= 1 }) else {
            throw BuildError(
                Self.hadamardSignWidthsKey + " must contain only positive widths")
        }
        guard !values.isEmpty else {
            throw BuildError("\(Self.hadamardSignValuesKey) must be non-empty")
        }
        guard widths.reduce(0, +) == values.count else {
            throw BuildError(
                "\(Self.hadamardSignWidthsKey) must sum to the "
                    + "\(Self.hadamardSignValuesKey) count (\(widths.reduce(0, +))"
                    + " != \(values.count))")
        }
        guard values.allSatisfy({ $0 == -1 || $0 == 1 }) else {
            throw BuildError(
                "\(Self.hadamardSignValuesKey) must contain only ±1 values")
        }
        guard (hadamard[Self.hadamardGDNVGroupedKey] as? Bool) == true else {
            throw BuildError("\(Self.hadamardGDNVGroupedKey) must be true")
        }

        var entries: [Entry] = []
        for (index, module) in modules.enumerated() {
            let path = modulePaths[index]
            let embedding = (module["embedding"] as? Bool) == true
            entries.append(
                Entry(
                    checkpointBase: path,
                    role: embedding ? .inversePackedEmbedding
                        : .forwardPackedLinear,
                    block: commonBlock,
                    dtypeName: stringValue(module["dtype"])))
        }

        self.block = commonBlock
        self.groupSize = Self.requiredGroupSize
        self.bits = Self.requiredBits
        self.entries = entries
        self.foldedWeightNames = folded
        self.inverseWeightNames = inverse
        self.signWidths = widths
        self.signValuesCount = values.count
    }
}

// MARK: - Install-plan resolution (pure, deterministic)

/// Pure install-plan resolution: maps a validated
/// ``PrismBonsaiHadamardPlan`` onto a concrete module tree (leaf paths) and a
/// concrete checkpoint weight dictionary (post-sanitize key forms), producing
/// the exact module updates and the exact `.signs`/packed keys to consume.
///
/// Checkpoint tensors may be published with or without the
/// `language_model.` / `model.` namespace prefixes (the pack publishes
/// `language_model.model.embed_tokens.*`; model sanitize may strip them), so
/// resolution matches leaf paths and weight keys through the same prefix
/// candidates the generic loader's `quantizedWeightBaseCandidates` inverts.
public enum PrismBonsaiInstall {

    public struct Error: LocalizedError, Equatable {
        public let reason: String
        public init(_ reason: String) { self.reason = reason }
        public var errorDescription: String? {
            "\(PrismBonsaiHadamardPlan.prismHadamardQwen35): \(reason)"
        }
    }

    public struct Update: Equatable, Sendable {
        /// Model leaf module path to replace (e.g. `model.embed_tokens`).
        public let modulePath: String
        /// Checkpoint tensor base from the plan (e.g.
        /// `language_model.model.embed_tokens`).
        public let checkpointBase: String
        public let role: PrismBonsaiHadamardPlan.Role
        public let block: Int
        /// Actual (post-sanitize) weight keys found on disk.
        public let weightKey: String
        public let scalesKey: String
        public let biasesKey: String
        public let signsKey: String
        /// Optional plain linear-bias key (`.*.bias`), nil when absent.
        public let optionalBiasKey: String?
        /// Every weight-dictionary key this update consumes (removed before
        /// the final `.noUnusedKeys` verification).
        public let weightKeysToConsume: [String]
    }

    public struct Plan: Equatable, Sendable {
        public let updates: [Update]
    }

    /// Module-leaf path candidates for a checkpoint tensor base: the base
    /// itself plus the namespace-stripped / re-namespaced forms.
    public static func modulePathCandidates(forCheckpointBase base: String)
        -> [String]
    {
        var seen = Set<String>()
        var out: [String] = []
        func add(_ value: String) {
            if seen.insert(value).inserted { out.append(value) }
        }
        add(base)
        if base.hasPrefix("language_model.model.") {
            add(String(base.dropFirst("language_model.model.".count)))
        }
        if base.hasPrefix("language_model.") {
            add(String(base.dropFirst("language_model.".count)))
        }
        if base.hasPrefix("model.") {
            add(String(base.dropFirst("model.".count)))
        }
        if !base.hasPrefix("language_model") && !base.hasPrefix("model.") {
            add("model.\(base)")
            add("language_model.\(base)")
            add("language_model.model.\(base)")
        }
        return out
    }

    /// Checkpoint key candidates for a module leaf path + suffix (the inverse
    /// mapping of ``modulePathCandidates(forCheckpointBase:)``).
    public static func checkpointKeyCandidates(
        moduleLeafPath: String, suffix: String
    ) -> [String] {
        var seen = Set<String>()
        var out: [String] = []
        func add(_ value: String) {
            if seen.insert(value).inserted { out.append(value) }
        }
        add(moduleLeafPath + suffix)
        if moduleLeafPath.hasPrefix("language_model.") {
            add(String(moduleLeafPath.dropFirst("language_model.".count)) + suffix)
        } else if moduleLeafPath.hasPrefix("model.") {
            add("language_model.\(moduleLeafPath)\(suffix)")
            add(String(moduleLeafPath.dropFirst("model.".count)) + suffix)
        } else {
            add("model.\(moduleLeafPath)\(suffix)")
            add("language_model.\(moduleLeafPath)\(suffix)")
            add("language_model.model.\(moduleLeafPath)\(suffix)")
        }
        return out
    }

    /// Resolve a validated plan against a model's leaf paths and the active
    /// checkpoint weight keys. Throws on the first inconsistency: unresolvable
    /// module, missing packed tensor or `.signs` vector, or an affine
    /// `.scales`/`.signs` tensor that no plan entry covers.
    public static func resolve(
        plan: PrismBonsaiHadamardPlan,
        leafModulePaths: Set<String>,
        checkpointKeys: Set<String>
    ) throws -> Plan {
        var updates: [Update] = []
        var coveredLeafPaths = Set<String>()

        for entry in plan.entries {
            guard let modulePath = modulePathCandidates(
                forCheckpointBase: entry.checkpointBase
            ).first(where: { leafModulePaths.contains($0) })
            else {
                throw Error(
                    "modules[] entry \(entry.checkpointBase) does not resolve "
                        + "to any model leaf path")
            }
            coveredLeafPaths.insert(modulePath)

            let weightKey = checkpointKeyCandidates(
                moduleLeafPath: modulePath, suffix: ".weight"
            ).first(where: { checkpointKeys.contains($0) })
            let scalesKey = checkpointKeyCandidates(
                moduleLeafPath: modulePath, suffix: ".scales"
            ).first(where: { checkpointKeys.contains($0) })
            let biasesKey = checkpointKeyCandidates(
                moduleLeafPath: modulePath, suffix: ".biases"
            ).first(where: { checkpointKeys.contains($0) })
            let signsKey = checkpointKeyCandidates(
                moduleLeafPath: modulePath, suffix: ".signs"
            ).first(where: { checkpointKeys.contains($0) })

            guard let weightKey, let scalesKey, let biasesKey else {
                throw Error(
                    "packed tensors missing for \(entry.checkpointBase) "
                        + "(need .weight, .scales, .biases)")
            }
            guard plan.block > 0, let signsKey else {
                throw Error(
                    "missing sign vector for \(entry.checkpointBase) "
                        + "(block \(plan.block))")
            }
            let biasKey = checkpointKeyCandidates(
                moduleLeafPath: modulePath, suffix: ".bias"
            ).first(where: { checkpointKeys.contains($0) })

            var consume = [weightKey, scalesKey, biasesKey, signsKey]
            if let biasKey { consume.append(biasKey) }
            updates.append(
                Update(
                    modulePath: modulePath,
                    checkpointBase: entry.checkpointBase,
                    role: entry.role,
                    block: plan.block,
                    weightKey: weightKey,
                    scalesKey: scalesKey,
                    biasesKey: biasesKey,
                    signsKey: signsKey,
                    optionalBiasKey: biasKey,
                    weightKeysToConsume: consume))
        }

        // Two manifest entries must never collapse onto the same model leaf
        // (namespace aliases would make the module swap ambiguous).
        guard Set(updates.map(\.modulePath)).count == updates.count else {
            throw Error("modules[] entries must map to distinct module paths")
        }

        // No affine companion or sign tensor may exist outside the plan: a
        // packed path without a transform module or an unplanned sign vector
        // would silently produce wrong output under the bypassed generic
        // affine swap.
        for key in checkpointKeys {
            for suffix in [".scales", ".signs", ".biases"] where key.hasSuffix(suffix) {
                let base = String(key.dropLast(suffix.count))
                guard let leaf = modulePathCandidates(forCheckpointBase: base)
                    .first(where: { leafModulePaths.contains($0) })
                else {
                    throw Error(
                        "affine tensor \(key) belongs to no model leaf module")
                }
                guard coveredLeafPaths.contains(leaf) else {
                    throw Error(
                        "affine tensor \(key) is outside the modules[] manifest")
                }
            }
        }

        return Plan(updates: updates)
    }
}