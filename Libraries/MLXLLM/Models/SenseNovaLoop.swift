// Copyright © 2026. Apache-2.0.
// SenseNova Loop architecture: OpenSenseNova modeling_sensenova_loop.py
// @329a65f701ba8c48b73c30d0cbaa6f0449733673 (Apache-2.0).
// Qwen3.5 primitives remain shared with the existing ordinary model.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

public struct SenseNovaLoopConfiguration: Codable, Sendable {
    struct Loop: Codable, Sendable {
        let start: Int
        let end: Int
        let evalCount: Int
        let counts: [Int]
        enum CodingKeys: String, CodingKey {
            case start, end, counts
            case evalCount = "eval_count"
        }
    }
    struct MHC: Codable, Sendable {
        let start: Int
        let end: Int
        let streams: Int
        let iterations: Int
        let preScale: Float
        let fixedIdentity: Bool
        enum CodingKeys: String, CodingKey {
            case start, end, streams, iterations
            case preScale = "pre_scale"
            case fixedIdentity = "fixed_identity"
        }
    }
    struct WeightQuantization: Codable, Sendable {
        let bits: Int
        let groupSize: Int
        let mode: String?
        enum CodingKeys: String, CodingKey {
            case bits, mode
            case groupSize = "group_size"
        }
    }
    let modelType: String
    let text: Qwen35TextConfiguration
    let loop: Loop
    let mhc: MHC
    let normConvention: String
    let quantization: WeightQuantization?
    let activationDType: String

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case text = "text_config"
        case loop = "loop_settings"
        case mhc
        case normConvention = "norm_convention"
        case quantization
        case activationDType = "activation_dtype"
    }

    public enum ContractError: Error, CustomStringConvertible {
        case invalid(String)
        public var description: String {
            switch self { case .invalid(let message): return "sensenova_loop: \(message)" }
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        struct HeadDeclaration: Decodable { let num_attention_heads: Int }
        // The reusable Qwen decoder derives a missing head_dim by division.
        // Reject zero heads before calling it, rather than trapping on input.
        guard try c.decode(HeadDeclaration.self, forKey: .text).num_attention_heads > 0
        else { throw ContractError.invalid("num_attention_heads must be positive") }
        text = try c.decode(Qwen35TextConfiguration.self, forKey: .text)
        loop = try c.decode(Loop.self, forKey: .loop)
        mhc = try c.decode(MHC.self, forKey: .mhc)
        normConvention = try c.decodeIfPresent(String.self, forKey: .normConvention)
            ?? "qwen3_5_raw_weight"
        quantization = try c.decodeIfPresent(WeightQuantization.self, forKey: .quantization)
        activationDType = try c.decodeIfPresent(String.self, forKey: .activationDType) ?? "checkpoint"
        guard activationDType == "checkpoint" || (activationDType == "float32" && quantization != nil)
        else { throw ContractError.invalid("FP32 activations require an explicit affine-quantized bundle") }
        if let quantization {
            guard [4, 6, 8].contains(quantization.bits), quantization.groupSize == 64,
                  quantization.mode == nil || quantization.mode == "affine"
            else { throw ContractError.invalid("unsupported weight quantization") }
        }
        guard modelType == "sensenova_loop", text.modelType == "qwen3_5_text",
              text.hiddenLayers > 0, text.hiddenLayers <= 128,
              text.hiddenSize > 0, text.intermediateSize > 0, text.vocabularySize > 0,
              text.attentionHeads > 0, text.kvHeads > 0,
              text.attentionHeads % text.kvHeads == 0,
              text.linearNumKeyHeads > 0, text.linearNumValueHeads > 0,
              text.linearNumValueHeads % text.linearNumKeyHeads == 0,
              text.linearKeyHeadDim > 0, text.linearValueHeadDim > 0,
              text.linearConvKernelDim > 0, text.fullAttentionInterval == 4,
              text.numExperts == 0, text.mtpNumHiddenLayers == 0,
              !text.tieWordEmbeddings, !text.attentionBias,
              text.rmsNormEps.isFinite, text.rmsNormEps > 0,
              text.ropeTheta.isFinite, text.ropeTheta > 0,
              text.partialRotaryFactor.isFinite,
              text.partialRotaryFactor > 0, text.partialRotaryFactor <= 1,
              (text.headDim ?? 0) > 0, text.maxPositionEmbeddings > 0
        else { throw ContractError.invalid("unsupported dense text configuration") }
        guard 0 <= loop.start, loop.start < loop.end, loop.end <= text.hiddenLayers,
              loop.start % 4 == 0, loop.end % 4 == 0,
              loop.evalCount == 2, loop.counts == [2],
              mhc.start == loop.start, mhc.end == loop.end,
              mhc.streams == 2, mhc.iterations == 20, !mhc.fixedIdentity,
              mhc.preScale.isFinite, mhc.preScale == 1
        else { throw ContractError.invalid("only the fixed two-pass/two-stream release is supported") }
        guard ["qwen3_5_raw_weight", "mlx_plus_one"].contains(normConvention)
        else { throw ContractError.invalid("unknown norm convention") }
        // Qwen35 primitives derive their topology from the interval. Check the
        // authored explicit declaration rather than silently reinterpret it.
        struct Declaration: Decodable {
            let layer_types: [String]
            let attn_output_gate: Bool
        }
        let declaration = try c.decode(Declaration.self, forKey: .text)
        guard declaration.attn_output_gate,
              declaration.layer_types == (0..<text.hiddenLayers).map({
                  ($0 + 1) % 4 == 0 ? "full_attention" : "linear_attention"
              })
        else { throw ContractError.invalid("layer_types/output gate differs from the supported topology") }
        // Unknown execution/routing settings must not silently change semantics.
        struct Key: CodingKey {
            let stringValue: String
            let intValue: Int? = nil
            init?(stringValue: String) { self.stringValue = stringValue }
            init?(intValue: Int) { return nil }
        }
        let loopKeys = try c.superDecoder(forKey: .loop).container(keyedBy: Key.self).allKeys
        let mhcKeys = try c.superDecoder(forKey: .mhc).container(keyedBy: Key.self).allKeys
        guard Set(loopKeys.map(\.stringValue)) == ["start", "end", "eval_count", "counts"],
              Set(mhcKeys.map(\.stringValue)) == ["start", "end", "streams", "iterations", "pre_scale", "fixed_identity"]
        else { throw ContractError.invalid("unknown loop/mHC settings") }
    }

    var executionPlan: [Int] {
        Array(0..<loop.start) + Array(repeating: Array(loop.start..<loop.end), count: 2).flatMap { $0 }
            + Array(loop.end..<text.hiddenLayers)
    }
}

final class SenseNovaMHCRouter: Module {
    static let requestsFusedSinkhorn =
        ProcessInfo.processInfo.environment["VMLX_SENSENOVA_FUSED_SINKHORN"] == "1"

#if canImport(Metal)
    // One thread handles each token's 2x2 matrix. Keeping all 20 iterations
    // in registers avoids scheduling 40 separate row/column reductions.
    private static let sinkhornKernel = MLXFast.metalKernel(
        name: "sensenova_sinkhorn_2x2",
        inputNames: ["x"], outputNames: ["y"],
        source: """
            uint start = thread_position_in_grid.x * 4;
            float a = x[start], b = x[start + 1], c = x[start + 2], d = x[start + 3];
            for (int i = 0; i < 20; ++i) {
                float row0 = sn_logsumexp(a, b), row1 = sn_logsumexp(c, d);
                a -= row0; b -= row0; c -= row1; d -= row1;
                float col0 = sn_logsumexp(a, c), col1 = sn_logsumexp(b, d);
                a -= col0; c -= col0; b -= col1; d -= col1;
            }
            y[start] = exp(a); y[start + 1] = exp(b);
            y[start + 2] = exp(c); y[start + 3] = exp(d);
            """,
        header: """
            inline float sn_logsumexp(float a, float b) {
                float m = max(a, b);
                return m + log(exp(a - m) + exp(b - m));
            }
            """)
#endif

    static func sinkhorn(_ logits: MLXArray, fused: Bool) -> MLXArray {
#if canImport(Metal)
        if fused && Device.defaultDevice().deviceType == .gpu {
            let count = logits.size / 4
            return sinkhornKernel([logits], grid: (count, 1, 1),
                threadGroup: (min(count, 32), 1, 1), outputShapes: [logits.shape],
                outputDTypes: [.float32])[0]
        }
#endif
        var mix = logits
        for _ in 0..<20 {
            mix = mix - mix.logSumExp(axis: -1, keepDims: true)
            mix = mix - mix.logSumExp(axis: -2, keepDims: true)
        }
        return exp(mix)
    }

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    init(dimensions: Int) {
        _weight.wrappedValue = MLXArray.zeros([8, 2 * dimensions], dtype: .float32)
        _bias.wrappedValue = MLXArray.zeros([8], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ residual: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        precondition(weight.dtype == .float32 && bias.dtype == .float32,
                     "SenseNova mHC router parameters must remain FP32")
        let flat = residual.flattened(start: -2).asType(.float32)
        let normalized = flat * rsqrt((flat * flat).mean(axis: -1, keepDims: true) + 1e-6)
        let logits = matmul(normalized, weight.T) + bias
        let pre = sigmoid(logits[.ellipsis, 0..<2])
        let post = 2 * sigmoid(logits[.ellipsis, 2..<4])
        let mixLogits = logits[.ellipsis, 4..<8].reshaped(Array(logits.shape.dropLast()) + [2, 2])
        let mix = Self.sinkhorn(mixLogits, fused: Self.requestsFusedSinkhorn)
        let branch = matmul(pre.asType(residual.dtype).expandedDimensions(axis: -2), residual)
            .squeezed(axis: -2)
        return (branch, post, mix)
    }

    static func merge(_ residual: MLXArray, branch: MLXArray, post: MLXArray, mix: MLXArray) -> MLXArray {
        matmul(mix.asType(residual.dtype), residual)
            + post.asType(branch.dtype).expandedDimensions(axis: -1)
                * branch.expandedDimensions(axis: -2)
    }
}

// Preserve FP32 shifted norm parameters without promoting the residual stream.
// MLX's ordinary RMSNorm returns result_type(input, weight), while the authored
// reference explicitly casts the weighted FP32 result back to the input dtype.
final class SenseNovaRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: weight, eps: eps).asType(x.dtype)
    }
}

final class SenseNovaLoopLayer: Module {
    let isLinear: Bool
    let physicalIndex: Int
    let configuration: SenseNovaLoopConfiguration
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention?
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen35GatedDeltaNet?
    @ModuleInfo(key: "input_layernorm") var inputNorm: SenseNovaRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: SenseNovaRMSNorm
    @ModuleInfo(key: "mlp") var mlp: Qwen3NextMLP
    @ModuleInfo(key: "mhc_attn") var mhcAttn: SenseNovaMHCRouter?
    @ModuleInfo(key: "mhc_ffn") var mhcFFN: SenseNovaMHCRouter?

    init(_ configuration: SenseNovaLoopConfiguration, index: Int) {
        self.configuration = configuration
        self.physicalIndex = index
        let text = configuration.text
        isLinear = (index + 1) % 4 != 0
        if isLinear {
            _linearAttn.wrappedValue = Qwen35GatedDeltaNet(text, layerIdx: index,
                l2NormSumEpsilon: 1e-6, usesFP32GDNReference: true)
        }
        else { _selfAttn.wrappedValue = Qwen35Attention(text, preservesNormInputDType: true) }
        _inputNorm.wrappedValue = SenseNovaRMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
        _postNorm.wrappedValue = SenseNovaRMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
        _mlp.wrappedValue = Qwen3NextMLP(dimensions: text.hiddenSize, hiddenDimensions: text.intermediateSize)
        if configuration.mhc.start <= index && index < configuration.mhc.end {
            _mhcAttn.wrappedValue = SenseNovaMHCRouter(dimensions: text.hiddenSize)
            _mhcFFN.wrappedValue = SenseNovaMHCRouter(dimensions: text.hiddenSize)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
                        ssmMask: MLXArray?, cache: KVCache?) -> MLXArray {
        func attention(_ branch: MLXArray) -> MLXArray {
            if let linearAttn { return linearAttn(branch, mask: ssmMask, cache: cache as? MambaCache) }
            return selfAttn!(branch, mask: attentionMask, cache: cache)
        }
        guard let mhcAttn, let mhcFFN else {
            let h = x + attention(inputNorm(x))
            return h + mlp(postNorm(h))
        }
        var residual: MLXArray
        if physicalIndex == configuration.mhc.start {
            residual = stacked([x, x], axis: -2)
        } else {
            residual = x.reshaped(Array(x.shape.dropLast()) + [2, configuration.text.hiddenSize])
        }
        let (a, postA, mixA) = mhcAttn(residual)
        residual = SenseNovaMHCRouter.merge(residual, branch: attention(inputNorm(a)), post: postA, mix: mixA)
        let (f, postF, mixF) = mhcFFN(residual)
        residual = SenseNovaMHCRouter.merge(residual, branch: mlp(postNorm(f)), post: postF, mix: mixF)
        return physicalIndex == configuration.mhc.end - 1
            ? residual.mean(axis: -2) : residual.flattened(start: -2)
    }
}

public final class SenseNovaLoopTextInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [SenseNovaLoopLayer]
    @ModuleInfo(key: "norm") var norm: SenseNovaRMSNorm
    let plan: [Int]

    init(_ configuration: SenseNovaLoopConfiguration) {
        plan = configuration.executionPlan
        _embedTokens.wrappedValue = Embedding(embeddingCount: configuration.text.vocabularySize,
                                             dimensions: configuration.text.hiddenSize)
        _layers.wrappedValue = (0..<configuration.text.hiddenLayers).map { SenseNovaLoopLayer(configuration, index: $0) }
        _norm.wrappedValue = SenseNovaRMSNorm(dimensions: configuration.text.hiddenSize, eps: configuration.text.rmsNormEps)
        super.init()
    }

    func forward(_ inputs: MLXArray, cache: [KVCache]?, capture: Bool = false) -> (MLXArray, [MLXArray]) {
        precondition(cache == nil || cache!.count == plan.count, "SenseNova requires one cache per execution visit")
        var h = embedTokens(inputs)
        var states = capture ? [h] : []
        let faVisit = plan.firstIndex { !layers[$0].isLinear }!
        let ssmVisit = plan.firstIndex { layers[$0].isLinear }!
        let faMask = createAttentionMask(h: h, cache: cache?[faVisit])
        let ssmMask = createSSMMask(h: h, cache: cache?[ssmVisit] as? MambaCache)
        for (visit, physical) in plan.enumerated() {
            let layer = layers[physical]
            h = layer(h, attentionMask: layer.isLinear ? .none : faMask,
                      ssmMask: layer.isLinear ? ssmMask : nil, cache: cache?[visit])
            if capture { states.append(h) }
        }
        h = norm(h)
        if capture { states[states.count - 1] = h }
        return (h, states)
    }
}

public final class SenseNovaLoopModel: Module, LLMModel, KVCacheDimensionProvider {
    let configuration: SenseNovaLoopConfiguration
    @ModuleInfo public var model: SenseNovaLoopTextInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    public var vocabularySize: Int { configuration.text.vocabularySize }
    public var kvHeads: [Int] { configuration.executionPlan.map { _ in configuration.text.kvHeads } }
    public var loraLayers: [Module] { model.layers }
    public var preservesCheckpointParameterDTypes: Bool { true }
    public var supportsWholeForwardCompilation: Bool { false }
    public var maximumSupportedDecodeBatchSize: Int? { 1 }
    public var cacheStorageDTypeIdentity: String? { "sensenova-loop-v4-preserve-fp32-gdn" }
    public var cacheComputationIdentity: String? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let routing = SenseNovaMHCRouter.requestsFusedSinkhorn ? "fused-sinkhorn-v1:" : ""
        return "sensenova-loop-v4:" + routing + (try! encoder.encode(configuration)).base64EncodedString()
    }

    public init(_ configuration: SenseNovaLoopConfiguration) {
        self.configuration = configuration
        self.model = SenseNovaLoopTextInner(configuration)
        _lmHead.wrappedValue = Linear(configuration.text.hiddenSize, configuration.text.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model.forward(inputs, cache: cache).0)
    }

    @_spi(SenseNovaDiagnostics)
    public func diagnosticForward(_ inputs: MLXArray) -> (MLXArray, [MLXArray]) {
        let (hidden, states) = model.forward(inputs, cache: nil, capture: true)
        return (lmHead(hidden), states)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        configuration.executionPlan.map { physical in
            if model.layers[physical].isLinear { return MambaCache() }
            if let maximum = parameters?.maxKVSize { return RotatingKVCache(maxSize: maximum, keep: 4) }
            return KVCacheSimple()
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        for (key, value) in weights {
            // Vision stripping is a separate, inventory-checked conversion.
            // The one unused source parameter is explicitly allow-listed.
            if key == "model.language_model.loop_gate.logits" { continue }
            let name = key.replacingOccurrences(of: "model.language_model.", with: "model.")
            var tensor = value
            if name.hasSuffix("conv1d.weight"), tensor.dim(-1) != 1 {
                tensor = tensor.movedAxis(source: 2, destination: 1)
            }
            let norms = [".input_layernorm.weight", ".post_attention_layernorm.weight",
                         "model.norm.weight", ".q_norm.weight", ".k_norm.weight"]
            if configuration.normConvention == "qwen3_5_raw_weight", tensor.ndim == 1,
               norms.contains(where: { name.hasSuffix($0) }) {
                // The authored norm adds one in FP32 before multiplying, even
                // when the checkpoint stores its zero-centered weight in BF16.
                tensor = tensor.asType(.float32) + 1
            }
            if name.contains(".mhc_attn.") || name.contains(".mhc_ffn.") {
                tensor = tensor.asType(.float32)
            }
            if configuration.activationDType == "float32", tensor.dtype.isFloatingPoint {
                // The packed integer weights stay shared/quantized. Promoting
                // affine metadata and small unquantized parameters seeds the
                // qualified FP32 stream without a resident dequantized model.
                tensor = tensor.asType(.float32)
            }
            if out[name] != nil {
                // LanguageModel sanitation cannot throw. Preserve a diagnostic
                // unused key so strict module loading rejects the collision.
                out["sensenova_duplicate_weight." + key] = tensor
            } else {
                out[name] = tensor
            }
        }
        return out
    }
}
