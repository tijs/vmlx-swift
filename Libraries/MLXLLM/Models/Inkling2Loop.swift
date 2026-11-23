// Copyright © 2026. Apache-2.0.
// Architecture adapted from HuggingFace/MLX-VLM Inkling2 at 392052b4439c
// plus nativ-community Inkling2Mini patch e504b0e4. Apache-2.0 notices retained
// in docs/INKLING2_SOURCE.md. This is the single-request text AR envelope.
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

public struct Inkling2LoopConfiguration: Decodable, Sendable {
    public struct Loop: Decodable, Sendable {
        let coreStart: Int, coreEnd: Int, passes: Int, gateRank: Int
        let variable: Bool
        enum CodingKeys: String, CodingKey {
            case coreStart = "core_start", coreEnd = "core_end", passes = "num_passes"
            case gateRank = "gate_rank", variable = "is_varloop"
        }
        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            coreStart = try c.decode(Int.self, forKey: .coreStart)
            coreEnd = try c.decode(Int.self, forKey: .coreEnd)
            passes = try c.decode(Int.self, forKey: .passes)
            gateRank = try c.decode(Int.self, forKey: .gateRank)
            variable = try c.decode(Bool.self, forKey: .variable)
            struct Key: CodingKey {
                let stringValue: String
                var intValue: Int? { nil }
                init?(stringValue: String) { self.stringValue = stringValue }
                init?(intValue: Int) { return nil }
            }
            let keys = try decoder.container(keyedBy: Key.self).allKeys.map(\.stringValue)
            guard Set(keys) == ["core_start", "core_end", "num_passes", "gate_rank", "is_varloop"]
            else { throw ContractError.invalid("unknown loop execution settings") }
        }
    }
    public struct Text: Decodable, Sendable {
        let modelType: String
        let hidden: Int, layers: Int, vocab: Int, unpaddedVocab: Int
        let heads: Int, kvHeads: Int, headDim: Int
        let localHeads: Int, localKVHeads: Int, localHeadDim: Int
        let relativeDim: Int, relativeExtent: Int, window: Int, convolution: Int
        let intermediate: Int, experts: Int, topK: Int, sharedExperts: Int, block: Int
        let localLayers: [Int]
        let eps: Float, routeScale: Float, qClip: Float, relativeClip: Float
        let gateCap: Float, upCap: Float, logitCap: Float, logAlpha: Float
        let logFloor: Int?
        let normalizeValues: Bool, normalizeV: Bool
        let loop: Loop
        enum CodingKeys: String, CodingKey {
            case modelType = "model_type", hidden = "hidden_size", layers = "num_hidden_layers"
            case vocab = "vocab_size", unpaddedVocab = "unpadded_vocab_size"
            case heads = "num_attention_heads", kvHeads = "num_key_value_heads", headDim = "head_dim"
            case localHeads = "swa_num_attention_heads", localKVHeads = "swa_num_key_value_heads"
            case localHeadDim = "swa_head_dim", relativeDim = "d_rel", relativeExtent = "rel_extent"
            case window = "sliding_window_size", convolution = "sconv_kernel_size"
            case intermediate = "intermediate_size", experts = "n_routed_experts", topK = "num_experts_per_tok"
            case sharedExperts = "n_shared_experts", block = "n_layers_per_attn_res_block"
            case localLayers = "local_layer_ids", eps = "rms_norm_eps", routeScale = "route_scale"
            case qClip = "qk_l2_clip_threshold", relativeClip = "rel_l2_clip_threshold"
            case gateCap = "mlp_gate_softcap", upCap = "mlp_up_softcap", logitCap = "final_logit_softcapping"
            case logAlpha = "log_scaling_alpha", logFloor = "log_scaling_n_floor"
            case normalizeValues = "attn_res_normalize_values", normalizeV = "use_value_norm_no_gamma"
            case loop = "loop_config"
        }
        var schedule: [Int] {
            Array(0..<loop.coreStart)
                + Array(repeating: Array(loop.coreStart..<loop.coreEnd), count: loop.passes).flatMap { $0 }
                + Array(loop.coreEnd..<layers)
        }
        func sliding(_ i: Int) -> Bool { localLayers.contains(i) }
    }
    struct Quant: Decodable, Sendable {
        let groupSize: Int, bits: Int, mode: String
        enum CodingKeys: String, CodingKey { case groupSize = "group_size", bits, mode }
    }
    let modelType: String
    let text: Text
    static let embeddingPath = "language_model.model.embed_tokens"
    static let headPath = "language_model.lm_head"
    let quantization: [String: Quant]
    public enum ContractError: Error, CustomStringConvertible {
        case invalid(String)
        public var description: String { switch self { case .invalid(let s): "inkling2_loop: \(s)" } }
    }
    enum CodingKeys: String, CodingKey { case modelType = "model_type", text = "text_config", quantization }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        text = try c.decode(Text.self, forKey: .text)
        // The per-module table shares the object with global scalar keys.
        struct Key: CodingKey {
            let stringValue: String
            var intValue: Int? { nil }
            init?(stringValue: String) { self.stringValue = stringValue }
            init?(intValue: Int) { return nil }
        }
        var qs: [String: Quant] = [:]
        if c.contains(.quantization) {
            let q = try c.nestedContainer(keyedBy: Key.self, forKey: .quantization)
            for key in q.allKeys where !["group_size", "bits", "mode"].contains(key.stringValue) {
                qs[key.stringValue] = try q.decode(Quant.self, forKey: key)
            }
        }
        quantization = qs
        let t = text, l = text.loop
        guard modelType == "inkling2_loop", t.modelType == "inkling2_loop_text",
            t.hidden > 0, t.layers > 0, t.layers <= 128, t.vocab > 0,
            t.unpaddedVocab > 0, t.unpaddedVocab <= t.vocab,
            t.heads > 0, t.kvHeads > 0, t.heads % t.kvHeads == 0,
            t.localHeads > 0, t.localKVHeads > 0, t.localHeads % t.localKVHeads == 0,
            t.headDim > 0, t.localHeadDim > 0, t.relativeDim > 0,
            t.relativeExtent > 0, t.window > 0, t.convolution > 0,
            t.block > 0, t.experts > 0, t.topK > 0, t.topK <= t.experts,
            t.sharedExperts == 0, t.intermediate > 0,
            t.eps.isFinite, t.eps > 0, t.qClip.isFinite, t.qClip > 0,
            t.relativeClip.isFinite, t.relativeClip > 0,
            t.gateCap.isFinite, t.gateCap > 0, t.upCap.isFinite, t.upCap > 0,
            t.logitCap.isFinite, t.logitCap >= 0, t.routeScale.isFinite, t.routeScale > 0,
            t.logAlpha.isFinite, (t.logFloor ?? 1) > 0,
            Set(t.localLayers).count == t.localLayers.count,
            t.localLayers.allSatisfy({ 0 <= $0 && $0 < t.layers }),
            l.coreStart > 0, l.coreStart < l.coreEnd, l.coreEnd <= t.layers,
            l.coreStart % t.block == 0, l.coreEnd % t.block == 0,
            l.passes == 3, !l.variable, l.gateRank > 0
        else { throw ContractError.invalid("unsupported or malformed fixed three-pass text topology") }
        for path in [Self.embeddingPath, Self.headPath] {
            if let q = qs.removeValue(forKey: path) {
                guard q.mode == "affine", q.bits == 4, q.groupSize == 64, t.hidden % 64 == 0 else {
                    throw ContractError.invalid("unsupported vocabulary quantization at \(path)")
                }
            }
        }
        for (path, q) in qs {
            guard path.hasPrefix("language_model.model.layers."), path.contains(".mlp.switch_mlp."),
                ["nvfp4", "mxfp8"].contains(q.mode),
                q.bits == (q.mode == "nvfp4" ? 4 : 8),
                q.groupSize == (q.mode == "nvfp4" ? 16 : 32)
            else { throw ContractError.invalid("unsupported native quantization at \(path)") }
        }
        if !qs.isEmpty {
            guard t.hidden % 32 == 0, t.intermediate % 32 == 0 else {
                throw ContractError.invalid("native expert dimensions must be divisible by 32")
            }
            let expected = Set((0..<t.layers).flatMap { i in
                ["gate_proj", "up_proj", "down_proj"].map { "language_model.model.layers.\(i).mlp.switch_mlp.\($0)" }
            })
            guard Set(qs.keys) == expected else { throw ContractError.invalid("incomplete native expert quantization map") }
        }
    }
    func expertQuant(layer: Int, projection: String) -> Quant? {
        quantization["language_model.model.layers.\(layer).mlp.switch_mlp.\(projection)"]
    }
}

/// Chronological K/V for one execution visit. The absolute offset includes
/// evicted positions and remains independent for each pass through the loop.
public final class Inkling2VisitCache: BaseKVCache {
    private var keys: MLXArray?, values: MLXArray?
    private let window: Int?

    public override init() {
        self.window = nil
        super.init()
    }

    init(window: Int) {
        precondition(window > 0)
        self.window = window
        super.init()
    }

    var convolutions: [MLXArray?] = Array(repeating: nil, count: 4)
    // Generic chunked prefill evaluates the cache through Evaluatable, not
    // through its serialized state getter. Include every owned backing array.
    public override func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 } + convolutions.compactMap { $0 }
    }
    override public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let chunkKeys = keys.map { concatenated([$0, newKeys], axis: 2) } ?? newKeys
        let chunkValues = values.map { concatenated([$0, newValues], axis: 2) } ?? newValues
        offset += newKeys.dim(2)
        if let window, chunkKeys.dim(2) > window {
            let start = chunkKeys.dim(2) - window
            keys = contiguous(chunkKeys[0..., 0..., start..., 0...])
            values = contiguous(chunkValues[0..., 0..., start..., 0...])
        } else {
            keys = chunkKeys
            values = chunkValues
        }
        // The first query in a multi-token chunk can still see history that
        // the last query has passed. Evict only from next call's saved state.
        return (chunkKeys, chunkValues)
    }
    override public var state: [MLXArray] {
        get { [keys, values].compactMap { $0 } + convolutions.compactMap { $0 } }
        set {
            // State setters are used by scheduler-owned copy/update, never
            // advertised as a standalone serialized restore contract.
            if newValue.isEmpty { keys = nil; values = nil; convolutions = Array(repeating: nil, count: 4); offset = 0 }
            else if newValue.count == 6 {
                keys = newValue[0]; values = newValue[1]
                convolutions = newValue.dropFirst(2).map { Optional($0) }
                offset = newValue[0].dim(2)
            }
        }
    }
    override public var metaState: [String] {
        get { ["inkling2-visit-v1", String(offset)] }
        set { if newValue.count == 2, newValue[0] == "inkling2-visit-v1", let n = Int(newValue[1]) { offset = n } }
    }
    override public func copy() -> any KVCache {
        let c = window.map { Inkling2VisitCache(window: $0) } ?? Inkling2VisitCache()
        c.keys = keys.map { $0 * 1 }; c.values = values.map { $0 * 1 }
        c.convolutions = convolutions.map { $0.map { $0 * 1 } }; c.offset = offset
        return c
    }
}

private func inklingAttentionResidualEager(_ states: MLXArray, query: MLXArray, normalizeValues: Bool) -> MLXArray {
    let states = states.asType(.float32)
    let normalized = MLXFast.rmsNorm(states, weight: MLXArray.ones([states.dim(-1)]), eps: 1e-6)
    let scores = matmul(normalized, query.reshaped([-1, states.dim(-1)]).T) * Float(pow(Double(states.dim(-1)), -0.5))
    let weights = softmax(scores, axis: -2).swappedAxes(-1, -2)
    return matmul(weights, normalizeValues ? normalized : states)
}


// Opt-in parity/performance experiment matching the authored @mx.compile
// boundary. These stateless closures capture neither weights nor visit caches.
// Keep shape-specific traces: the query reshape contains the hidden dimension.
private let inklingCompileResidual = ProcessInfo.processInfo.environment["VMLX_INKLING_COMPILE_ATTENTION_RESIDUAL"] == "1"
private let inklingCompiledNormalizedResidual = vmlxTrustedCompile { arrays in
    [inklingAttentionResidualEager(arrays[0], query: arrays[1], normalizeValues: true)]
}
private let inklingCompiledRawResidual = vmlxTrustedCompile { arrays in
    [inklingAttentionResidualEager(arrays[0], query: arrays[1], normalizeValues: false)]
}
func inklingAttentionResidual(_ states: MLXArray, query: MLXArray, normalizeValues: Bool) -> MLXArray {
    if inklingCompileResidual && Device.defaultDevice().deviceType == .gpu {
        return (normalizeValues ? inklingCompiledNormalizedResidual : inklingCompiledRawResidual)([states, query])[0]
    }
    return inklingAttentionResidualEager(states, query: query, normalizeValues: normalizeValues)
}

func inklingL2Clip(_ x: MLXArray, threshold: Float, scale: MLXArray? = nil) -> MLXArray {
    Inkling2CompiledPrimitives.clip(x, threshold: threshold, scale: scale)
}

final class InklingAttentionResidual: Module {
    @ParameterInfo var query: MLXArray
    let normalizeValues: Bool
    init(_ t: Inkling2LoopConfiguration.Text, queries: Int = 1) {
        self.normalizeValues = t.normalizeValues
        _query.wrappedValue = .zeros(queries == 1 ? [t.hidden] : [queries, t.hidden])
        super.init()
    }
    func callAsFunction(_ blocks: [MLXArray], _ partial: MLXArray) -> MLXArray {
        inklingAttentionResidual(stacked(blocks + [partial], axis: -2), query: query, normalizeValues: normalizeValues)
    }
}

final class InklingMessageGate: Module {
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "forget_gate_proj") var forget: Linear
    @ModuleInfo(key: "message_gate_proj") var gain: Linear
    let eps: Float
    init(_ t: Inkling2LoopConfiguration.Text) {
        eps = t.eps
        _down.wrappedValue = Linear(2 * t.hidden, t.loop.gateRank, bias: false)
        _forget.wrappedValue = Linear(t.loop.gateRank, t.hidden, bias: false)
        _gain.wrappedValue = Linear(t.loop.gateRank, t.hidden, bias: false)
        super.init()
    }
    func callAsFunction(_ hidden: MLXArray, _ message: MLXArray) -> MLXArray {
        let h = hidden.asType(.float32), m = message.asType(.float32)
        let input = concatenated([MLXFast.rmsNorm(h, weight: MLXArray.ones([h.dim(-1)]), eps: eps), MLXFast.rmsNorm(m, weight: MLXArray.ones([h.dim(-1)]), eps: eps)], axis: -1)
        let g = down(input.asType(down.weight.dtype)).asType(.float32)
        let f = tanh(matmul(g, forget.weight.asType(.float32).T))
        let a = tanh(matmul(g, gain.weight.asType(.float32).T))
        return (1 - f) * h + a * m
    }
}

final class InklingMessageAttention: Module {
    @ParameterInfo var query: MLXArray
    init(_ t: Inkling2LoopConfiguration.Text) {
        _query.wrappedValue = .zeros([t.loop.coreEnd - t.loop.coreStart, 4, t.hidden])
        super.init()
    }
    func callAsFunction(_ blocks: [MLXArray], _ partial: MLXArray, layer: Int) -> MLXArray {
        inklingAttentionResidual(stacked(blocks + [partial], axis: -2), query: query[layer], normalizeValues: true)
    }
}

final class InklingStaticConvolutionWeight: Module {
    @ParameterInfo var weight: MLXArray
    init(channels: Int, kernel: Int) { _weight.wrappedValue = .zeros([channels, kernel, 1]); super.init() }
}
final class InklingShortConvolution: Module {
    @ModuleInfo var conv: InklingStaticConvolutionWeight
    let kernel: Int, index: Int
    init(channels: Int, kernel: Int, index: Int) {
        self.kernel = kernel; self.index = index
        _conv.wrappedValue = InklingStaticConvolutionWeight(channels: channels, kernel: kernel)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, _ cache: Inkling2VisitCache?) -> MLXArray {
        let b = x.dim(0), n = x.dim(1), c = x.dim(2), xf = x.asType(.float32)
        let old = cache?.convolutions[index] ?? .zeros([b, kernel - 1, c])
        let context = concatenated([old, xf], axis: 1)
        cache?.convolutions[index] = context[0..., (context.dim(1) - kernel + 1)..., 0...]
        if Inkling2InferenceKernels.fusedShortConvolutionEnabled,
            Device.defaultDevice().deviceType == .gpu, x.dtype == .bfloat16,
            kernel == 4, b <= 4, n > 0, n <= 8, c % 128 == 0 {
            return Inkling2InferenceKernels.virtualShortConvolution(
                state: old, weight: conv.weight, input: x, width: kernel)
        }
        var out = MLXArray.zeros(x.shape)
        for tap in 0..<kernel {
            out = out + context[0..., tap..<(tap+n), 0...] * conv.weight[0..., tap, 0].asType(.float32)
        }
        return ((out + xf) / sqrt(Float(kernel + 1))).asType(x.dtype)
    }
}
final class InklingDynamicConvolution: Module {
    @ParameterInfo(key: "static_weight") var taps: MLXArray
    @ModuleInfo(key: "weight_proj") var projection: Linear
    let kernel: Int, heads: Int, index: Int
    init(_ t: Inkling2LoopConfiguration.Text, channels: Int, heads: Int, index: Int) {
        kernel = t.convolution; self.heads = heads; self.index = index
        _taps.wrappedValue = .zeros([kernel, channels])
        _projection.wrappedValue = Linear(t.hidden, heads * kernel, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, input: MLXArray, cache: Inkling2VisitCache?) -> MLXArray {
        let b = x.dim(0), n = x.dim(1), c = x.dim(2), xf = x.asType(.float32)
        let old = cache?.convolutions[index] ?? .zeros([b, kernel - 1, c])
        let context = concatenated([old, xf], axis: 1)
        cache?.convolutions[index] = context[0..., (context.dim(1) - kernel + 1)..., 0...]
        let dynamic = projection(input).asType(.float32).reshaped([b,n,heads,1,kernel])
        let weights = taps.T.reshaped([heads,c/heads,kernel]) + dynamic
        var out = MLXArray.zeros([b,n,heads,c/heads])
        for lag in 0..<kernel {
            out = out + context[0..., (kernel-1-lag)..<(kernel-1-lag+n), 0...].reshaped(out.shape) * weights[.ellipsis, lag]
        }
        return ((out.reshaped(x.shape) + xf) / sqrt(Float(2*kernel+1))).asType(x.dtype)
    }
}

final class InklingAttention: Module {
    @ModuleInfo(key:"q_proj") var q: Linear
    @ModuleInfo(key:"k_proj") var k: Linear
    @ModuleInfo(key:"v_proj") var v: Linear
    @ModuleInfo(key:"r_proj") var r: Linear
    @ModuleInfo(key:"o_proj") var o: Linear
    @ModuleInfo(key:"gate_proj") var gate: Linear
    @ModuleInfo(key:"sink_proj") var sink: Linear
    @ParameterInfo(key:"q_scale") var qScale: MLXArray
    @ParameterInfo(key:"rel_scale") var relScale: MLXArray
    @ParameterInfo(key:"rel_proj") var relProjection: MLXArray
    @ModuleInfo(key:"k_sconv") var keyConvolution: Module
    @ModuleInfo(key:"v_sconv") var valueConvolution: Module
    let t: Inkling2LoopConfiguration.Text, local: Bool
    let heads: Int, kvHeads: Int, dim: Int, extent: Int
    init(_ t: Inkling2LoopConfiguration.Text, layer: Int) {
        self.t=t; local=t.sliding(layer); heads=local ? t.localHeads:t.heads
        kvHeads=local ? t.localKVHeads:t.kvHeads; dim=local ? t.localHeadDim:t.headDim
        extent=local ? t.window:t.relativeExtent
        _q.wrappedValue=Linear(t.hidden,heads*dim,bias:false)
        _k.wrappedValue=Linear(t.hidden,kvHeads*dim,bias:false)
        _v.wrappedValue=Linear(t.hidden,kvHeads*dim,bias:false)
        _r.wrappedValue=Linear(t.hidden,heads*t.relativeDim,bias:false)
        _o.wrappedValue=Linear(heads*dim,t.hidden,bias:false)
        _gate.wrappedValue=Linear(t.hidden,heads*dim,bias:false)
        _sink.wrappedValue=Linear(t.hidden,heads,bias:false)
        _qScale.wrappedValue = .ones([heads,dim]); _relScale.wrappedValue = .ones([heads,t.relativeDim])
        _relProjection.wrappedValue = .zeros([t.relativeDim,extent])
        if local {
            _keyConvolution.wrappedValue=InklingShortConvolution(channels:kvHeads*dim,kernel:t.convolution,index:0)
            _valueConvolution.wrappedValue=InklingShortConvolution(channels:kvHeads*dim,kernel:t.convolution,index:1)
        } else {
            _keyConvolution.wrappedValue=InklingDynamicConvolution(t,channels:kvHeads*dim,heads:kvHeads,index:0)
            _valueConvolution.wrappedValue=InklingDynamicConvolution(t,channels:kvHeads*dim,heads:kvHeads,index:1)
        }
        super.init()
    }
    func callAsFunction(_ x: MLXArray, keyInput: MLXArray, valueInput: MLXArray, cache: Inkling2VisitCache?) -> MLXArray {
        let b=x.dim(0), n=x.dim(1), dtype=x.dtype
        let key = local ? (keyConvolution as! InklingShortConvolution)(k(keyInput),cache) : (keyConvolution as! InklingDynamicConvolution)(k(keyInput),input:keyInput,cache:cache)
        let val = local ? (valueConvolution as! InklingShortConvolution)(v(valueInput),cache) : (valueConvolution as! InklingDynamicConvolution)(v(valueInput),input:valueInput,cache:cache)
        var query=inklingL2Clip(q(x).reshaped([b,n,heads,dim]),threshold:t.qClip,scale:qScale)
        let keyClipped=inklingL2Clip(key.reshaped([b,n,kvHeads,dim]),threshold:t.qClip)
        var value=val.reshaped([b,n,kvHeads,dim])
        if t.normalizeV { value=MLXFast.rmsNorm(value.asType(.float32),weight:MLXArray.ones([dim]),eps:t.eps).asType(dtype) }
        var relative=inklingL2Clip(r(x).reshaped([b,n,heads,t.relativeDim]),threshold:t.relativeClip,scale:relScale)
        let oldOffset=cache?.offset ?? 0
        if !local, let floor=t.logFloor {
            let positions=MLXArray(Int32(oldOffset+1)..<Int32(oldOffset+n+1)).asType(.float32)
            let tau=1+t.logAlpha*log(maximum(positions/Float(floor),1))
            query=(query.asType(.float32)*tau.reshaped([1,n,1,1])).asType(dtype)
            relative=(relative.asType(.float32)*tau.reshaped([1,n,1,1])).asType(dtype)
        }
        query=query.transposed(0,2,1,3)
        var keys=keyClipped.transposed(0,2,1,3)
        value=value.transposed(0,2,1,3)
        if let cache { (keys,value)=cache.update(keys:keys,values:value) }
        let count=keys.dim(2)
        // Mask distances are relative to the oldest returned key. Position
        // scaling above still uses the absolute offset, including evictions.
        let mask=inklingBandedMask(relative:relative,projection:relProjection.asType(dtype),offset:count-n,count:count,window:local ? t.window:0,extent:extent)
        let sinks=sink(x).transposed(0,2,1).expandedDimensions(axis:-1)
        let additive=concatenated([mask,sinks],axis:-1)
        keys=concatenated([keys,.zeros([b,kvHeads,1,dim],dtype:dtype)],axis:2)
        value=concatenated([value,.zeros([b,kvHeads,1,dim],dtype:dtype)],axis:2)
        var output=MLXFast.scaledDotProductAttention(queries:query,keys:keys,values:value,scale:1/sqrt(Float(dim)),mask:.array(additive))
        output=output.transposed(0,2,1,3).reshaped([b,n,heads*dim])
        output=(output.asType(.float32)*sigmoid(gate(x).asType(.float32))).asType(dtype)
        return o(output)
    }
}

func inklingBandedMask(relative: MLXArray, projection: MLXArray, offset: Int, count: Int, window: Int, extent: Int) -> MLXArray {
    let n=relative.dim(1), b=relative.dim(0), h=relative.dim(2)
    let projected=matmul(relative,projection).transposed(0,2,1,3)
    if Inkling2InferenceKernels.fusedMaskEnabled, Device.defaultDevice().deviceType == .gpu,
       let mask = Inkling2InferenceKernels.bandedMask(
           projected, count: count, offset: offset, window: window, extent: extent) {
        return mask.asType(relative.dtype)
    }
    let qp=MLXArray(Int32(offset)..<Int32(offset+n)).expandedDimensions(axis:1)
    let kp=MLXArray(Int32(0)..<Int32(count)).expandedDimensions(axis:0)
    let distance=qp-kp
    let indices=broadcast(clip(distance,min:0,max:extent-1).reshaped([1,1,n,count]),to:[b,h,n,count])
    var bias=takeAlong(projected,indices,axis:-1)
    bias=which(distance.reshaped([1,1,n,count]) .>= extent,MLXArray(Float(0)),bias)
    var invalid=distance .< 0
    if window>0 { invalid=invalid | (distance .>= window) }
    return which(invalid.reshaped([1,1,n,count]),MLXArray(Float(-1e30)),bias).asType(relative.dtype)
}

final class InklingExpertProjection: Module {
    @ParameterInfo var weight: MLXArray
    @ParameterInfo var scales: MLXArray?
    @ParameterInfo(key:"weight_scale_2") var outer: MLXArray?
    let quant: Inkling2LoopConfiguration.Quant?
    init(input:Int,output:Int,experts:Int,quant:Inkling2LoopConfiguration.Quant?) {
        self.quant=quant
        if let quant {
            _weight.wrappedValue = .zeros([experts,output,input*quant.bits/32],dtype:.uint32)
            _scales.wrappedValue = .zeros([experts,output,input/quant.groupSize],dtype:.uint8)
            if quant.mode=="nvfp4" { _outer.wrappedValue = .ones([experts],dtype:.float32) }
        } else { _weight.wrappedValue = .zeros([experts,output,input]) }
        super.init()
    }
    func callAsFunction(_ x:MLXArray,_ indices:MLXArray,sortedIndices:Bool = false)->MLXArray {
        guard let quant else { return gatherMM(x,weight.swappedAxes(-1,-2),rhsIndices:indices) }
        let mode:QuantizationMode=quant.mode=="nvfp4" ? .nvfp4:.mxfp8
        if let outer {
            if Device.defaultDevice().deviceType == .gpu {
                // MLX NVFP4 globalScale is tensor absmax, not the stored
                // post-product multiplier. Fuse it BEFORE BF16 rounding,
                // matching MLX-VLM ScaledQuantizedSwitchLinear exactly.
                return gatherQuantizedMM(x,weight,scales:scales!,biases:nil,rhsIndices:indices,
                    transpose:true,groupSize:quant.groupSize,bits:quant.bits,mode:mode,
                    globalScale:outer.asType(.float32) * 2688,sortedIndices:sortedIndices)
            }
            let y=gatherQuantizedMM(x.asType(.float32),weight,scales:scales!,biases:nil,rhsIndices:indices,
                transpose:true,groupSize:quant.groupSize,bits:quant.bits,mode:mode,sortedIndices:sortedIndices)
            let selected=outer[indices]
            return (y*selected.reshaped(selected.shape+[1,1])).asType(x.dtype)
        }
        return gatherQuantizedMM(x,weight,scales:scales!,biases:nil,rhsIndices:indices,
            transpose:true,groupSize:quant.groupSize,bits:quant.bits,mode:mode,sortedIndices:sortedIndices)
    }
}
final class InklingSwitchMLP:Module {
    @ModuleInfo(key:"gate_proj") var gate:InklingExpertProjection
    @ModuleInfo(key:"up_proj") var up:InklingExpertProjection
    @ModuleInfo(key:"down_proj") var down:InklingExpertProjection
    let t:Inkling2LoopConfiguration.Text
    init(_ c:Inkling2LoopConfiguration,layer:Int) {
        t=c.text
        _gate.wrappedValue=InklingExpertProjection(input:t.hidden,output:t.intermediate,experts:t.experts,quant:c.expertQuant(layer:layer,projection:"gate_proj"))
        _up.wrappedValue=InklingExpertProjection(input:t.hidden,output:t.intermediate,experts:t.experts,quant:c.expertQuant(layer:layer,projection:"up_proj"))
        _down.wrappedValue=InklingExpertProjection(input:t.intermediate,output:t.hidden,experts:t.experts,quant:c.expertQuant(layer:layer,projection:"down_proj"))
        super.init()
    }
    func callAsFunction(_ x:MLXArray,_ indices:MLXArray)->MLXArray {
        let sorted = Inkling2InferenceKernels.sortedExpertPrefillEnabled
            && Device.defaultDevice().deviceType == .gpu
            && x.dim(1) > 8 && indices.size >= 64
        return project(x, indices, sortPrefill: sorted)
    }
    // Match the authored token/expert gather-sort and scatter-unsort boundary.
    // Weights stay in their original packed format. Restore route order before
    // the caller's FP32 weighted sum; selected IDs and scores never change.
    func project(_ x: MLXArray, _ indices: MLXArray, sortPrefill: Bool) -> MLXArray {
        var expanded=x.expandedDimensions(axes:[-2,-3]), selected=indices
        var inverse: MLXArray?
        if sortPrefill {
            let sorted = gatherSort(x: expanded, indices: indices)
            expanded=sorted.0; selected=sorted.1; inverse=sorted.2
        }
        let g=gate(expanded,selected,sortedIndices:sortPrefill)
        let u=up(expanded,selected,sortedIndices:sortPrefill)
        let act=Inkling2CompiledPrimitives.activation(g,u,gateCap:t.gateCap,upCap:t.upCap)
        var result=down(act,selected,sortedIndices:sortPrefill)
        if let inverse { result=scatterUnsort(x:result,invOrder:inverse,shape:indices.shape) }
        return result.squeezed(axis:-2)
    }
}
final class InklingMoE:Module {
    @ParameterInfo(key:"gate_weight") var router:MLXArray
    @ParameterInfo(key:"e_score_correction_bias") var correction:MLXArray
    @ModuleInfo(key:"switch_mlp") var experts:InklingSwitchMLP
    let t:Inkling2LoopConfiguration.Text
    private let compiledRoute: @Sendable ([MLXArray]) -> [MLXArray]
    init(_ c:Inkling2LoopConfiguration,layer:Int) {
        t=c.text
        compiledRoute=Inkling2CompiledPrimitives.makeRoute(topK:c.text.topK)
        _router.wrappedValue = .zeros([t.experts,t.hidden]);_correction.wrappedValue = .zeros([t.experts])
        _experts.wrappedValue=InklingSwitchMLP(c,layer:layer);super.init()
    }
    func callAsFunction(_ x:MLXArray)->MLXArray {
        let args=[x,router,correction,MLXArray(t.routeScale)]
        let routed=Inkling2CompiledPrimitives.useCompiled ? compiledRoute(args)
            : Inkling2CompiledPrimitives.routeEager(args,topK:t.topK)
        let out=experts(x,routed[0])
        return Inkling2CompiledPrimitives.combine(out,routed[1])
    }
}

final class InklingDecoderLayer:Module {
    @ModuleInfo(key:"attention_input") var attentionInput:InklingAttentionResidual?
    @ModuleInfo(key:"mlp_input") var mlpInput:InklingAttentionResidual
    @ModuleInfo(key:"self_attn") var attention:InklingAttention
    @ModuleInfo var mlp:InklingMoE
    @ModuleInfo(key:"input_layernorm") var localNorm:RMSNorm?
    @ModuleInfo(key:"q_input_layernorm") var qNorm:RMSNorm?
    @ModuleInfo(key:"k_input_layernorm") var kNorm:RMSNorm?
    @ModuleInfo(key:"v_input_layernorm") var vNorm:RMSNorm?
    @ModuleInfo(key:"post_attention_layernorm") var postNorm:RMSNorm
    @ModuleInfo(key:"attn_sconv") var attnConv:InklingShortConvolution
    @ModuleInfo(key:"mlp_sconv") var mlpConv:InklingShortConvolution
    @ModuleInfo(key:"q_message_gate") var qMessage:InklingMessageGate?
    @ModuleInfo(key:"k_message_gate") var kMessage:InklingMessageGate?
    @ModuleInfo(key:"v_message_gate") var vMessage:InklingMessageGate?
    @ModuleInfo(key:"mlp_message_gate") var mlpMessage:InklingMessageGate?
    let local:Bool,boundary:Bool
    init(_ c:Inkling2LoopConfiguration,layer:Int) {
        let t=c.text;local=t.sliding(layer);boundary=layer%t.block==0
        if layer>0 { _attentionInput.wrappedValue=InklingAttentionResidual(t,queries:local ? 1:3) }
        _mlpInput.wrappedValue=InklingAttentionResidual(t);_attention.wrappedValue=InklingAttention(t,layer:layer)
        _mlp.wrappedValue=InklingMoE(c,layer:layer)
        if local { _localNorm.wrappedValue=RMSNorm(dimensions:t.hidden,eps:t.eps) }
        else { _qNorm.wrappedValue=RMSNorm(dimensions:t.hidden,eps:t.eps);_kNorm.wrappedValue=RMSNorm(dimensions:t.hidden,eps:t.eps);_vNorm.wrappedValue=RMSNorm(dimensions:t.hidden,eps:t.eps) }
        _postNorm.wrappedValue=RMSNorm(dimensions:t.hidden,eps:t.eps)
        _attnConv.wrappedValue=InklingShortConvolution(channels:t.hidden,kernel:t.convolution,index:2)
        _mlpConv.wrappedValue=InklingShortConvolution(channels:t.hidden,kernel:t.convolution,index:3)
        if t.loop.coreStart<=layer && layer<t.loop.coreEnd {
            _qMessage.wrappedValue=InklingMessageGate(t);_kMessage.wrappedValue=InklingMessageGate(t)
            _vMessage.wrappedValue=InklingMessageGate(t);_mlpMessage.wrappedValue=InklingMessageGate(t)
        }
        super.init()
    }
    func callAsFunction(_ x:MLXArray,blocks:inout [MLXArray],cache:Inkling2VisitCache?,messages:MLXArray?)->MLXArray {
        let mixed=attentionInput?(blocks,x) ?? x.expandedDimensions(axis:-2)
        var parts=(0..<3).map { mixed[.ellipsis,min($0,mixed.dim(-2)-1),0...] }
        if let messages {
            parts=zip(parts,[qMessage!,kMessage!,vMessage!]).enumerated().map { i,pair in pair.1(pair.0,messages[.ellipsis,i,0...]) }
        }
        let normalized=zip(parts,local ? [localNorm!,localNorm!,localNorm!]:[qNorm!,kNorm!,vNorm!]).map { $0.1($0.0).asType(x.dtype) }
        let a=attention(normalized[0],keyInput:normalized[1],valueInput:normalized[2],cache:cache)
        var residual=attnConv(a,cache).asType(.float32)
        if boundary { blocks.append(x) } else { residual=residual+x.asType(.float32) }
        var h=mlpInput(blocks,residual)[.ellipsis,0,0...]
        if let messages { h=mlpMessage!(h,messages[.ellipsis,3,0...]) }
        h=mlp(postNorm(h).asType(x.dtype))
        return (residual+mlpConv(h,cache).asType(.float32)).asType(x.dtype)
    }
}
final class InklingTextModel:Module {
    @ModuleInfo(key:"embed_tokens") var embed:Embedding
    @ModuleInfo(key:"embed_norm") var embedNorm:RMSNorm
    @ModuleInfo var layers:[InklingDecoderLayer]
    @ModuleInfo(key:"final_input") var finalInput:InklingAttentionResidual
    @ModuleInfo var norm:RMSNorm
    @ModuleInfo(key:"message_input") var messageInput:InklingMessageAttention
    let t:Inkling2LoopConfiguration.Text
    init(_ c:Inkling2LoopConfiguration) {
        t = c.text
        if let q = c.quantization[Inkling2LoopConfiguration.embeddingPath] {
            _embed.wrappedValue = QuantizedEmbedding(
                weight: .zeros([t.vocab, t.hidden * q.bits / 32], dtype: .uint32),
                scales: .zeros([t.vocab, t.hidden / q.groupSize], dtype: .bfloat16),
                biases: .zeros([t.vocab, t.hidden / q.groupSize], dtype: .bfloat16),
                groupSize: q.groupSize, bits: q.bits)
        } else {
            _embed.wrappedValue = Embedding(embeddingCount: t.vocab, dimensions: t.hidden)
        }
        _embedNorm.wrappedValue=RMSNorm(dimensions:t.hidden,eps:t.eps)
        _layers.wrappedValue=(0..<t.layers).map { InklingDecoderLayer(c,layer:$0) }
        _finalInput.wrappedValue=InklingAttentionResidual(t);_norm.wrappedValue=RMSNorm(dimensions:t.hidden,eps:t.eps)
        _messageInput.wrappedValue=InklingMessageAttention(t);super.init()
    }
    func callAsFunction(_ input:MLXArray,cache:[KVCache]?)->MLXArray {
        var x=embedNorm(embed(input)),blocks:[MLXArray]=[]
        var preLoop:(MLXArray,[MLXArray])?,previous:([MLXArray],MLXArray)?
        for (visit,physical) in t.schedule.enumerated() {
            if physical==t.loop.coreStart {
                if let saved=preLoop {
                    previous=(Array(blocks.dropFirst(saved.1.count+1)),x)
                    x=saved.0;blocks=saved.1
                } else { preLoop=(x,blocks) }
            }
            let messages:MLXArray?
            if let previous, physical<t.loop.coreEnd {
                messages=messageInput(previous.0,previous.1,layer:physical-t.loop.coreStart)
            } else { messages=nil }
            x=layers[physical](x,blocks:&blocks,cache:cache?[visit] as? Inkling2VisitCache,messages:messages)
        }
        return norm(finalInput(blocks,x)[.ellipsis,0,0...]).asType(embedNorm.weight.dtype)
    }
}
final class InklingLanguageModel:Module {
    @ModuleInfo var model:InklingTextModel
    @ModuleInfo(key:"lm_head") var head:Linear
    init(_ c:Inkling2LoopConfiguration) {
        _model.wrappedValue = InklingTextModel(c)
        if let q = c.quantization[Inkling2LoopConfiguration.headPath] {
            let t = c.text
            _head.wrappedValue = QuantizedLinear(
                weight: .zeros([t.vocab, t.hidden * q.bits / 32], dtype: .uint32),
                scales: .zeros([t.vocab, t.hidden / q.groupSize], dtype: .bfloat16),
                biases: .zeros([t.vocab, t.hidden / q.groupSize], dtype: .bfloat16),
                groupSize: q.groupSize, bits: q.bits)
        } else {
            _head.wrappedValue = Linear(c.text.hidden, c.text.vocab, bias: false)
        }
        super.init()
    }
}
private final class InklingHeadProjectionCache {
    var weight: MLXArray?
}
public final class Inkling2LoopModel:Module,LLMModel,KVCacheDimensionProvider,ThrowingWeightSanitizing {
    public let alignedVisitPrefixIdentity = UUID()
    public let configuration:Inkling2LoopConfiguration
    private let headProjectionCache = InklingHeadProjectionCache()
    private let boundedLocalKV = ProcessInfo.processInfo.environment["VMLX_INKLING_BOUNDED_LOCAL_KV"] != "0"
    @ModuleInfo(key:"language_model") var language:InklingLanguageModel
    public var preservesCheckpointParameterDTypes: Bool { true }
    public var vocabularySize:Int { configuration.text.unpaddedVocab }
    public var kvHeads:[Int] { configuration.text.schedule.map { configuration.text.sliding($0) ? configuration.text.localKVHeads:configuration.text.kvHeads } }
    public var loraLayers:[Module] { language.model.layers }
    public var cacheStorageDTypeIdentity: String? {
        boundedLocalKV
            ? "inkling2mini-window-\(configuration.text.window)-kv-four-fp32-conv-v1"
            : "inkling2mini-full-kv-four-fp32-conv-v1"
    }
    public var cacheComputationIdentity:String? { "inkling2mini-3pass-ar-float-activations-v1" }
    public init(_ configuration:Inkling2LoopConfiguration) {
        self.configuration=configuration;_language.wrappedValue=InklingLanguageModel(configuration);super.init()
    }
    public func callAsFunction(_ inputs:MLXArray,cache:[KVCache]?)->MLXArray {
        let h=language.model(inputs,cache:cache).asType(.float32)
        // The reference retains one FP32 head projection; repeatedly promoting
        // the full vocabulary table would rematerialize ~1.85GB every token.
        var logits: MLXArray
        if let head = language.head as? QuantizedLinear {
            logits = head(h).asType(.float32)
        } else if Inkling2HeadProjection.enabled && language.model.embedNorm.weight.dtype == .bfloat16
            && Inkling2HeadProjection.supports(h, weight: language.head.weight) {
            // Prefill still uses the authored FP32 matrix path. Release its
            // derived copy when decode can read BF16 storage directly.
            headProjectionCache.weight = nil
            logits = Inkling2HeadProjection.project(h, weight: language.head.weight)
        } else {
            if headProjectionCache.weight == nil {
                headProjectionCache.weight = language.head.weight.asType(.float32)
            }
            logits = matmul(h,headProjectionCache.weight!.T)
        }
        let cap=configuration.text.logitCap
        if cap>0 { logits=tanh(logits/cap)*cap }
        return logits[.ellipsis,0..<vocabularySize]
    }
    @discardableResult
    public override func update(parameters: ModuleParameters, verify: Module.VerifyUpdate,
                                path: [String] = [], modulePath: [String] = []) throws -> Self {
        headProjectionCache.weight = nil
        return try super.update(parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }
    public func newCache(parameters:GenerateParameters?)->[KVCache] {
        makeVisitCaches(bounded: boundedLocalKV)
    }

    func makeVisitCaches(bounded: Bool) -> [KVCache] {
        let text = configuration.text
        return text.schedule.map { physical in
            if bounded && text.localLayers.contains(physical) {
                return Inkling2VisitCache(window: text.window)
            }
            return Inkling2VisitCache()
        }
    }
    public func sanitizeWeights(_ weights:[String:MLXArray],metadata:[String:String]) throws ->[String:MLXArray] {
        // Published target bundle is already sanitized. Only its explicitly
        // separate vision tower is outside this first text-only envelope.
        let expected=Dictionary(uniqueKeysWithValues:parameters().flattened())
        var out:[String:MLXArray]=[:]
        for (key,value) in weights {
            if key.hasPrefix("vision_tower.") { continue }
            guard let placeholder=expected[key],value.shape==placeholder.shape else {
                throw Inkling2LoopConfiguration.ContractError.invalid("unexpected/malformed checkpoint tensor \(key) \(value.shape)")
            }
            if placeholder.dtype == .uint32 || placeholder.dtype == .uint8 {
                guard value.dtype==placeholder.dtype else { throw Inkling2LoopConfiguration.ContractError.invalid("packed dtype mismatch at \(key)") }
            } else {
                guard value.dtype.isFloatingPoint else { throw Inkling2LoopConfiguration.ContractError.invalid("floating dtype required at \(key)") }
                if [Inkling2LoopConfiguration.headPath, Inkling2LoopConfiguration.embeddingPath].contains(where: {
                    key == $0 + ".scales" || key == $0 + ".biases"
                }) {
                    guard value.dtype == .bfloat16 else {
                        throw Inkling2LoopConfiguration.ContractError.invalid("expected BF16 affine metadata at \(key)")
                    }
                } else if key.hasSuffix(".gate_weight") {
                    // The published Mini checkpoint stores all 24 routers in
                    // BF16. Preserve that storage; InklingMoE promotes the
                    // operands to FP32 for routing, as the reference does.
                    guard value.dtype == .bfloat16 || value.dtype == .float32 else {
                        throw Inkling2LoopConfiguration.ContractError.invalid("BF16/FP32 router required at \(key)")
                    }
                } else if key.hasSuffix("weight_scale_2") || key.hasSuffix(".query") || key.hasSuffix(".e_score_correction_bias") {
                    guard value.dtype == .float32 else { throw Inkling2LoopConfiguration.ContractError.invalid("FP32-sensitive tensor \(key)") }
                }
            }
            out[key]=value
        }
        guard Set(out.keys)==Set(expected.keys) else {
            throw Inkling2LoopConfiguration.ContractError.invalid("missing checkpoint tensors: \(Set(expected.keys).subtracting(out.keys).sorted())")
        }
        return out
    }
}

extension Inkling2LoopModel: AlignedVisitPrefixModel {
    public var alignedVisitPrefixLength: Int { 1024 }
}
