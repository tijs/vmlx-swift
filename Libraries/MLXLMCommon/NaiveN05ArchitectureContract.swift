import Foundation

/// Source-derived architecture metadata, not registration or executable model support.
/// Generation settings belong to the model bundle and are intentionally not decoded here.
public struct NaiveN05ArchitectureContract: Decodable, Sendable {
    public enum ContractError: Error, Equatable { case unsupported(String) }
    public enum AttentionKind: Int, Sendable {
        case sparse = 0
        case sliding = 1
    }
    public enum IndexerPrecision: String, Decodable, Sendable {
        case bf16
        case fp8E4M3 = "fp8_e4m3"
    }

    public struct Attention: Sendable {
        public let heads: Int
        public let kvHeads: Int
        public let keyDimensions: Int
        public let valueDimensions: Int
        public let rotaryDimensions: Int
        public let ropeTheta: Double
        public let hasSink: Bool
    }

    public let layerCount: Int
    public let hiddenDimensions: Int
    public let denseDimensions: Int
    public let expertDimensions: Int
    public let expertCount: Int
    public let routes: Int
    public let vocabularySize: Int
    public let contextLimit: Int
    public let normEpsilon: Double
    public let attentionKinds: [AttentionKind]
    public let routedLayers: [Bool]
    public let fullAttention: Attention
    public let slidingAttention: Attention
    public let window: Int
    public let valueScale: Double?
    public let requestedRoutingScale: Double?
    public let routingScaleWasProvided: Bool
    public let routingScale: Double
    public let normalizeRoutes: Bool
    public let attentionBias: Bool
    public let indexerHeads: Int
    public let indexerDimensions: Int
    public let indexerTopK: Int
    public let indexerPrecision: IndexerPrecision

    private struct Key: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init(_ name: String) { stringValue = name }
        init?(stringValue: String) { self.init(stringValue) }
        init?(intValue: Int) { return nil }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Key.self)
        // Absent values use the vendor defaults. Explicit null is rejected for
        // non-nullable geometry instead of silently replacing malformed metadata.
        func field<T: Decodable>(_ name: String, _ fallback: T) throws -> T {
            c.contains(Key(name)) ? try c.decode(T.self, forKey: Key(name)) : fallback
        }
        func positive(_ value: Int, _ name: String) throws -> Int {
            guard value > 0 else { throw ContractError.unsupported(name) }
            return value
        }
        guard try c.decode(String.self, forKey: Key("model_type")) == "naive_n05_flash"
        else { throw ContractError.unsupported("model_type") }
        guard try field("attention_projection_layout", "split") == "split",
            try field("index_n_kv_heads", 1) == 1,
            try field("enable_dsa", true),
            try field("scoring_func", "sigmoid") == "sigmoid",
            try field("topk_method", "noaux_tc") == "noaux_tc",
            try field("n_group", 1) == 1, try field("topk_group", 1) == 1,
            try field("hidden_act", "silu") == "silu",
            try field("tie_word_embeddings", false) == false
        else { throw ContractError.unsupported("unsupported Naive architecture variant") }
        if let implementation = try c.decodeIfPresent(
            String.self, forKey: Key("_attn_implementation")),
            implementation != "eager"
        {
            throw ContractError.unsupported("attention implementation")
        }
        if let shared = try c.decodeIfPresent(Int.self, forKey: Key("n_shared_experts")),
            shared != 0
        {
            throw ContractError.unsupported("shared experts")
        }
        layerCount = try positive(field("num_hidden_layers", 48), "num_hidden_layers")
        // Bound derived metadata allocation; this is a runtime-supported subset,
        // not a claim that larger vendor configurations are malformed.
        guard layerCount <= 4096 else {
            throw ContractError.unsupported("layer count exceeds contract allocation limit")
        }
        hiddenDimensions = try positive(field("hidden_size", 4096), "hidden_size")
        denseDimensions = try positive(field("intermediate_size", 16384), "intermediate_size")
        expertDimensions = try positive(
            field("moe_intermediate_size", 2048), "moe_intermediate_size")
        expertCount = try positive(field("n_routed_experts", 256), "n_routed_experts")
        routes = try positive(field("num_experts_per_tok", 8), "num_experts_per_tok")
        guard routes <= expertCount else {
            throw ContractError.unsupported("routes exceed experts")
        }
        vocabularySize = try positive(field("vocab_size", 152576), "vocab_size")
        contextLimit = try positive(
            field("max_position_embeddings", 1_048_576), "max_position_embeddings")
        normEpsilon = try field("layernorm_epsilon", 1e-5)
        window = try positive(field("sliding_window", 128), "sliding_window")
        attentionBias = try field("attention_bias", false)
        normalizeRoutes = try field("norm_topk_prob", true)
        indexerHeads = try positive(field("index_n_heads", 16), "index_n_heads")
        indexerDimensions = try positive(field("index_head_dim", 128), "index_head_dim")
        indexerTopK = try positive(field("index_top_k", 2048), "index_top_k")
        indexerPrecision = try field("indexer_activation_dtype", IndexerPrecision.fp8E4M3)
        valueScale =
            c.contains(Key("attention_value_scale"))
            ? try c.decodeIfPresent(Double.self, forKey: Key("attention_value_scale")) : 0.707
        requestedRoutingScale = try c.decodeIfPresent(
            Double.self, forKey: Key("routed_scaling_factor"))
        routingScaleWasProvided = c.contains(Key("routed_scaling_factor"))
        let requestedScale = requestedRoutingScale
        // Vendor config uses `routed_scaling_factor or 1.0`, including explicit zero.
        routingScale = requestedScale == nil || requestedScale == 0 ? 1 : requestedScale!
        guard normEpsilon.isFinite, normEpsilon > 0, routingScale.isFinite, routingScale > 0,
            valueScale.map({ $0.isFinite }) ?? true
        else { throw ContractError.unsupported("nonfinite or unsupported scalar") }

        let partial = try field("partial_rotary_factor", 0.334)
        guard partial.isFinite, partial > 0, partial <= 1
        else { throw ContractError.unsupported("partial_rotary_factor") }
        func attention(_ prefix: String, defaultKV: Int, theta: Double, sink: Bool) throws
            -> Attention
        {
            let heads = try positive(field(prefix + "num_attention_heads", 64), "heads")
            let kv = try positive(field(prefix + "num_key_value_heads", defaultKV), "kv_heads")
            let key = try positive(field(prefix + "head_dim", 192), "head_dim")
            let value = try positive(field(prefix + "v_head_dim", 128), "v_head_dim")
            let rotaryValue = Double(key) * partial
            guard rotaryValue < Double(Int.max), heads.isMultiple(of: kv)
            else { throw ContractError.unsupported("head geometry") }
            let rotary = Int(rotaryValue)
            guard rotary > 0, rotary <= key, rotary.isMultiple(of: 2)
            else { throw ContractError.unsupported("rotary geometry") }
            let base = try field(prefix + "rope_theta", theta)
            guard base.isFinite, base > 0 else { throw ContractError.unsupported("rope_theta") }
            return Attention(
                heads: heads, kvHeads: kv, keyDimensions: key, valueDimensions: value,
                rotaryDimensions: rotary, ropeTheta: base,
                hasSink: try field(
                    prefix.isEmpty ? "add_full_attention_sink_bias" : "add_swa_attention_sink_bias",
                    sink))
        }
        fullAttention = try attention("", defaultKV: 4, theta: 10_000_000, sink: false)
        slidingAttention = try attention("swa_", defaultKV: 8, theta: 10_000, sink: true)
        guard fullAttention.rotaryDimensions <= indexerDimensions else {
            throw ContractError.unsupported("indexer rotary dimensions")
        }
        let hybrid = try c.decodeIfPresent([Int].self, forKey: Key("hybrid_layer_pattern"))
        let pattern =
            hybrid.flatMap { $0.isEmpty ? nil : $0 }
            ?? (0 ..< layerCount).map { $0 != 0 && $0 % 6 != 5 ? 1 : 0 }
        let mlp = try c.decodeIfPresent([Int].self, forKey: Key("moe_layer_freq"))
        let routed =
            mlp.flatMap { $0.isEmpty ? nil : $0 } ?? (0 ..< layerCount).map { $0 == 0 ? 0 : 1 }
        guard pattern.count == layerCount, routed.count == layerCount,
            pattern.allSatisfy({ $0 == 0 || $0 == 1 }), routed.allSatisfy({ $0 == 0 || $0 == 1 })
        else { throw ContractError.unsupported("layer schedules") }
        attentionKinds = pattern.map { AttentionKind(rawValue: $0)! }
        routedLayers = routed.map { $0 == 1 }
    }
}
