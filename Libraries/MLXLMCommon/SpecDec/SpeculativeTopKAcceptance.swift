// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
import MLX

/// Batched, lossless speculative-sampling acceptance over the target's top-K support.
///
/// The per-row path (`SpeculativeSamplingController.acceptOrCorrect`) builds every row's
/// full-vocabulary filtered distribution (a 248k-wide sort for top-p) and then reads it back
/// with 3-5 host syncs per draft. When the request has a top-k filter, the target distribution
/// lives entirely on the row's K largest logits, so the whole acceptance can run on the host
/// from ONE readback of `K` ids + log-probabilities per row:
///
///     kept(i)  ⇔  i ∈ topK  ∧  Σ_{j: p_j > p_i} p_j < topP  ∧  lp_i ≥ max lp + log(minP)
///
/// which is the same set `SpeculativeSamplingController.probabilities` keeps (its top-p keeps
/// tokens whose ascending cumulative probability exceeds `1 - topP`, i.e. whose strictly-larger
/// mass is below `topP`; top-k and min-p intersect it). Acceptance is the standard
/// Leviathan/Chen rule — accept `d` with `min(1, p(d)/q(d))`, else sample the residual
/// `max(p - q, 0)` — so the emitted tokens are exact samples of the target distribution. The
/// residual only needs `q` where `p > 0`, i.e. on the kept ids, so the draft distribution is
/// gathered at those `K` ids on the GPU in the same readback.
struct SpeculativeTopKAcceptance {

    /// The request parameters this path reproduces. `nil` when the request has no top-k filter
    /// (the support is then the whole vocabulary) or K is too large to read back cheaply.
    struct Filter {
        let temperature: Float
        let topP: Float
        let topK: Int
        let minP: Float
        /// z-lab DFlash 2 order: top-k first, then top-p over the RENORMALIZED top-k mass.
        /// `false` = `SpeculativeSamplingController` order (top-p over the full distribution).
        var nucleusWithinTopK = false

        init?(temperature: Float, topP: Float, topK: Int, minP: Float) {
            guard temperature > 0, topK >= 1, topK <= 256 else { return nil }
            if ProcessInfo.processInfo.environment["VMLX_SPEC_TOPK_ACCEPT"] == "0" { return nil }
            self.temperature = temperature
            self.topP = topP
            self.topK = topK
            self.minP = minP
        }
    }

    /// The device half: per row, the K largest scaled logits as (ids, log-probabilities over
    /// the FULL vocabulary). `logits` is `[R, V]` or `[1, R, V]`.
    static func topK(logits: MLXArray, filter: Filter) -> (ids: MLXArray, logprobs: MLXArray) {
        let vocab = logits.dim(-1)
        let rows = logits.size / vocab
        let x = logits.reshaped(rows, vocab).asType(.float32) * (1 / filter.temperature)
        let lse = logSumExp(x, axis: -1, keepDims: true)
        let k = Swift.min(filter.topK, vocab)
        let ids = argPartition(-x, kth: k - 1, axis: -1)[0..., ..<k]
        let logprobs = takeAlong(x, ids, axis: -1) - lse
        return (ids.asType(.int32), logprobs)
    }

    /// One row's target distribution on its kept ids (renormalized), host side.
    static func distribution(ids: ArraySlice<Int32>, logprobs: ArraySlice<Float>, filter: Filter)
        -> [(id: Int, p: Double)]
    {
        var pairs = zip(ids, logprobs).map { (id: Int($0.0), lp: Double($0.1)) }
        pairs.sort { $0.lp > $1.lp }
        if filter.nucleusWithinTopK, let top = pairs.first?.lp {
            let lse = top + log(pairs.reduce(0) { $0 + exp($1.lp - top) })
            pairs = pairs.map { (id: $0.id, lp: $0.lp - lse) }
        }
        let maxLP = pairs.first?.lp ?? 0
        let minLP = filter.minP > 0 ? maxLP + log(Double(filter.minP)) : -Double.infinity
        let useTopP = filter.topP > 0 && filter.topP < 1
        var kept: [(id: Int, p: Double)] = []
        kept.reserveCapacity(pairs.count)
        var larger = 0.0
        for pair in pairs {
            let p = exp(pair.lp)
            if useTopP && larger >= Double(filter.topP) { break }
            if pair.lp >= minLP { kept.append((pair.id, p)) }
            larger += p
        }
        if kept.isEmpty, let first = pairs.first { kept = [(first.id, 1)] }
        let total = kept.reduce(0) { $0 + $1.p }
        return kept.map { ($0.id, $0.p / total) }
    }

    struct Decision {
        let accepted: Int
        /// Accepted drafts followed by the correction (or bonus) token.
        let tokenIds: [Int]
        let acceptanceProbabilitySum: Double
        let acceptanceProbabilityCount: Int
    }

    /// Host acceptance for a chain of `drafts`. `p[r]` = target distribution of row r (r = 0…D),
    /// `qAtKept[r][j]` = draft probability of `p[r][j].id`, `qAtDraft[r]` = draft probability of
    /// `drafts[r]` (r < D).
    static func acceptChain(
        drafts: [Int], p: [[(id: Int, p: Double)]], qAtKept: [[Double]], qAtDraft: [Double],
        rng: inout SpeculativeHostRNG
    ) -> Decision {
        var sum = 0.0
        var count = 0
        for (r, draft) in drafts.enumerated() {
            let pd = p[r].first(where: { $0.id == draft })?.p ?? 0
            let qd = qAtDraft[r]
            let alpha = qd <= 0 ? (pd > 0 ? 1.0 : 0.0) : Swift.min(1, pd / qd)
            sum += alpha
            count += 1
            if alpha >= 1 { continue }
            if rng.uniform() < alpha { continue }
            // Residual max(p - q, 0) on the kept support.
            var residual: [(id: Int, p: Double)] = []
            for (j, entry) in p[r].enumerated() {
                let d = entry.p - qAtKept[r][j]
                if d > 0 { residual.append((entry.id, d)) }
            }
            let correction = residual.isEmpty
                ? rng.sample(p[r]) : rng.sample(residual)
            return Decision(
                accepted: r, tokenIds: Array(drafts.prefix(r)) + [correction],
                acceptanceProbabilitySum: sum, acceptanceProbabilityCount: count)
        }
        let bonus = rng.sample(p[drafts.count])
        return Decision(
            accepted: drafts.count, tokenIds: drafts + [bonus],
            acceptanceProbabilitySum: sum, acceptanceProbabilityCount: count)
    }
}

/// Host RNG for batched acceptance (SplitMix64; seeded from the request seed when it has one).
public struct SpeculativeHostRNG {
    private var state: UInt64

    public init(seed: UInt64?) {
        state = (seed ?? UInt64.random(in: 0 ... UInt64.max)) &+ 0x2545_F491_4F6C_DD1D
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [0, 1) with 53 random bits.
    public mutating func uniform() -> Double {
        Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Sample an id from unnormalized weights.
    public mutating func sample(_ weights: [(id: Int, p: Double)]) -> Int {
        let total = weights.reduce(0) { $0 + $1.p }
        var u = uniform() * total
        for w in weights {
            u -= w.p
            if u < 0 { return w.id }
        }
        return weights.last?.id ?? 0
    }
}
