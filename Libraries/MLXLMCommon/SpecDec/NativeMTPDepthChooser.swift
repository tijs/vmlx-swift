// Native-MTP adaptive depth chooser: expected committed tokens per measured second (the DFlash2 width
// chooser's model, DFlash2WidthChooser.swift, applied to MTP draft depth). When enabled
// (VMLX_MTP_DEPTH_CHOOSER=1, OPT-IN — it lost to the existing governor in matched-clock A/B, see `enabled`)
// it replaces the acceptance-floor / wall-clock-demote heuristics for explicitly ADAPTIVE requests on the
// staged greedy verifier. AR safety (pause/resume vs measured AR) is unchanged.
//
// WHY (measured 2026-10-06, Flash-Next JANG_4S, max2, greedy, 300 tokens, tok/s):
//   fixed depth   D1    D2    D3    D4    D5     old adaptive
//   prose         57.0  56.2  50.5  44.0  40.7   45.7
//   code          70.1  81.2  86.7  84.6  79.7   64.7
//   easy_code     71.3  85.2  95.2  96.9  100.1  74.9
//   easy_prose    70.0  82.6  92.3  89.6  90.1   72.2
// The best depth differs per text (prose D1, code D3, repetitive D3-D5) and the floor heuristics picked
// wrong both ways (promoted to D5 where D3 wins; AR-safety demoted prose through D3->D1->pause against a
// stale AR reference).
//
// MODEL: E[tokens | d] = 1 + sum_{j=1}^{d} prod_{k<=j} p_k, p_k = P(draft k accepted | drafts 1..k-1 were).
//   p_k is learned from every cycle at any depth (positions 1..accepted hit, accepted+1 missed when it was
//   drafted), counts decay x0.85 per cycle; a thinly-observed position leans on the previous position's
//   estimate (one pseudo-trial, prior 0.7).
//   cost(d) = EMA(0.25) of the measured period between consecutive committed cycles at depth d (drafting +
//   verify + sampling + commit, everything the user waits on), shared process-wide per model; the first
//   period at each depth is discarded (kernel compile / admission). Unmeasured depths are visited first;
//   then the least-recently-timed neighbour depth is re-timed every 48 cycles.
// Correctness: depth only changes how many drafts are verified, never the acceptance rule.

import Foundation

final class NativeMTPDepthCostTable: @unchecked Sendable {
    private let lock = NSLock()
    private var cost: [Int: Double] = [:]
    private var warm: Set<Int> = []

    func observe(depth: Int, seconds: Double) {
        lock.lock()
        defer { lock.unlock() }
        if !warm.contains(depth) {
            warm.insert(depth)
            return
        }
        if let c = cost[depth] {
            cost[depth] = c + 0.25 * (seconds - c)
        } else {
            cost[depth] = seconds
        }
    }

    func cost(of depth: Int) -> Double? {
        lock.lock()
        defer { lock.unlock() }
        return cost[depth]
    }

    nonisolated(unsafe) private static var tables: [ObjectIdentifier: NativeMTPDepthCostTable] = [:]
    private static let tablesLock = NSLock()

    static func shared(for model: AnyObject) -> NativeMTPDepthCostTable {
        tablesLock.lock()
        defer { tablesLock.unlock() }
        let key = ObjectIdentifier(model)
        if let t = tables[key] { return t }
        let t = NativeMTPDepthCostTable()
        tables[key] = t
        return t
    }
}

struct NativeMTPDepthChooser {
    static let decay =
        Double(ProcessInfo.processInfo.environment["VMLX_MTP_DEPTH_CHOOSER_DECAY"] ?? "") ?? 0.95
    static let probeEvery = 64
    static let priorP = 0.7
    /// OPT-IN (VMLX_MTP_DEPTH_CHOOSER=1). Measured 2026-10-06 on 4S, interleaved at matched GPU clock:
    /// it did NOT beat the existing governor (code 58.3 vs 62.6, easy_prose 58.2 vs 68.2 @~1.1 GHz;
    /// 64.8 vs 73.2 @1.3 GHz with decay 0.85). Kept as an experiment platform; see the Swift handoff.
    static let enabled = ProcessInfo.processInfo.environment["VMLX_MTP_DEPTH_CHOOSER"] == "1"

    let maximumDepth: Int
    private(set) var depth: Int
    private var trials: [Double]
    private var hits: [Double]
    private var cycles = 0
    private var lastTimed: [Int: Int] = [:]
    private let costs: NativeMTPDepthCostTable

    init(initialDepth: Int, maximumDepth: Int, costs: NativeMTPDepthCostTable) {
        self.maximumDepth = Swift.max(1, maximumDepth)
        self.depth = Swift.min(Swift.max(1, initialDepth), self.maximumDepth)
        self.trials = Array(repeating: 0, count: self.maximumDepth + 1)
        self.hits = Array(repeating: 0, count: self.maximumDepth + 1)
        self.costs = costs
    }

    func expectedTokens(_ d: Int) -> Double {
        var e = 1.0
        var run = 1.0
        var prev = Self.priorP
        if d >= 1 {
            for k in 1 ... d {
                prev = (hits[k] + prev) / (trials[k] + 1.0)
                run *= prev
                e += run
            }
        }
        return e
    }

    /// One committed cycle at `cycleDepth` drafts: `accepted` drafts accepted; `seconds` = measured period
    /// (nil when the previous cycle was not adjacent, e.g. across an AR pause or a skipped cycle).
    /// `ceiling` bounds the next depth (an AR-safety demotion holds it down for a window).
    mutating func observe(cycleDepth: Int, accepted: Int, seconds: Double?, ceiling: Int) {
        for k in 1 ..< trials.count {
            trials[k] *= Self.decay
            hits[k] *= Self.decay
        }
        let lastDrafted = Swift.min(accepted + 1, cycleDepth)
        if lastDrafted >= 1 {
            for k in 1 ... lastDrafted where k < trials.count {
                trials[k] += 1
                if k <= accepted { hits[k] += 1 }
            }
        }
        cycles += 1
        if let seconds, seconds > 0, cycleDepth >= 1, cycleDepth <= maximumDepth {
            costs.observe(depth: cycleDepth, seconds: seconds)
            lastTimed[cycleDepth] = cycles
        }
        let top = Swift.max(1, Swift.min(maximumDepth, ceiling))
        let candidates = Array(1 ... top)
        if let unseen = candidates.first(where: { costs.cost(of: $0) == nil }) {
            depth = unseen
            return
        }
        let best =
            candidates.max {
                expectedTokens($0) / costs.cost(of: $0)! < expectedTokens($1) / costs.cost(of: $1)!
            } ?? depth
        if cycles % Self.probeEvery == 0 {
            let neighbours = [best - 1, best + 1].filter { $0 >= 1 && $0 <= top }
            if let stale = neighbours.min(by: { (lastTimed[$0] ?? -1) < (lastTimed[$1] ?? -1) }) {
                depth = stale
                return
            }
        }
        depth = best
    }

    var summary: String {
        (1 ... maximumDepth).map { d in
            String(
                format: "d%d:E=%.2f,c=%@", d, expectedTokens(d),
                costs.cost(of: d).map { String(format: "%.1fms", $0 * 1000) } ?? "-")
        }.joined(separator: " ")
    }
}
