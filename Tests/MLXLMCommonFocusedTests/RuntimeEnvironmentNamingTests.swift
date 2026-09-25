// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The rename is only safe if the legacy spelling keeps working, and only finished if no read
// site is left on the old name alone. Neither is visible from a build, so both are asserted.

import Foundation
import MLXLMCommon
import Testing

@Suite("Runtime environment naming")
struct RuntimeEnvironmentNamingTests {

    @Test("the current spelling is read")
    func currentSpellingRead() {
        #expect(RuntimeEnvironment.value("VMLX_THING", in: ["VMLX_THING": "a"]) == "a")
    }

    @Test("live lookup matches snapshots without retaining stale environment values")
    func liveLookupParity() {
        let suffix = "LOOKUP_TEST_" + UUID().uuidString.replacingOccurrences(of: "-", with: "_")
        let current = "VMLX_" + suffix
        let legacy = "VMLINUX_" + suffix
        defer {
            unsetenv(current)
            unsetenv(legacy)
            unsetenv(suffix)
        }
        #expect(RuntimeEnvironment.value(current) == nil)
        setenv(legacy, "legacy", 1)
        #expect(RuntimeEnvironment.value(current) == "legacy")
        for raw in ["", "new", " true ", "0", "yes", "off", "한글 🌍"] {
            setenv(current, raw, 1)
            let snapshot = ProcessInfo.processInfo.environment
            #expect(RuntimeEnvironment.value(current) == RuntimeEnvironment.value(current, in: snapshot))
            for fallback in [false, true] {
                #expect(RuntimeEnvironment.flag(current, default: fallback)
                    == RuntimeEnvironment.flag(current, default: fallback, in: snapshot))
            }
        }
        let retained = RuntimeEnvironment.value(current)
        unsetenv(current)
        #expect(retained == "한글 🌍")
        #expect(RuntimeEnvironment.value(current) == "legacy")
        #expect(RuntimeEnvironment.value(suffix) == nil)
        setenv(suffix, "unprefixed", 1)
        #expect(RuntimeEnvironment.value(suffix) == "unprefixed")
        #expect(RuntimeEnvironment.value(suffix + "\u{0}ignored") == nil)
    }

    @Test("live lookup host-cost diagnostic against the previous dictionary path")
    func liveLookupHostCost() {
        let key = "VMLX_LOOKUP_COST_" + UUID().uuidString
        let iterations = 10_000
        var missing = 0
        for direct in [false, true, true, false] {
            let start = DispatchTime.now().uptimeNanoseconds
            for _ in 0..<iterations {
                let result = direct ? RuntimeEnvironment.value(key)
                    : RuntimeEnvironment.value(key, in: ProcessInfo.processInfo.environment)
                if result == nil { missing += 1 }
            }
            let elapsed = DispatchTime.now().uptimeNanoseconds - start
            print("ENV_LOOKUP_COST direct=\(direct) calls=\(iterations) ns=\(elapsed)")
        }
        #expect(missing == 4 * iterations)
    }

    /// The whole point of the migration: someone's existing script must not break.
    @Test("the legacy spelling is still honoured")
    func legacySpellingHonoured() {
        #expect(RuntimeEnvironment.value("VMLX_THING", in: ["VMLINUX_THING": "b"]) == "b")
        #expect(RuntimeEnvironment.flag("VMLX_THING", in: ["VMLINUX_THING": "1"]))
    }

    @Test("the current spelling wins when both are set")
    func currentWins() {
        let both = ["VMLX_THING": "new", "VMLINUX_THING": "old"]
        #expect(RuntimeEnvironment.value("VMLX_THING", in: both) == "new")
    }

    /// A name that is not ours gets no derived fallback — otherwise asking for `PATH` would
    /// quietly consult `VMLINUX_PATH`.
    @Test("an unprefixed name gets no legacy fallback")
    func unprefixedHasNoFallback() {
        #expect(RuntimeEnvironment.legacyName(of: "PATH") == nil)
        #expect(RuntimeEnvironment.value("PATH", in: ["VMLINUX_PATH": "x"]) == nil)
        #expect(RuntimeEnvironment.value("PATH", in: ["PATH": "x"]) == "x")
    }

    @Test("flag accepts the spellings the call sites already accepted")
    func flagSpellings() {
        for yes in ["1", "true", "yes", "on", "TRUE", " on "] {
            #expect(RuntimeEnvironment.flag("VMLX_F", in: ["VMLX_F": yes]), "\(yes)")
        }
        for no in ["0", "false", "no", "off", ""] {
            #expect(!RuntimeEnvironment.flag("VMLX_F", in: ["VMLX_F": no]), "\(no)")
        }
        #expect(RuntimeEnvironment.flag("VMLX_F", default: true, in: [:]))
    }

    /// The public constants must NAME the current variable. A consumer reads these to build a
    /// settings UI or documentation, so a legacy value here publishes the old name outward —
    /// which is exactly what this migration is for.
    @Test("public environment-name constants carry the current spelling")
    func publicConstantsAreCurrent() {
        let current: [String] = [
            AccelerationMode.environmentVariable,
            DeepseekV4ReasoningPolicy.rawMaxEnvironmentKey,
            DeepseekV4ReasoningPolicy.forceDirectRailEnvironmentKey,
            RuntimeMoETopKOverride.environmentVariable,
        ]
        for name in current {
            #expect(name.hasPrefix(RuntimeEnvironment.prefix), "\(name) is not a current name")
        }
        // And each keeps its legacy sibling, so the old spelling stays reachable.
        let legacy: [String] = [
            AccelerationMode.legacyEnvironmentVariable,
            DeepseekV4ReasoningPolicy.legacyRawMaxEnvironmentKey,
            DeepseekV4ReasoningPolicy.legacyForceDirectRailEnvironmentKey,
            RuntimeMoETopKOverride.legacyEnvironmentVariable,
        ]
        for (new, old) in zip(current, legacy) {
            #expect(RuntimeEnvironment.legacyName(of: new) == old, "\(new) -> \(old)")
        }
    }

    /// Reading the accelerator through the legacy spelling must still resolve. The read went
    /// through `environmentVariable` alone, so flipping that constant without widening the read
    /// would have dropped every existing user — and nothing else would have said so.
    @Test("the accelerator still resolves from the legacy variable")
    func acceleratorLegacyStillResolves() {
        #expect(
            AccelerationRuntime.requestedMode(
                environment: [AccelerationMode.legacyEnvironmentVariable: "auto"]) == .auto)
        #expect(
            AccelerationRuntime.requestedMode(
                environment: [AccelerationMode.environmentVariable: "auto"]) == .auto)
        #expect(AccelerationRuntime.requestedMode(environment: [:]) == .metal)
    }
}
