// Copyright © 2026
//
// Pinned-runtime parity tests for the Bonsai 2 Prism-Hadamard block-1024
// transform seam (Source/MLXNN/PrismBonsaiHadamard.swift `hadamardFWHT`).
//
// The fixture (Resources/PrismBonsaiPinnedFWHTFixture.json) is produced by
// scripts/generate-bonsai2-prism-fwht-fixture.py, which imports the PINNED
// bundled pack runtime runtime/runtime.py `fwht(x, block, signs,
// inverse=False)` verbatim from
// prism-ml/Ternary-Bonsai-2-27B-mlx-2bit @
// 3f926b415992eaa2ae9dd7b573706494d6bbf787 and evaluates it on deterministic
// synthetic activations with the pack's REAL hadamard.json sign vectors
// (widths 5120 / 6144 / 17408, block_size 1024, sign_mode "explicit").
//
// These tests therefore gate the Swift transform against the external
// pinned reference — not a Swift self-reference. No model weights, no
// network, no Metal server, no Bonsai payload.

import Foundation
import MLX
import MLXNN
import Testing

@Suite("Prism Bonsai pinned-runtime FWHT parity (block 1024)")
struct PrismBonsaiHadamardPinnedParityTests {

    /// The immutable pinned pack revision the fixture was computed from.
    private static let pinnedRevision =
        "3f926b415992eaa2ae9dd7b573706494d6bbf787"
    private static let pinnedRepo = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
    private static let pinnedSource = "runtime/runtime.py"
    private static let pinnedBlock = 1024
    private static let pinnedTransform = "normalized-sylvester-walsh-hadamard"
    private static let pinnedSignMode = "explicit"
    private static let pinnedSignWidths = [5120, 6144, 17408]

    // MARK: Fixture loading

    private struct FixtureCase {
        let name: String
        let width: Int
        let shape: [Int]
        let signs: [Float]
        let x: [Float]
        let forward: [Float]
        let inverse: [Float]
        let roundtrip: [Float]?
    }

    // Immutable fixture loaded once from the packaged JSON (contents are
    // never mutated); `nonisolated(unsafe)` documents that the payload's
    // [String: Any] shape is not Sendable by construction but is read-only
    // after load, so the concurrency-safety diagnostic is not applicable.
    private static nonisolated(unsafe) let fixture: (pin: [String: Any], cases: [FixtureCase]) = {
        guard
            let url = Bundle.module.url(
                forResource: "PrismBonsaiPinnedFWHTFixture", withExtension: "json"),
            let data = try? Data(contentsOf: url),
            let root = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any],
            let pin = root["pin"] as? [String: Any],
            let rawCases = root["cases"] as? [[String: Any]]
        else {
            Issue.record("pinned FWHT fixture failed to load")
            return (pin: [:], cases: [])
        }

        func floats(_ value: Any?) -> [Float] {
            guard let list = value as? [Any] else { return [] }
            return list.compactMap { ($0 as? NSNumber)?.floatValue }
        }
        func int(_ value: Any?) -> Int {
            (value as? NSNumber)?.intValue ?? 0
        }
        func ints(_ value: Any?) -> [Int] {
            guard let list = value as? [Any] else { return [] }
            return list.compactMap { ($0 as? NSNumber)?.intValue }
        }

        var cases: [FixtureCase] = []
        for raw in rawCases {
            cases.append(
                FixtureCase(
                    name: raw["name"] as? String ?? "",
                    width: int(raw["width"]),
                    shape: ints(raw["shape"]),
                    signs: floats(raw["signs"]),
                    x: floats(raw["x"]),
                    forward: floats(raw["forward"]),
                    inverse: floats(raw["inverse"]),
                    roundtrip: floats(raw["roundtrip"]).isEmpty
                        ? nil : floats(raw["roundtrip"])))
        }
        return (pin: pin, cases: cases)
    }()

    private func fixtureCase(_ name: String) -> Self.FixtureCase? {
        Self.fixture.cases.first { $0.name == name }
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        precondition(a.shape == b.shape, "shape mismatch \(a.shape) vs \(b.shape)")
        return MLX.max(MLX.abs(a.asType(.float32) - b.asType(.float32)))
            .item(Float.self)
    }

    /// f16-staged parity tolerance: the reference (mlx 0.32) and the Swift
    /// build (vMLX mlx fork) stage in float32 and cast back to float16, so
    /// results agree within a few float16 sample steps at the observed scale.
    private func parityTolerance(referenceMaxAbs: Float) -> Float {
        0.01 * (1 + referenceMaxAbs)
    }

    private func referenceMaxAbs(_ values: [Float]) -> Float {
        values.map { Swift.abs($0) }.max() ?? 0
    }

    // MARK: 1. Fixture provenance pin

    @Test("fixture provenance matches the pinned pack revision and manifest")
    func fixtureProvenanceIsPinned() throws {
        let pin = Self.fixture.pin
        #expect(pin["repo"] as? String == Self.pinnedRepo)
        #expect(pin["revision"] as? String == Self.pinnedRevision)
        #expect(pin["runtime_source"] as? String == Self.pinnedSource)
        #expect((pin["block"] as? NSNumber)?.intValue == Self.pinnedBlock)
        #expect(
            pin["transform"] as? String == Self.pinnedTransform)
        #expect(pin["sign_mode"] as? String == Self.pinnedSignMode)
        #expect(
            (pin["sign_widths"] as? [Any])?
                .compactMap { ($0 as? NSNumber)?.intValue }
                == Self.pinnedSignWidths)
        #expect(Self.fixture.cases.count == 4)
    }

    // MARK: 2. Forward FWHT parity against the pinned runtime

    @Test("forward block-1024 FWHT matches the pinned runtime at width 5120")
    func forwardParityWidth5120() throws {
        let c = try #require(fixtureCase("pack-width-5120"))
        let x = MLXArray(c.x).reshaped(c.shape)
        let actual = hadamardFWHT(
            x, block: Self.pinnedBlock, signs: MLXArray(c.signs), inverse: false)
        let expected = MLXArray(c.forward).reshaped(c.shape)
        // The transform casts back to the INPUT dtype (f16 pack contract);
        // the fixture arrays load as f32, so the invariant is
        // output-dtype == input-dtype.
        #expect(actual.dtype == x.dtype)
        #expect(actual.shape == expected.shape)
        #expect(
            maxAbsDiff(actual, expected)
                <= parityTolerance(referenceMaxAbs: referenceMaxAbs(c.forward)))
    }

    @Test("forward block-1024 FWHT matches the pinned runtime at width 6144")
    func forwardParityWidth6144() throws {
        let c = try #require(fixtureCase("pack-width-6144"))
        let x = MLXArray(c.x).reshaped(c.shape)
        let actual = hadamardFWHT(
            x, block: Self.pinnedBlock, signs: MLXArray(c.signs), inverse: false)
        #expect(
            maxAbsDiff(actual, MLXArray(c.forward).reshaped(c.shape))
                <= parityTolerance(referenceMaxAbs: referenceMaxAbs(c.forward)))
    }

    @Test("forward block-1024 FWHT matches the pinned runtime at width 17408")
    func forwardParityWidth17408() throws {
        let c = try #require(fixtureCase("pack-width-17408"))
        let x = MLXArray(c.x).reshaped(c.shape)
        let actual = hadamardFWHT(
            x, block: Self.pinnedBlock, signs: MLXArray(c.signs), inverse: false)
        #expect(
            maxAbsDiff(actual, MLXArray(c.forward).reshaped(c.shape))
                <= parityTolerance(referenceMaxAbs: referenceMaxAbs(c.forward)))
    }

    // MARK: 3. Inverse FWHT parity against the pinned runtime

    @Test("inverse block-1024 FWHT matches the pinned runtime at width 5120")
    func inverseParityWidth5120() throws {
        let c = try #require(fixtureCase("pack-width-5120"))
        let x = MLXArray(c.x).reshaped(c.shape)
        let actual = hadamardFWHT(
            x, block: Self.pinnedBlock, signs: MLXArray(c.signs), inverse: true)
        #expect(
            maxAbsDiff(actual, MLXArray(c.inverse).reshaped(c.shape))
                <= parityTolerance(referenceMaxAbs: referenceMaxAbs(c.inverse)))
    }

    @Test("inverse block-1024 FWHT matches the pinned runtime at width 6144")
    func inverseParityWidth6144() throws {
        let c = try #require(fixtureCase("pack-width-6144"))
        let x = MLXArray(c.x).reshaped(c.shape)
        let actual = hadamardFWHT(
            x, block: Self.pinnedBlock, signs: MLXArray(c.signs), inverse: true)
        #expect(
            maxAbsDiff(actual, MLXArray(c.inverse).reshaped(c.shape))
                <= parityTolerance(referenceMaxAbs: referenceMaxAbs(c.inverse)))
    }

    @Test("inverse block-1024 FWHT matches the pinned runtime at width 17408")
    func inverseParityWidth17408() throws {
        let c = try #require(fixtureCase("pack-width-17408"))
        let x = MLXArray(c.x).reshaped(c.shape)
        let actual = hadamardFWHT(
            x, block: Self.pinnedBlock, signs: MLXArray(c.signs), inverse: true)
        #expect(
            maxAbsDiff(actual, MLXArray(c.inverse).reshaped(c.shape))
                <= parityTolerance(referenceMaxAbs: referenceMaxAbs(c.inverse)))
    }

    // MARK: 4. Inverse/round-trip semantics

    @Test("forward-then-inverse FWHT matches the pinned runtime round trip")
    func roundTripMatchesPinnedRuntime() throws {
        let c = try #require(fixtureCase("pack-width-5120"))
        let rt = try #require(c.roundtrip)
        let x = MLXArray(c.x).reshaped(c.shape)
        let signs = MLXArray(c.signs)
        let swiftRoundTrip = hadamardFWHT(
            hadamardFWHT(x, block: Self.pinnedBlock, signs: signs, inverse: false),
            block: Self.pinnedBlock, signs: signs, inverse: true)
        let expected = MLXArray(rt).reshaped(c.shape)
        #expect(
            maxAbsDiff(swiftRoundTrip, expected)
                <= parityTolerance(referenceMaxAbs: referenceMaxAbs(rt)))
    }

    @Test("forward-then-inverse FWHT recovers the input at every pack width")
    func roundTripRecoversInput() throws {
        for name in ["pack-width-5120", "pack-width-6144", "pack-width-17408"] {
            let c = try #require(fixtureCase(name))
            let x = MLXArray(c.x).reshaped(c.shape)
            let signs = MLXArray(c.signs)
            let roundTripped = hadamardFWHT(
                hadamardFWHT(
                    x, block: Self.pinnedBlock, signs: signs, inverse: false),
                block: Self.pinnedBlock, signs: signs, inverse: true)
            // Orthonormal H with unit signs: inverse(forward(x)) == x (the
            // pinned runtime is an exact involution, see fixture roundtrip).
            #expect(
                maxAbsDiff(roundTripped, x)
                    <= parityTolerance(referenceMaxAbs: referenceMaxAbs(c.x)),
                "round trip diverged for \(name)")
        }
    }

    // MARK: 5. Normalization (scale 1/sqrt(block), per-1024 blocks)

    @Test("normalization spike matches the pinned runtime at scale 1/sqrt(1024)")
    func normalizationSpikeScale() throws {
        let c = try #require(fixtureCase("normalization-spike-1024"))
        // Fixture: H_1024/32 of all-ones with unit signs == [32, 0, ..., 0].
        let expected = MLXArray(c.forward)
        #expect(expected[0].item(Float.self) == Float(32))
        let ones = MLXArray.ones([1, 1024], dtype: .float16)
        let actual = hadamardFWHT(
            ones, block: Self.pinnedBlock, signs: MLXArray(c.signs), inverse: false)
        #expect(
            maxAbsDiff(actual, expected.reshaped(c.shape))
                <= parityTolerance(referenceMaxAbs: referenceMaxAbs(c.forward)))
        // The transform is blockwise: the spike lands once per 1024 block.
        let multi = MLXArray.ones([1, 2048], dtype: .float16)
        let multiOut = hadamardFWHT(
            multi, block: Self.pinnedBlock,
            signs: MLXArray(Array(repeating: Float(1), count: 2048)), inverse: false)
        let values = multiOut.asType(.float32).reshaped([2048])
            .asArray(Float.self)
        #expect(Swift.abs(values[0] - 32) < 0.01)
        #expect(Swift.abs(values[1024] - 32) < 0.01)
        for i in 1 ..< 1024 {
            #expect(Swift.abs(values[i]) < 0.01)
            #expect(Swift.abs(values[1024 + i]) < 0.01)
        }
    }

    // MARK: 6. Sign placement is pinned to the reference order

    @Test("sign placement matches the pinned runtime order (pre-transform forward)")
    func signPlacementPinnedForward() throws {
        let c = try #require(fixtureCase("pack-width-5120"))
        let x = MLXArray(c.x).reshaped(c.shape)
        let signs = MLXArray(c.signs)
        // Pinned order: forward is H((x * signs)) * scale.
        let pinnedOrder = hadamardFWHT(
            x, block: Self.pinnedBlock, signs: signs, inverse: false)
        let expected = MLXArray(c.forward).reshaped(c.shape)
        #expect(
            maxAbsDiff(pinnedOrder, expected)
                <= parityTolerance(referenceMaxAbs: referenceMaxAbs(c.forward)))
        // Wrong order (transform first, then signs) must NOT satisfy the
        // pinned reference — this is what makes sign placement observable.
        let wrongOrder =
            hadamardTransform(
                x.asType(.float32).reshaped([-1, Self.pinnedBlock]),
                scale: 1 / sqrt(Float(Self.pinnedBlock))
            )
            .reshaped(c.shape)
            * signs.asType(.float32)
        #expect(
            maxAbsDiff(wrongOrder, expected)
                > parityTolerance(referenceMaxAbs: referenceMaxAbs(c.forward)))
    }

    @Test("sign placement matches the pinned runtime order (post-transform inverse)")
    func signPlacementPinnedInverse() throws {
        let c = try #require(fixtureCase("pack-width-6144"))
        let x = MLXArray(c.x).reshaped(c.shape)
        let signs = MLXArray(c.signs)
        // Pinned order: inverse is H(x) * scale, then * signs.
        let pinnedOrder = hadamardFWHT(
            x, block: Self.pinnedBlock, signs: signs, inverse: true)
        let expected = MLXArray(c.inverse).reshaped(c.shape)
        #expect(
            maxAbsDiff(pinnedOrder, expected)
                <= parityTolerance(referenceMaxAbs: referenceMaxAbs(c.inverse)))
        // Wrong order (signs before the transform in inverse) must not match.
        let wrongOrder = hadamardFWHT(
            x.asType(.float32) * signs.asType(.float32),
            block: Self.pinnedBlock,
            signs: MLXArray(Array(repeating: Float(1), count: c.width)),
            inverse: true)
        #expect(
            maxAbsDiff(wrongOrder, expected)
                > parityTolerance(referenceMaxAbs: referenceMaxAbs(c.inverse)))
    }
}
