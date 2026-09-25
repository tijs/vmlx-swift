// Copyright © 2026
//
// Deterministic parity harness for the Bonsai 2 Prism-Hadamard block-1024
// transform: replays Source/MLXNN/PrismBonsaiHadamard.swift `hadamardFWHT`
// against the pinned-runtime fixture
// (Tests/MLXLMTests/Resources/PrismBonsaiPinnedFWHTFixture.json), which was
// produced by importing runtime/runtime.py `fwht` verbatim from
// prism-ml/Ternary-Bonsai-2-27B-mlx-2bit @
// 3f926b415992eaa2ae9dd7b573706494d6bbf787 (see
// scripts/generate-bonsai2-prism-fwht-fixture.py).
//
// This target depends only on MLX + MLXNN, so it validates the transform even
// when the full MLXLMTests target is blocked by unrelated pre-existing
// build issues (e.g. the MLXLMCommon Evaluate.swift type-check timeout).
//
// Usage, from the repo root:
//
//     swift run BonsaiPrismFWHTParity
//
// Exits non-zero on the first failing check.

import Foundation
import MLX
import MLXNN

private let pinnedRevision =
    "3f926b415992eaa2ae9dd7b573706494d6bbf787"
private let pinnedRepo = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
private let pinnedSource = "runtime/runtime.py"
private let pinnedBlock = 1024
private let pinnedTransform = "normalized-sylvester-walsh-hadamard"
private let pinnedSignMode = "explicit"
private let pinnedSignWidths = [5120, 6144, 17408]

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

private struct Fixture {
    let pin: [String: Any]
    let cases: [FixtureCase]
}

private func loadFixture(at url: URL) throws -> Fixture {
    let data = try Data(contentsOf: url)
    guard
        let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
        let pin = root["pin"] as? [String: Any],
        let rawCases = root["cases"] as? [[String: Any]]
    else {
        throw NSError(domain: "fixture", code: 1)
    }
    func floats(_ value: Any?) -> [Float] {
        (value as? [Any])?.compactMap { ($0 as? NSNumber)?.floatValue } ?? []
    }
    func int(_ value: Any?) -> Int {
        (value as? NSNumber)?.intValue ?? 0
    }
    func ints(_ value: Any?) -> [Int] {
        (value as? [Any])?.compactMap { ($0 as? NSNumber)?.intValue } ?? []
    }
    var cases: [FixtureCase] = []
    for raw in rawCases {
        let rt = floats(raw["roundtrip"])
        cases.append(
            FixtureCase(
                name: raw["name"] as? String ?? "",
                width: int(raw["width"]),
                shape: ints(raw["shape"]),
                signs: floats(raw["signs"]),
                x: floats(raw["x"]),
                forward: floats(raw["forward"]),
                inverse: floats(raw["inverse"]),
                roundtrip: rt.isEmpty ? nil : rt))
    }
    return Fixture(pin: pin, cases: cases)
}

private final class Checks {
    var failures = 0
    var total = 0

    func check(_ name: String, _ condition: Bool, detail: String = "") {
        total += 1
        if condition {
            print("PASS \(name)")
        } else {
            failures += 1
            print("FAIL \(name) \(detail)")
        }
    }

    func checkCase(
        _ name: String, _ actual: MLXArray, _ expected: MLXArray,
        _ expectedValues: [Float], testName: String
    ) {
        let maxAbs = MLX.max(MLX.abs(actual.asType(.float32) - expected.asType(.float32)))
            .item(Float.self)
        let maxRef = expectedValues.map { Swift.abs($0) }.max() ?? 0
        let tol = Float(0.01) * (1 + maxRef)
        check(
            "\(testName) [\(name)] maxAbsDiff=\(maxAbs) tol=\(tol)",
            maxAbs <= tol)
    }
}

private func run() throws {
    let checks = Checks()

    let fixtureURL: URL
    if CommandLine.arguments.count > 1 {
        fixtureURL = URL(fileURLWithPath: CommandLine.arguments[1])
    } else {
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        fixtureURL = cwd.appendingPathComponent(
            "Tests/MLXLMTests/Resources/PrismBonsaiPinnedFWHTFixture.json")
    }
    let fixture = try loadFixture(at: fixtureURL)

    // 1. Provenance pin
    checks.check(
        "fixture pin revision == \(pinnedRevision)",
        fixture.pin["revision"] as? String == pinnedRevision,
        detail: "got \(fixture.pin["revision"] ?? "nil")")
    checks.check(
        "fixture pin repo == \(pinnedRepo)",
        fixture.pin["repo"] as? String == pinnedRepo)
    checks.check(
        "fixture pin source == \(pinnedSource)",
        fixture.pin["runtime_source"] as? String == pinnedSource)
    checks.check(
        "fixture pin block == \(pinnedBlock)",
        (fixture.pin["block"] as? NSNumber)?.intValue == pinnedBlock)
    checks.check(
        "fixture pin transform == \(pinnedTransform)",
        fixture.pin["transform"] as? String == pinnedTransform)
    checks.check(
        "fixture pin sign_mode == \(pinnedSignMode)",
        fixture.pin["sign_mode"] as? String == pinnedSignMode)
    checks.check(
        "fixture pin sign_widths == \(pinnedSignWidths)",
        (fixture.pin["sign_widths"] as? [Any])?
            .compactMap { ($0 as? NSNumber)?.intValue } == pinnedSignWidths)
    checks.check("fixture has 4 cases", fixture.cases.count == 4)

    func fixtureCase(_ name: String) -> FixtureCase? {
        fixture.cases.first { $0.name == name }
    }

    // 2. Forward parity at the three real pack widths
    for name in ["pack-width-5120", "pack-width-6144", "pack-width-17408"] {
        guard let c = fixtureCase(name) else {
            checks.check("fixture case \(name) present", false)
            continue
        }
        let x = MLXArray(c.x).reshaped(c.shape)
        let actual = hadamardFWHT(
            x, block: pinnedBlock, signs: MLXArray(c.signs), inverse: false)
        let expected = MLXArray(c.forward).reshaped(c.shape)
        checks.checkCase(
            name, actual, expected, c.forward, testName: "forward parity")
    }

    // 3. Inverse parity
    for name in ["pack-width-5120", "pack-width-6144", "pack-width-17408"] {
        guard let c = fixtureCase(name) else { continue }
        let x = MLXArray(c.x).reshaped(c.shape)
        let actual = hadamardFWHT(
            x, block: pinnedBlock, signs: MLXArray(c.signs), inverse: true)
        let expected = MLXArray(c.inverse).reshaped(c.shape)
        checks.checkCase(
            name, actual, expected, c.inverse, testName: "inverse parity")
    }

    // 4. Round-trip semantics (reference round trip + input recovery)
    if let c = fixtureCase("pack-width-5120"), let rt = c.roundtrip {
        let x = MLXArray(c.x).reshaped(c.shape)
        let signs = MLXArray(c.signs)
        let swiftRT = hadamardFWHT(
            hadamardFWHT(x, block: pinnedBlock, signs: signs, inverse: false),
            block: pinnedBlock, signs: signs, inverse: true)
        checks.checkCase(
            c.name, swiftRT, MLXArray(rt).reshaped(c.shape), rt,
            testName: "round-trip parity vs pinned runtime")
        let diff = MLX.max(
            MLX.abs(swiftRT.asType(.float32) - x.asType(.float32))
        ).item(Float.self)
        checks.check(
            "round-trip recovers input [\(c.name)] maxAbsDiff=\(diff)",
            diff <= Float(0.01) * (1 + (c.x.map { Swift.abs($0) }.max() ?? 0)))
    }
    for name in ["pack-width-6144", "pack-width-17408"] {
        guard let c = fixtureCase(name) else { continue }
        let x = MLXArray(c.x).reshaped(c.shape)
        let signs = MLXArray(c.signs)
        let swiftRT = hadamardFWHT(
            hadamardFWHT(x, block: pinnedBlock, signs: signs, inverse: false),
            block: pinnedBlock, signs: signs, inverse: true)
        let diff = MLX.max(
            MLX.abs(swiftRT.asType(.float32) - x.asType(.float32))
        ).item(Float.self)
        checks.check(
            "round-trip recovers input [\(name)] maxAbsDiff=\(diff)",
            diff <= Float(0.01) * (1 + (c.x.map { Swift.abs($0) }.max() ?? 0)))
    }

    // 5. Normalization: scale 1/sqrt(1024), spike per 1024 block
    if let c = fixtureCase("normalization-spike-1024") {
        let expected = MLXArray(c.forward)
        checks.check(
            "fixture spike head == 32.0",
            expected[0].item(Float.self) == Float(32))
        let ones = MLXArray.ones([1, 1024], dtype: .float16)
        let actual = hadamardFWHT(
            ones, block: pinnedBlock, signs: MLXArray(c.signs), inverse: false)
        checks.checkCase(
            c.name, actual, expected, c.forward, testName: "normalization spike")
        let multi = MLXArray.ones([1, 2048], dtype: .float16)
        let multiOut = hadamardFWHT(
            multi, block: pinnedBlock,
            signs: MLXArray(Array(repeating: Float(1), count: 2048)), inverse: false)
        let values = multiOut.asType(.float32).reshaped([2048]).asArray(Float.self)
        checks.check(
            "two-block spike heads == 32.0",
            Swift.abs(values[0] - 32) < 0.01 && Swift.abs(values[1024] - 32) < 0.01,
            detail: "heads \(values[0]), \(values[1024])")
        var tailsClean = true
        for i in 1 ..< 1024
        where Swift.abs(values[i]) >= 0.01
            || Swift.abs(values[1024 + i]) >= 0.01
        {
            tailsClean = false
            break
        }
        checks.check("two-block spike tails zero", tailsClean)
    }

    // 6. Sign placement pinned to the reference order
    if let c = fixtureCase("pack-width-5120") {
        let x = MLXArray(c.x).reshaped(c.shape)
        let signs = MLXArray(c.signs)
        let expected = MLXArray(c.forward).reshaped(c.shape)
        let pinnedOrder = hadamardFWHT(
            x, block: pinnedBlock, signs: signs, inverse: false)
        checks.checkCase(
            c.name, pinnedOrder, expected, c.forward,
            testName: "sign placement (forward, pinned order)")
        let wrongOrder =
            hadamardTransform(
                x.asType(.float32).reshaped([-1, pinnedBlock]),
                scale: 1 / sqrt(Float(pinnedBlock))
            )
            .reshaped(c.shape)
            * signs.asType(.float32)
        let wrongDiff = MLX.max(
            MLX.abs(wrongOrder.asType(.float32) - expected.asType(.float32))
        )
        .item(Float.self)
        checks.check(
            "sign placement observable: wrong order diverges (\(wrongDiff))",
            wrongDiff > Float(0.01) * (1 + (c.forward.map { Swift.abs($0) }.max() ?? 0)))
    }
    if let c = fixtureCase("pack-width-6144") {
        let x = MLXArray(c.x).reshaped(c.shape)
        let signs = MLXArray(c.signs)
        let expected = MLXArray(c.inverse).reshaped(c.shape)
        let pinnedOrder = hadamardFWHT(
            x, block: pinnedBlock, signs: signs, inverse: true)
        checks.checkCase(
            c.name, pinnedOrder, expected, c.inverse,
            testName: "sign placement (inverse, pinned order)")
        let wrongOrder = hadamardFWHT(
            x.asType(.float32) * signs.asType(.float32),
            block: pinnedBlock, signs: MLXArray(Array(repeating: 1, count: c.width)),
            inverse: true)
        let wrongDiff = MLX.max(
            MLX.abs(wrongOrder.asType(.float32) - expected.asType(.float32))
        )
        .item(Float.self)
        checks.check(
            "sign placement observable: wrong order diverges (\(wrongDiff))",
            wrongDiff > Float(0.01) * (1 + (c.inverse.map { Swift.abs($0) }.max() ?? 0)))
    }

    print("checks: \(checks.total), failures: \(checks.failures)")
    if checks.failures > 0 {
        exit(1)
    }
}

try run()
print("BonsaiPrismFWHTParity: all checks passed")
