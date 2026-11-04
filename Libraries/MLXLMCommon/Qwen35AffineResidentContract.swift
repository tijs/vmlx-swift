// Copyright © 2026 Apple Inc.
import Foundation
import MLXNN

/// Unwired research validation of already-sanitized resident tensor descriptors.
/// Capture before quantization/module replacement. The caller must independently
/// reject sanitizer key collisions, bind the exact checkpoint/config, and ensure
/// its pinned policy is the policy actually used by the loader. This validator
/// is not a loading or inference admission gate.
public struct Qwen35AffineResidentContract {
    public struct ContractError: Error, CustomStringConvertible {
        public let description: String
        init(_ description: String) { self.description = description }
    }
    public struct Tensor: Equatable {
        public let shape: [Int]
        /// Safetensors-style dtype, e.g. U32, BF16, F16, F32.
        public let dtype: String
        public init(shape: [Int], dtype: String) { self.shape = shape; self.dtype = dtype }
    }
    public struct Affine: Equatable {
        public let bits: Int
        public let groupSize: Int
        public init(bits: Int, groupSize: Int) { self.bits = bits; self.groupSize = groupSize }
    }
    private let expected: [String: [Int]]
    private let affine: [String: Affine]
    private let routed: Set<String>

    /// Shapes are the fresh pre-swap model parameter shapes, not checkpoint shapes.
    /// `affine` keys name the exact Linear/Embedding leaves quantized by the
    /// caller's pinned policy. Only affine 2/4/8 bit and groups 32/64/128 are supported.
    init(expected: [String: [Int]], affine: [String: Affine], routed: Set<String>, tiedHead: Bool) throws {
        guard !expected.isEmpty, !routed.isEmpty,
            routed.allSatisfy({ expected[$0] != nil && $0.hasSuffix(".weight")
                && AffineStreamingExpertCatalog.isRoutedTensorKey($0, layout: .qwen35) }),
            Set(expected.keys.filter { AffineStreamingExpertCatalog.isRoutedTensorKey($0, layout: .qwen35) }) == routed,
            (expected["language_model.lm_head.weight"] == nil) == tiedHead
        else { throw ContractError("invalid routed exclusions or tied-head contract") }
        for (key, shape) in expected {
            guard !key.isEmpty, shape.allSatisfy({ $0 >= 0 }) else { throw ContractError("invalid expected shape") }
        }
        for (path, quant) in affine {
            guard let shape = expected[path + ".weight"], shape.count == 2,
                !routed.contains(path + ".weight"), shape[0] > 0, shape[1] > 0,
                [2, 4, 8].contains(quant.bits), [32, 64, 128].contains(quant.groupSize),
                shape[1] % quant.groupSize == 0,
                !shape[1].multipliedReportingOverflow(by: quant.bits).overflow,
                shape[1] * quant.bits % 32 == 0,
                expected[path + ".scales"] == nil, expected[path + ".biases"] == nil
            else { throw ContractError("unsupported affine policy: \(path)") }
        }
        self.expected = expected; self.affine = affine; self.routed = routed
    }

    /// Capture is intended only for a freshly constructed Qwen35, before any
    /// update. The affine policy must resolve only to native Linear/Embedding.
    public static func capture(model: Module, affine: [String: Affine], routed: Set<String>, tiedHead: Bool) throws -> Self {
        let parameters = model.parameters().flattened()
        let expected = try uniqueEntries(parameters.map { ($0.0, $0.1.shape) })
        let leaves = try uniqueEntries(model.leafModules().flattened())
        for path in affine.keys {
            guard let leaf = leaves[path], type(of: leaf) == Linear.self || type(of: leaf) == Embedding.self else {
                throw ContractError("affine target is not a pristine Linear or Embedding: \(path)")
            }
        }
        return try Self(expected: expected, affine: affine, routed: routed, tiedHead: tiedHead)
    }

    static func uniqueEntries<Value>(_ entries: [(String, Value)]) throws -> [String: Value] {
        var result: [String: Value] = [:]
        for (key, value) in entries {
            guard result.updateValue(value, forKey: key) == nil else {
                throw ContractError("duplicate model path: \(key)")
            }
        }
        return result
    }

    public func validate(_ tensors: [String: Tensor]) throws {
        var required = Set(expected.keys).subtracting(routed)
        for (path, policy) in affine {
            let shape = expected[path + ".weight"]!
            let packed = Tensor(shape: [shape[0], shape[1] * policy.bits / 32], dtype: "U32")
            let scalesShape = [shape[0], shape[1] / policy.groupSize]
            guard tensors[path + ".weight"] == packed,
                let scales = tensors[path + ".scales"], let biases = tensors[path + ".biases"],
                scales.shape == scalesShape, biases.shape == scalesShape,
                ["BF16", "F16", "F32"].contains(scales.dtype), biases.dtype == scales.dtype
            else { throw ContractError("invalid affine tensors: \(path)") }
            required.insert(path + ".scales"); required.insert(path + ".biases")
        }
        guard Set(tensors.keys) == required else { throw ContractError("resident tensor key coverage mismatch") }
        for (key, shape) in expected where !routed.contains(key)
            && !(key.hasSuffix(".weight") && affine[String(key.dropLast(".weight".count))] != nil) {
            guard let tensor = tensors[key], tensor.shape == shape,
                ["BF16", "F16", "F32"].contains(tensor.dtype)
            else { throw ContractError("invalid dense tensor: \(key)") }
        }
    }
}
