// Copyright © 2026 Apple Inc.
import Foundation
import MLXNN

/// Unwired research-only validation of routed descriptors. Capture before
/// model updates and quantized module swaps; the caller must bind the model,
/// policy, metadata and sanitized names to a validated immutable checkpoint.
/// This type does not load or admit a model and does not validate resident
/// weights, sanitization collisions or actual loader quantization decisions.
/// Models with additional routed leaves, including native MTP experts, are
/// rejected; this prototype only covers the declared base-model layers.
public struct Qwen35AffineRoutedContract {
    public struct ContractError: Error, CustomStringConvertible {
        public let description: String
        init(_ description: String) { self.description = description }
    }
    public struct Policy: Equatable {
        public let bits: Int
        public let groupSize: Int
        public init(bits: Int, groupSize: Int) { self.bits = bits; self.groupSize = groupSize }
    }
    public struct Tensor: Equatable {
        public let shape: [Int]
        public let dtype: String
        public init(shape: [Int], dtype: String) { self.shape = shape; self.dtype = dtype }
    }
    private struct Projection {
        let shape: [Int]
        let policy: Policy
    }
    private let projections: [String: Projection]

    /// `policies` must cover the three exact switch projection paths for every
    /// layer. Expert count and dimensions are checked against model geometry.
    public static func capture(model: Module, layerCount: Int, numExperts: Int,
        inputDims: Int, hiddenDims: Int, policies: [String: Policy]) throws -> Self {
        guard layerCount > 0, numExperts > 0, inputDims > 0, hiddenDims > 0 else {
            throw ContractError("invalid model geometry")
        }
        var expected = Set<String>()
        for layer in 0..<layerCount {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                expected.insert("language_model.model.layers.\(layer).mlp.switch_mlp.\(projection)")
            }
        }
        guard Set(policies.keys) == expected else { throw ContractError("incomplete routed policy") }
        var leaves: [String: Module] = [:]
        for (path, module) in model.leafModules().flattened() {
            guard leaves.updateValue(module, forKey: path) == nil else {
                throw ContractError("duplicate model leaf: \(path)")
            }
        }
        // Reject additional routed leaves, including extra layers/projections.
        let actual = Set(leaves.keys.filter { $0.split(separator: ".").contains("switch_mlp") })
        guard actual == expected else { throw ContractError("model routed layout mismatch") }
        var projections: [String: Projection] = [:]
        for path in expected {
            guard let leaf = leaves[path], type(of: leaf) == SwitchLinear.self,
                let linear = leaf as? SwitchLinear, linear.bias == nil,
                let policy = policies[path], [2, 4, 8].contains(policy.bits),
                [32, 64, 128].contains(policy.groupSize)
            else { throw ContractError("unsupported routed leaf or affine policy: \(path)") }
            let down = path.hasSuffix(".down_proj")
            let input = down ? hiddenDims : inputDims
            let output = down ? inputDims : hiddenDims
            guard linear.numExperts == numExperts, linear.inputDims == input,
                linear.outputDims == output, linear.weight.shape == [numExperts, output, input],
                input % policy.groupSize == 0,
                !input.multipliedReportingOverflow(by: policy.bits).overflow,
                input * policy.bits % 32 == 0
            else { throw ContractError("invalid routed model geometry: \(path)") }
            projections[path] = Projection(shape: [numExperts, output, input], policy: policy)
        }
        return Self(projections: projections)
    }

    /// Pass only the routed tensors after collision-safe normalization. Every
    /// packed weight and BF16 affine companion is mandatory, with no extras.
    public func validate(_ tensors: [String: Tensor]) throws {
        var required = Set<String>()
        for (path, projection) in projections {
            let shape = projection.shape
            let packed = [shape[0], shape[1], shape[2] * projection.policy.bits / 32]
            let companions = [shape[0], shape[1], shape[2] / projection.policy.groupSize]
            guard tensors[path + ".weight"] == Tensor(shape: packed, dtype: "U32"),
                tensors[path + ".scales"] == Tensor(shape: companions, dtype: "BF16"),
                tensors[path + ".biases"] == Tensor(shape: companions, dtype: "BF16")
            else { throw ContractError("invalid routed affine descriptors: \(path)") }
            required.formUnion([path + ".weight", path + ".scales", path + ".biases"])
        }
        guard Set(tensors.keys) == required else { throw ContractError("routed key coverage mismatch") }
    }
}
