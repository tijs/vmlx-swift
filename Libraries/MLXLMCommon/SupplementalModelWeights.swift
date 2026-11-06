import MLXNN

/// Logical weights held outside Module.parameters(), for example ready mapped
/// packed expert banks. These counts are not resident bytes or a wired limit.
/// Each conforming leaf reports only its own tensors, excluding child modules.
public protocol SupplementalModelWeights: AnyObject {
    var supplementalWeightBytes: Int { get }
    var supplementalParameterCount: Int { get }
}

public struct ModelWeightAccounting: Sendable {
    public let parameterArrayBytes: Int
    public let supplementalMappedBytes: Int
    public var logicalWeightBytes: Int { parameterArrayBytes + supplementalMappedBytes }
}

extension Module {
    /// Metadata-only accounting. Never evaluates, copies, or wires mapped arrays.
    public func modelWeightAccounting() -> ModelWeightAccounting {
        let arrays = parameters().flattenedValues().reduce(0) { $0 + $1.nbytes }
        return ModelWeightAccounting(parameterArrayBytes: arrays,
            supplementalMappedBytes: supplementalWeightTotals().bytes)
    }

    func supplementalWeightTotals() -> (bytes: Int, parameters: Int) {
        var seen = Set<ObjectIdentifier>()
        var bytes = 0, parameters = 0
        for module in [self] + leafModules().flattenedValues() {
            guard let extra = module as? any SupplementalModelWeights,
                  seen.insert(ObjectIdentifier(extra)).inserted else { continue }
            precondition(extra.supplementalWeightBytes >= 0 && extra.supplementalParameterCount >= 0,
                         "Supplemental model weights must have nonnegative metadata counts")
            bytes += extra.supplementalWeightBytes
            parameters += extra.supplementalParameterCount
        }
        return (bytes, parameters)
    }
}
