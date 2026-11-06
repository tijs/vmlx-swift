import MLX
import MLXNN

/// Explicit diagnostic: selected B1/top8 decode with host route readback, and
/// evaluated per-layer whole-bank prefill. No compiled-execution qualification.
/// Only the shared owner cache retains packed expert views between calls.
final class JANGHSelectedRoutedExpertLayer: Module, WeightedRoutedExpertLayer, SupplementalModelWeights {
    private let source: JANGHMappedBanks.SourceLease
    private let owner: JANGHExpertMappedBanks
    private let kernel: JANGHSelectedExpertDecode
    private let parent: String
    private let gateModule: String, upModule: String, downModule: String
    private let width: Int
    private let routes: Int
    private let limit: Float?
    let supplementalWeightBytes: Int
    let supplementalParameterCount: Int

    init(source: JANGHMappedBanks.SourceLease, owner: JANGHExpertMappedBanks,
         parentModule: String, inputDimensions: Int, activationLimit: Float?, routes: Int = 8) throws {
        guard (1 ... JANGHSelectedExpertDecode.maxRoutes).contains(routes) else {
            throw JANGHFormatContract.ValidationError.invalid("selected routed layer route count out of range")
        }
        guard owner.isBacked(by: source), inputDimensions > 0, activationLimit == nil || (activationLimit!.isFinite && activationLimit! > 0) else {
            throw JANGHFormatContract.ValidationError.invalid("invalid selected routed layer dimensions or clamp")
        }
        let gate = parentModule + ".gate_proj", up = parentModule + ".up_proj", down = parentModule + ".down_proj"
        var bytes = 0, parameters = 0
        for module in [gate, up, down] {
            guard let pair = source.plan.projections[module], let projection = source.contract.projections[module],
                  let header = source.metadata.shards[pair.packed.shard]?.tensors[pair.packed.tensor] else {
                throw JANGHFormatContract.ValidationError.invalid("missing selected routed layer projection")
            }
            let input = header.shape[2] * 32 / projection.bits
            guard module == down ? header.shape[1] == inputDimensions : input == inputDimensions else {
                throw JANGHFormatContract.ValidationError.invalid("selected routed layer width mismatch")
            }
            let bankBytes = pair.packed.byteCount.addingReportingOverflow(pair.scales.byteCount)
            let totalBytes = bytes.addingReportingOverflow(bankBytes.partialValue)
            let rows = header.shape[0].multipliedReportingOverflow(by: header.shape[1])
            let count = rows.partialValue.multipliedReportingOverflow(by: input)
            let totalCount = parameters.addingReportingOverflow(count.partialValue)
            guard !bankBytes.overflow, !totalBytes.overflow, !rows.overflow, !count.overflow, !totalCount.overflow else {
                throw JANGHFormatContract.ValidationError.invalid("selected routed accounting overflow")
            }
            bytes = totalBytes.partialValue
            parameters = totalCount.partialValue
        }
        self.source = source; self.owner = owner; parent = parentModule
        gateModule = gate; upModule = up; downModule = down
        width = inputDimensions; limit = activationLimit; self.routes = routes
        supplementalWeightBytes = bytes; supplementalParameterCount = parameters
        kernel = try JANGHSelectedExpertDecode(owner: owner, gateModule: gate, upModule: up, downModule: down, routes: routes)
        super.init()
    }

    public func callRouted(_ input: MLXArray, indices: MLXArray, scores: MLXArray) -> MLXArray {
        do { return try routed(input, indices: indices, scores: scores) }
        catch { preconditionFailure("selected JANGH diagnostic rejected execution: \(error)") }
    }

    func routed(_ input: MLXArray, indices: MLXArray, scores: MLXArray) throws -> MLXArray {
        guard input.ndim >= 2, input.dim(-1) == width, indices.dtype == .uint32,
              indices.ndim == input.ndim, indices.shape.dropLast() == input.shape.dropLast(),
              indices.dim(-1) == self.routes, scores.shape == indices.shape else {
            throw JANGHFormatContract.ValidationError.invalid("selected diagnostic requires the admitted top-k shape")
        }
        let flat = input.reshaped(-1, width), routes = indices.reshaped(-1, self.routes)
        let weights = scores.reshaped(routes.shape).asType(.float32)
        if flat.dim(0) == 1 {
            // This synchronization is explicit and included in full-model timing.
            let ids = routes.asArray(UInt32.self)
            let gate = try owner.selection(module: gateModule, expertIDs: ids)
            let up = try owner.selection(module: upModule, expertIDs: ids)
            let down = try owner.selection(module: downModule, expertIDs: ids)
            let hidden = try kernel.activatePreparedInput(kernel.prepareInput(flat), gate: gate, up: up, limit: limit)
            return try kernel.projectPreparedHidden(hidden, down: down, scores: weights, outputDType: input.dtype)
                .reshaped(input.shape)
        }
        // The previous layer's output is evaluated before its bank owner drops.
        // No whole-model bank owner is retained alongside the selected path.
        let banks = try JANGHMappedBanks(source: source, modules: [gateModule, upModule, downModule])
        let block = try JANGHRoutedDecodeBlock(banks: banks, parentModule: parent, activationLimit: limit)
        let output = try block.routed(flat, indices: routes, scores: weights, outputDType: input.dtype)
        eval(output)
        withExtendedLifetime(banks) {}
        return output.reshaped(input.shape)
    }
}
