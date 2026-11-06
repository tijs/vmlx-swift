import Foundation
import MLX

/// Invalid prepared model output must fail generation before the rank-three
/// last-position subscript. This does not reshape, replace, or sample it.
enum PreparedLogitsValidationError: LocalizedError, Equatable {
    case invalidShape([Int])

    var errorDescription: String? {
        switch self {
        case .invalidShape(let shape):
            return
                "Prepared logits require nonempty [batch, sequence, vocabulary] dimensions; received \(shape)."
        }
    }
}

/// Call only after checking the error scope that owns model.prepare. Shape
/// inspection is not a recovery mechanism for an already failed MLX array.
func validatePreparedLogitsForSampling(_ logits: MLXArray) throws {
    let shape = logits.shape
    guard shape.count == 3, shape.allSatisfy({ $0 > 0 }) else {
        throw PreparedLogitsValidationError.invalidShape(shape)
    }
}
