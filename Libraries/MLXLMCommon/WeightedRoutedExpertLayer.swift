import MLX
import MLXNN

/// A routed bank that owns weighting as well as expert projection. This preserves
/// formats whose decode reduction must accumulate in F32 before its final cast.
/// The model retains responsibility for routing, shared experts and score dtype.
public protocol WeightedRoutedExpertLayer: Module {
    func callRouted(_ input: MLXArray, indices: MLXArray, scores: MLXArray) -> MLXArray
}

extension SwitchGLU: WeightedRoutedExpertLayer {
    public func callRouted(
        _ input: MLXArray, indices: MLXArray, scores: MLXArray
    ) -> MLXArray {
        if let fused = qwen4ExpReduced(input, indices: indices, scores: scores) {
            return fused
        }
        let routed = self(input, indices)
        return (routed * expandedDimensions(scores, axis: -1)).sum(axis: -2)
    }
}
