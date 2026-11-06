import Foundation

/// Structured originating failure carried by terminal generation metadata.
/// The message is diagnostic data for the requesting client, not model output.
public struct GenerationFailure: LocalizedError, Sendable, Equatable {
    public enum Stage: String, Sendable { case preparation, decoding }
    public let stage: Stage
    public let cause: String

    public init(stage: Stage, cause: String) {
        self.stage = stage
        self.cause = cause
    }

    public var errorDescription: String? { cause }
}
