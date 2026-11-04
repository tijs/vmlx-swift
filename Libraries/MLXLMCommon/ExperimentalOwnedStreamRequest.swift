// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// A caller-supplied handle and its exact synchronous stream drain. The drain
/// closure must retain the underlying stream owner and synchronize that stream.
/// A successful callback does not independently prove GPU error reporting.
public struct ExperimentalOwnedStream {
    fileprivate let drain: () throws -> Void
    fileprivate let typedStream: MLX.Stream?
    public init(drain: @escaping () throws -> Void) { self.drain = drain; self.typedStream = nil }
    /// Drains this exact MLX stream through its experimental error boundary.
    public init(stream: MLX.Stream) {
        self.drain = { try stream.experimentalSynchronize() }
        self.typedStream = stream
    }
}

public struct ExperimentalOwnedStreamsDrainError: Error {
    public struct Failure: Sendable {
        /// Zero-based index in the stream list supplied at admission.
        public let streamIndex: Int
        public let error: any Error
    }
    public let failures: [Failure]
}

public enum ExperimentalOwnedStreamRequestError: Error {
    case missingStreamOwner
    case emptyStreams
    case streamContextMismatch
}

// The initializer is private to this file: callers cannot forge a binding in a
// generic arena or substitute callback-only drains for these exact streams.
private final class ExperimentalBoundStreamContext {
    let streams: [MLX.Stream]
    init(streams: [MLX.Stream]) { self.streams = streams }
}

/// Owns callbacks and stream handles so their captures survive failed cleanup.
private final class ExperimentalStreamBundle {
    let streams: [ExperimentalOwnedStream]
    let operation: (ExperimentalRequestArena) throws -> Void
    let invalidate: () throws -> Void
    init(streams: [ExperimentalOwnedStream], operation: @escaping (ExperimentalRequestArena) throws -> Void,
         invalidate: @escaping () throws -> Void) {
        self.streams = streams
        self.operation = operation
        self.invalidate = invalidate
    }
    func drainAll() throws {
        var failures: [ExperimentalOwnedStreamsDrainError.Failure] = []
        for (index, stream) in streams.enumerated() {
            do { try stream.drain() }
            catch { failures.append(.init(streamIndex: index, error: error)) }
        }
        if !failures.isEmpty { throw ExperimentalOwnedStreamsDrainError(failures: failures) }
    }
}

/// Research-only wrapper around the arena gate. All supplied stream owners,
/// operation captures and invalidation captures are retained before operation.
/// Every supplied drain is attempted, including after another drain fails.
/// Recovery retries the same drains, then invalidates while admission is held.
///
/// Callers must supply all exact streams and register/retain every other GPU
/// resource before submission, stop and join submitting workers before return,
/// and prevent other serving paths and escaped aliases from using those objects.
/// Invalidation receives no arena: dynamic state that it must reset must be
/// reachable through its own retained capture, established before submission.
/// This API accepts no model parameter and returns no source, but cannot prevent
/// closure captures from escaping or prove retention of MLX temporaries. The
/// dedicated gate does not serialize the separate generic arena gate or other
/// serving paths. A drain callback reporting success does not prove GPU error detection. Swift
/// task cancellation is checked before and after operation; cancellation after
/// the latter check can race with successful cleanup and return.
public enum ExperimentalOwnedStreamRequest {
    // Dedicated owner prevents generic arena recovery from releasing this
    // quarantine without retrying its registered stream callbacks.
    private static let ownedGate = ExperimentalRequestArenaGate()
    public static func run(
        streams: [ExperimentalOwnedStream],
        operation: @escaping (ExperimentalRequestArena) throws -> Void,
        invalidate: @escaping () throws -> Void
    ) throws {
        try run(gate: ownedGate, streams: streams, operation: operation, invalidate: invalidate)
    }

    static func run(
        gate: ExperimentalRequestArenaGate,
        streams: [ExperimentalOwnedStream],
        operation: @escaping (ExperimentalRequestArena) throws -> Void,
        invalidate: @escaping () throws -> Void
    ) throws {
        guard !streams.isEmpty else { throw ExperimentalOwnedStreamRequestError.emptyStreams }
        let bundle = ExperimentalStreamBundle(streams: streams, operation: operation, invalidate: invalidate)
        try gate.withRequest(operation: { arena in
            try arena.retain(bundle)
            let typedStreams = streams.compactMap { $0.typedStream }
            if !typedStreams.isEmpty {
                try arena.retain(ExperimentalBoundStreamContext(streams: typedStreams))
            }
            try Task<Never, Never>.checkCancellation()
            try bundle.operation(arena)
            try Task<Never, Never>.checkCancellation()
            return .finished(())
        }, drain: { _ in try bundle.drainAll() },
           invalidate: { _ in try bundle.invalidate() })
    }

    /// Reject generic arenas and stream lists other than the immutable typed
    /// stream list owned and drained by this request. This does not prove that
    /// callers submitted work only to those streams. This check is not an
    /// in-flight operation lease: it does not prevent sealing immediately
    /// afterward. The operation must stop and join all concurrent submitters
    /// before returning or throwing.
    public static func validateMappingStreams(arena: ExperimentalRequestArena, streams: [MLX.Stream]) throws {
        guard arena.isOpen(),
              let context = arena.registeredResource(ofType: ExperimentalBoundStreamContext.self),
              !context.streams.isEmpty, context.streams == streams else {
            throw ExperimentalOwnedStreamRequestError.streamContextMismatch
        }
    }

    public static func recover() throws { try recover(gate: ownedGate) }

    static func recover(gate: ExperimentalRequestArenaGate) throws {
        try gate.recover(drain: { arena in
            guard let bundle = arena.registeredResource(ofType: ExperimentalStreamBundle.self) else {
                throw ExperimentalOwnedStreamRequestError.missingStreamOwner
            }
            try bundle.drainAll()
        }, invalidate: { arena in
            guard let bundle = arena.registeredResource(ofType: ExperimentalStreamBundle.self) else {
                throw ExperimentalOwnedStreamRequestError.missingStreamOwner
            }
            try bundle.invalidate()
        })
    }
}
