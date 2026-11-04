// Copyright © 2026 Apple Inc.
import MLX
import Testing
@testable import MLXLMCommon

struct ExperimentalExpertMappingSlotTests {
    final class Token {}
    enum Failure: Error { case operation, drain }

    @Test("failed checkpoint retains mapping through arena quarantine and recovery")
    func quarantine() throws {
        let gate = ExperimentalRequestArenaGate()
        weak var reference: Token?
        do {
            let _: Void = try gate.withRequest(operation: { arena in
                let slot = ExperimentalExpertMappingSlot(streams: [.cpu], maxBytes: 128)
                try arena.retain(slot)
                do {
                    let token = Token(); reference = token
                    try slot.install(token, bytes: 128)
                }
                try slot.checkpoint { throw Failure.operation }
                return .finished(())
            }, drain: { _ in throw Failure.drain }, invalidate: { _ in })
            Issue.record("failed checkpoint was accepted")
        } catch let error as ExperimentalRequestArenaCleanupError {
            #expect(error.primary is Failure)
            #expect(error.cleanup is Failure)
        }
        #expect(reference != nil)
        try gate.recover(drain: { _ in }, invalidate: { _ in })
        #expect(reference == nil)
    }

    @Test("successful drain retires slot; slot and byte budget fail closed")
    func checkpoints() throws {
        let gate = ExperimentalRequestArenaGate()
        weak var reference: Token?
        try gate.withRequest(operation: { arena in
            let slot = ExperimentalExpertMappingSlot(streams: [.cpu], maxBytes: 128)
            try arena.retain(slot)
            #expect(throws: (any Error).self) { try slot.checkCapacity(129) }
            do {
                let token = Token(); reference = token
                try slot.install(token, bytes: 128)
            }
            #expect(throws: (any Error).self) { try slot.checkCapacity(1) }
            #expect(reference != nil)
            try slot.checkpoint { }
            #expect(reference == nil)
            try slot.checkCapacity(128)
            return .finished(())
        }, drain: { _ in }, invalidate: { _ in })
    }
}
