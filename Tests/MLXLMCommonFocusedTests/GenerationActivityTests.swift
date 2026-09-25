import Foundation
import Testing
@testable import MLXLMCommon

#if os(macOS)
    private final class ActivityRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var started = 0
        private var ended = 0
        private var live: Set<ObjectIdentifier> = []
        private var options: [ProcessInfo.ActivityOptions] = []

        func make(disabled: Bool = false) -> GenerationActivity {
            GenerationActivity(
                disabled: disabled,
                begin: { options, _ in
                    let token = NSObject()
                    self.lock.withLock {
                        self.started += 1
                        self.live.insert(ObjectIdentifier(token))
                        self.options.append(options)
                    }
                    return token
                },
                end: { token in
                    self.lock.withLock {
                        self.ended += 1
                        #expect(self.live.remove(ObjectIdentifier(token)) != nil)
                    }
                })
        }

        var counts: (started: Int, ended: Int, live: Int) {
            lock.withLock { (started, ended, live.count) }
        }

        var recordedOptions: [ProcessInfo.ActivityOptions] {
            lock.withLock { options }
        }
    }

    @Suite("Finite generation process activity")
    struct GenerationActivityTests {
        @Test func normalCompletionCoversTailAndEndsExactlyOnce() {
            let recorder = ActivityRecorder()
            func work() {
                let activity = recorder.make()
                defer { activity.end() }
                // Visible completion does not end the scope before cache tail.
                #expect(recorder.counts.live == 1)
                #expect(recorder.counts.ended == 0)
            }
            work()
            #expect(recorder.counts.started == 1)
            #expect(recorder.counts.ended == 1)
            #expect(recorder.counts.live == 0)
            #expect(recorder.recordedOptions == [.userInitiatedAllowingIdleSystemSleep])
            #expect(!GenerationActivity.options.contains(.idleSystemSleepDisabled))
            #expect(!GenerationActivity.options.contains(.idleDisplaySleepDisabled))
        }

        @Test func throwingPreparationEndsActivity() {
            enum Failure: Error { case preparation }
            let recorder = ActivityRecorder()
            func work() throws {
                let activity = recorder.make()
                defer { activity.end() }
                throw Failure.preparation
            }
            #expect(throws: Failure.self) { try work() }
            #expect(recorder.counts.ended == 1)
            #expect(recorder.counts.live == 0)
        }

        @Test func cancellationEndsActivity() async {
            let recorder = ActivityRecorder()
            let (started, continuation) = AsyncStream<Void>.makeStream()
            let task = Task {
                let activity = recorder.make()
                defer { activity.end() }
                continuation.yield(())
                continuation.finish()
                try await Task.sleep(for: .seconds(10))
            }
            for await _ in started { break }
            #expect(recorder.counts.live == 1)
            task.cancel()
            do {
                try await task.value
                Issue.record("Expected cancellation")
            } catch {
                #expect(error is CancellationError)
            }
            #expect(recorder.counts.ended == 1)
            #expect(recorder.counts.live == 0)
        }

        @Test func overlappingScopesDoNotReleaseOneAnother() {
            let recorder = ActivityRecorder()
            let first = recorder.make()
            let second = recorder.make()
            #expect(recorder.counts.live == 2)
            first.end()
            first.end()
            #expect(recorder.counts.ended == 1)
            #expect(recorder.counts.live == 1)
            second.end()
            #expect(recorder.counts.ended == 2)
            #expect(recorder.counts.live == 0)
        }

        @Test func diagnosticDisableDoesNotAcquireActivity() {
            let recorder = ActivityRecorder()
            let activity = recorder.make(disabled: true)
            activity.end()
            #expect(recorder.counts.started == 0)
            #expect(recorder.counts.ended == 0)
        }

        @Test func unendedLeaseIsReleasedOnDestruction() {
            let recorder = ActivityRecorder()
            func work() {
                let activity = recorder.make()
                withExtendedLifetime(activity) { #expect(recorder.counts.live == 1) }
            }
            work()
            #expect(recorder.counts.ended == 1)
            #expect(recorder.counts.live == 0)
        }
    }
#endif
