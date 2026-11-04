// Copyright © 2026 Apple Inc.
import Foundation
import XCTest
import Cmlx
@testable import MLX

final class ExperimentalStreamDrainTests: XCTestCase {
    enum Failure: Error { case injected }
    private final class CrossThreadState: @unchecked Sendable {
        let lock = NSLock()
        let submitted = DispatchSemaphore(value: 0)
        let finished = DispatchSemaphore(value: 0)
        var output: MLXArray?
        var error: String?
        var available = false
        var values: [Float] = []
    }

    func testStatusAndSwiftErrorPropagation() throws {
        try Stream.cpu.experimentalSynchronize(using: { 0 })
        XCTAssertThrowsError(try Stream.cpu.experimentalSynchronize(using: { 9 })) {
            XCTAssertEqual($0 as? ExperimentalStreamDrainError, .nonzeroStatus(9))
        }
        XCTAssertThrowsError(try Stream.cpu.experimentalSynchronize(using: { throw Failure.injected })) {
            XCTAssertTrue($0 is Failure)
        }
        try Stream.cpu.experimentalSynchronize()
    }

    func testSynchronousMLXCallbackIsScopedAndRecoverable() throws {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        XCTAssertThrowsError(try Stream.cpu.experimentalSynchronize(using: {
            var result = mlx_array_new()
            defer { mlx_array_free(result) }
            var shape = [Int32(1)]
            return path.withUnsafeFileSystemRepresentation { ptr -> Int32 in
                guard let ptr else { return 1 }
                return mlx_array_new_mmap_file_region(&result, ptr, 0, 4, &shape, 1, DType.float32.cmlxDtype)
            }
        })) { error in
            XCTAssertTrue(String(describing: error).contains(path.path))
        }
        try Stream.cpu.experimentalSynchronize(using: { 0 })
    }

    func testCPUStreamSubmittedAndDrainedOnDifferentThreads() {
        let stream = Stream(Device.cpu)
        let state = CrossThreadState()
        Thread.detachNewThread {
            let input = MLXArray((0..<4096).map { Float($0 % 7) })
            let output = MLX.add(input, MLXArray(Float(3)), stream: .stream(stream))
            MLX.asyncEval([output])
            state.lock.withLock { state.output = output }
            state.submitted.signal() // No more producers submit to this stream.
        }
        Thread.detachNewThread {
            defer { state.finished.signal() }
            guard state.submitted.wait(timeout: .now() + 5) == .success else {
                state.lock.withLock { state.error = "producer timed out" }
                return
            }
            do {
                try stream.experimentalSynchronize()
                guard let output = state.lock.withLock({ state.output }) else {
                    state.lock.withLock { state.error = "missing submitted output" }
                    return
                }
                var available = false
                let status = _mlx_array_is_available(&available, output.ctx)
                guard status == 0 && available else {
                    state.lock.withLock { state.error = "output not available after exact-stream drain" }
                    return
                }
                let values = output.asArray(Float.self)
                state.lock.withLock { state.available = available; state.values = values }
            } catch {
                state.lock.withLock { state.error = String(describing: error) }
            }
        }
        guard state.finished.wait(timeout: .now() + 10) == .success else {
            XCTFail("drain worker timed out")
            return
        }
        let (error, available, values) = state.lock.withLock { (state.error, state.available, state.values) }
        XCTAssertNil(error)
        XCTAssertTrue(available)
        XCTAssertEqual(values, (0..<4096).map { Float($0 % 7) + 3 })
    }
}
