import Foundation
import Cmlx

private let explicitMLXTestMetalLibrary: Void = {
    if let path = ProcessInfo.processInfo.environment["VMLX_TEST_METALLIB_PATH"] {
        precondition(FileManager.default.isReadableFile(atPath: path))
        precondition(mlx_metal_set_metallib_path(path) == 0)
    }
}()

func prepareMLXTestMetalLibrary() {
    _ = explicitMLXTestMetalLibrary
}

let mlxTestSerializationQueue = DispatchQueue(label: "mlx.metal.test.serializer")

final class MLXTestSerializationToken {
    private let done: DispatchSemaphore
    private var unlocked = false

    init(done: DispatchSemaphore) {
        self.done = done
    }

    func unlock() {
        guard !unlocked else { return }
        unlocked = true
        done.signal()
    }

    deinit {
        unlock()
    }
}

func lockSerializedMLXTest() -> MLXTestSerializationToken {
    prepareMLXTestMetalLibrary()
    let started = DispatchSemaphore(value: 0)
    let done = DispatchSemaphore(value: 0)
    mlxTestSerializationQueue.async {
        started.signal()
        done.wait()
    }
    started.wait()
    return MLXTestSerializationToken(done: done)
}
