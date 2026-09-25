// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

/// Parameter type for all MLX operations.
///
/// Use this to control where operations are evaluated:
///
/// ```swift
/// // produced on cpu
/// let a = MLXRandom.uniform([100, 100], stream: .cpu)
///
/// // produced on gpu
/// let b = MLXRandom.uniform([100, 100], stream: .gpu)
/// ```
///
/// If omitted it will use the ``default``, which will be ``Device/gpu`` unless
/// set otherwise.
///
/// ### See Also
/// - <doc:using-streams>
/// - ``Stream``
/// - ``Device``
public struct StreamOrDevice: Sendable, CustomStringConvertible, Equatable {

    public let stream: Stream

    private init(_ stream: Stream) {
        self.stream = stream
    }

    /// The default stream on the default device.
    ///
    /// This will be ``Device/gpu`` unless ``Device/setDefault(device:)``
    /// sets it otherwise.
    public static var `default`: StreamOrDevice {
        StreamOrDevice(Stream.defaultStream ?? Device.defaultStream())
    }

    public static func device(_ device: Device) -> StreamOrDevice {
        StreamOrDevice(Stream.defaultStream(device))
    }

    /// The ``Stream/defaultStream(_:)`` on the ``Device/cpu``
    public static let cpu = device(.cpu)

    /// The ``Stream/defaultStream(_:)`` on the ``Device/gpu``
    ///
    /// ### See Also
    /// - ``GPU``
    public static let gpu = device(.gpu)

    public static func stream(_ stream: Stream) -> StreamOrDevice {
        StreamOrDevice(stream)
    }

    /// Internal context -- used with Cmlx calls.
    public var ctx: mlx_stream {
        stream.ctx
    }

    public var description: String {
        stream.description
    }
}

/// A stream of evaluation attached to a particular device.
///
/// Typically this is used via the `stream: ` parameter on a method with a ``StreamOrDevice``:
///
/// ```swift
/// let a: MLXArray ...
/// let result = sqrt(a, stream: .gpu)
/// ```
///
/// Read more at <doc:using-streams>.
///
/// ### See Also
/// - <doc:using-streams>
/// - ``StreamOrDevice``
/// Box for passing Swift closures through C void* context.
private class ClosureBox {
    let closure: () -> Void
    init(_ closure: @escaping () -> Void) { self.closure = closure }
}

public final class Stream: @unchecked Sendable, Equatable {

    let ctx: mlx_stream

    // Swift tasks can resume on another OS thread. These streams use the
    // core's cross-thread encoder registry; evalLock serializes their use.
    private static func newStreamThreadUnsafe(_ type: mlx_device_type) -> mlx_stream {
        // Do not initialize Device.gpu/cpu here: Device owns these default
        // streams, so doing so would recursively initialize the static value.
        evalLock.withLock {
            let device = mlx_device_new_type(type, 0)
            defer { mlx_device_free(device) }
            return mlx_stream_new_thread_unsafe(device)
        }
    }

    public static let gpu = Stream(newStreamThreadUnsafe(MLX_GPU))
    public static let cpu = Stream(newStreamThreadUnsafe(MLX_CPU))

    @TaskLocal static var defaultStream: Stream?

    /// Set the ``StreamOrDevice/default`` scoped to a Task.
    public static func withNewDefaultStream<R>(device: Device? = nil, _ body: () throws -> R)
        rethrows -> R
    {
        let device = device ?? Device.defaultDevice()
        return try $defaultStream.withValue(Stream(device), operation: body)
    }

    /// Set the ``StreamOrDevice/default`` scoped to a Task.
    public static func withNewDefaultStream<R>(
        device: Device? = nil, _ body: () async throws -> R
    ) async rethrows -> R {
        let device = device ?? Device.defaultDevice()
        return try await $defaultStream.withValue(Stream(device), operation: body)
    }

    /// Set the C++ default stream on the current OS thread.
    ///
    /// Core 0.32 uses thread-local defaults. This affects native operations
    /// on this thread only, not Swift's explicit StreamOrDevice defaults.
    /// Do not span an async suspension; use withNewDefaultStream for Tasks.
    ///
    /// For native calls that omit their stream, bracket a synchronous scope
    /// with ``runWith(_:)``. For Swift model operations use
    /// ``withNewDefaultStream(device:_:)-5bwc3`` instead:
    /// ```swift
    /// Stream.withNewDefaultStream {
    ///     let result = model(input)  // Swift defaults select this Task's stream
    ///     asyncEval(result)
    ///     StreamOrDevice.default.stream.synchronize()
    /// }
    /// ```
    public static func setDefault(_ stream: Stream) {
        _ = evalLock.withLock { mlx_set_default_stream(stream.ctx) }
    }

    /// Restore the original default stream for the device.
    public static func restoreDefault(device: Device = Device.defaultDevice()) {
        let defaultStream = Stream.defaultStream(device)
        _ = evalLock.withLock { mlx_set_default_stream(defaultStream.ctx) }
    }

    /// Run a synchronous closure with this stream as the C++ thread-local default.
    ///
    /// Sets the C++ scheduler's default stream, runs the closure, restores the
    /// original in one C++ call, under the shared recursive evaluation lock.
    /// This does not override Swift's explicit `StreamOrDevice` parameters.
    ///
    /// - Parameter body: Closure to run with this stream as default.
    public func runWith(_ body: @escaping () -> Void) {
        // Use Unmanaged to pass the closure as a void* context to C
        let box = ClosureBox(body)
        let unmanaged = Unmanaged.passRetained(box)
        defer { unmanaged.release() }
        _ = evalLock.withLock {
            mlx_stream_run_with(ctx, { context in
                let box = Unmanaged<ClosureBox>.fromOpaque(context!).takeUnretainedValue()
                box.closure()
            }, unmanaged.toOpaque())
        }
    }

    init(_ ctx: mlx_stream) {
        self.ctx = ctx
    }

    /// The effective Swift default stream, including a scoped Task default.
    ///
    /// This shares the existing stream; it does not allocate a new queue.
    /// Use `Stream(device)` to create a new stream on a specified device.
    public init() {
        let selected = StreamOrDevice.default.stream
        self.ctx = evalLock.withLock {
            var ctx = mlx_stream_new()
            _ = mlx_stream_set(&ctx, selected.ctx)
            return ctx
        }
    }

    @available(*, deprecated, message: "use init(Device) -- index not supported")
    public init(index: Int32, _ device: Device) {
        self.ctx = evalLock.withLock {
            mlx_stream_new_thread_unsafe(device.ctx)
        }
    }

    /// New stream on the given device.
    ///
    /// See also ``withNewDefaultStream(device:_:)-5bwc3``
    public init(_ device: Device) {
        self.ctx = evalLock.withLock {
            mlx_stream_new_thread_unsafe(device.ctx)
        }
    }

    deinit {
        _ = evalLock.withLock {
            mlx_stream_free(ctx)
        }
    }

    /// Synchronize with the given stream
    public func synchronize() {
        _ = evalLock.withLock {
            mlx_synchronize(ctx)
        }
    }

    static public func defaultStream(_ device: Device) -> Stream {
        switch device.deviceType {
        case .cpu: .cpu
        case .gpu: .gpu
        default: fatalError("Unexpected device type: \(device)")
        }
    }

    public static func == (lhs: Stream, rhs: Stream) -> Bool {
        mlx_stream_equal(lhs.ctx, rhs.ctx)
    }
}

extension Stream: CustomStringConvertible {
    public var description: String {
        var s = mlx_string_new()
        mlx_stream_tostring(&s, ctx)
        defer { mlx_string_free(s) }
        return String(cString: mlx_string_data(s), encoding: .utf8)!
    }
}
