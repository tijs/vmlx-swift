#if canImport(Darwin)
import Foundation
import MLX

/// Chronological full-attention history for one layer or visit. Each complete
/// page is written once. A partial page stays in RAM, ready for the next append.
/// A failed append invalidates the generation that owns this history.
package final class ActiveKVSpillHistory {
    private let directory: URL
    package let pageTokens: Int
    private let byteLimit: Int
    private var file: ActiveKVSpillFile?
    private var sealed: [ActiveKVSpillFile] = []
    private var sealedPageCount = 0
    private var inheritedBytes = 0
    private var layout: Layout?
    private var tail: (keys: MLXArray, values: MLXArray)?
    package private(set) var count = 0
    package private(set) var bytesRead: UInt64 = 0
    package private(set) var bytesWritten: UInt64 = 0

    private struct Layout: Equatable {
        var keyShape: [Int]
        var valueShape: [Int]
        var keyType: DType
        var valueType: DType
        var keyBytes: Int
        var valueBytes: Int
    }

    package init(directory: URL, pageTokens: Int, byteLimit: Int) {
        precondition(pageTokens > 0 && byteLimit > 0)
        self.directory = directory
        self.pageTokens = pageTokens
        self.byteLimit = byteLimit
    }

    package func append(keys: MLXArray, values: MLXArray) throws {
        try Task.checkCancellation()
        guard keys.ndim == 4, values.ndim == 4,
              keys.dim(2) > 0, keys.dim(2) == values.dim(2) else {
            throw ActiveKVSpillFile.Failure.invalidGeometry
        }
        let tokens = keys.dim(2)
        var keyShape = keys.shape
        var valueShape = values.shape
        keyShape[2] = pageTokens
        valueShape[2] = pageTokens
        let (keyBytes, keyOverflow) = (keys.nbytes / tokens).multipliedReportingOverflow(by: pageTokens)
        let (valueBytes, valueOverflow) = (values.nbytes / tokens).multipliedReportingOverflow(by: pageTokens)
        let (pageBytes, pageOverflow) = keyBytes.addingReportingOverflow(valueBytes)
        guard !keyOverflow, !valueOverflow, !pageOverflow, pageBytes > 0 else {
            throw ActiveKVSpillFile.Failure.invalidGeometry
        }
        let layout = Layout(
            keyShape: keyShape, valueShape: valueShape, keyType: keys.dtype,
            valueType: values.dtype, keyBytes: keyBytes, valueBytes: valueBytes)
        if let existing = self.layout, existing != layout {
            throw ActiveKVSpillFile.Failure.invalidGeometry
        }
        self.layout = layout

        let joinedKeys = tail.map { concatenated([$0.keys, keys], axis: 2) } ?? keys
        let joinedValues = tail.map { concatenated([$0.values, values], axis: 2) } ?? values
        let rows = joinedKeys.dim(2)
        var start = 0
        if rows >= pageTokens {
            if file == nil {
                file = try ActiveKVSpillFile(directory: directory, pageBytes: pageBytes,
                                                  byteLimit: byteLimit - inheritedBytes)
            }
            guard let file else { throw ActiveKVSpillFile.Failure.invalidGeometry }
            while rows - start >= pageTokens {
                let end = start + pageTokens
                try withError { error in
                    var bytes = joinedKeys[.ellipsis, start..<end, 0...].asData().data
                    try error.check()
                    bytes.append(joinedValues[.ellipsis, start..<end, 0...].asData().data)
                    try error.check()
                    _ = try bytes.withUnsafeBytes(file.append)
                    bytesWritten += UInt64(pageBytes)
                }
                start = end
            }
        }
        if start == rows {
            tail = nil
        } else {
            try withError {
                let nextKeys = contiguous(joinedKeys[.ellipsis, start..., 0...])
                let nextValues = contiguous(joinedValues[.ellipsis, start..., 0...])
                eval(nextKeys, nextValues)
                tail = (nextKeys, nextValues)
            }
        }
        count += tokens
    }

    /// The consumer must evaluate anything it needs from this tile before it
    /// returns. Ranges use absolute key positions, including the resident tail.
    package func forEachTile(
        selectedPages: IndexSet? = nil,
        _ consume: (Range<Int>, MLXArray, MLXArray) throws -> Void
    ) throws {
        let filePages = sealedPageCount + (file?.pageCount ?? 0)
        if let layout, filePages > 0 {
            var bytes = Data(count: layout.keyBytes + layout.valueBytes)
            var pageBase = 0
            let files = sealed + (file.map { [$0] } ?? [])
            for segment in files {
                let segmentEnd = pageBase + segment.pageCount
                let readable = IndexSet(integersIn: pageBase..<segmentEnd)
                for page in selectedPages.map({ $0.intersection(readable) }) ?? readable {
                    try bytes.withUnsafeMutableBytes { try segment.read(page: page - pageBase, into: $0) }
                    bytesRead += UInt64(segment.pageBytes)
                    let keys = MLXArray(bytes.prefix(layout.keyBytes), layout.keyShape, dtype: layout.keyType)
                    let values = MLXArray(bytes.suffix(layout.valueBytes), layout.valueShape, dtype: layout.valueType)
                    let start = page * pageTokens
                    try consume(start..<(start + pageTokens), keys, values)
                }
                pageBase = segmentEnd
            }
        }
        if let tail, selectedPages?.contains(filePages) != false {
            try Task.checkCancellation()
            try consume((count - tail.keys.dim(2))..<count, tail.keys, tail.values)
        }
    }

    /// Split at an exact request boundary. Full pages become immutable for
    /// both branches; subsequent appends never mutate a shared descriptor.
    /// The bound limits descriptors even for indefinitely extended chats.
    package func fork(into empty: ActiveKVSpillHistory? = nil) -> ActiveKVSpillHistory? {
        guard (empty == nil || empty?.count == 0),
              (empty == nil || empty?.pageTokens == pageTokens),
              sealed.count + (file == nil ? 0 : 1) <= 8 else { return nil }
        let target = ActiveKVSpillHistory(
            directory: empty?.directory ?? directory,
            pageTokens: pageTokens, byteLimit: empty?.byteLimit ?? byteLimit)
        let currentBytes = file.map { $0.pageCount * $0.pageBytes } ?? 0
        let totalBytes = inheritedBytes + currentBytes
        guard totalBytes <= target.byteLimit else { return nil }
        if let file {
            if file.pageCount > 0 {
                sealed.append(file)
                sealedPageCount += file.pageCount
                inheritedBytes = totalBytes
            }
            self.file = nil
        }
        target.sealed = sealed
        target.sealedPageCount = sealedPageCount
        target.inheritedBytes = totalBytes
        target.layout = layout
        target.count = count
        target.tail = tail
        return target
    }

    package var storedBytes: Int {
        inheritedBytes + (file.map { $0.pageCount * $0.pageBytes } ?? 0)
    }

    /// Only the tail is resident in the model cache. Never rebuild full history
    /// through KVCache.state, snapshot copies, or generic cache quantization.
    package var residentState: [MLXArray] {
        tail.map { [$0.keys, $0.values] } ?? []
    }
}
#endif
