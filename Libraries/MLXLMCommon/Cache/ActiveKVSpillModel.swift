#if canImport(Darwin)
import Foundation

/// Fixed query and storage geometry for eager, single-sequence native KV.
public struct ActiveKVSpillGeometry: Sendable {
    public let pageTokens: Int
    public let prefillStepSize: Int
    public let diskBytesPerToken: Int
    public let residentBytesPerToken: Int
    public let fixedResidentBytes: Int

    package init(
        pageTokens: Int, prefillStepSize: Int, diskBytesPerToken: Int,
        residentBytesPerToken: Int, fixedResidentBytes: Int
    ) {
        self.pageTokens = pageTokens
        self.prefillStepSize = prefillStepSize
        self.diskBytesPerToken = diskBytesPerToken
        self.residentBytesPerToken = residentBytesPerToken
        self.fixedResidentBytes = fixedResidentBytes
    }

    /// Round up so every layer can finish its final page within the allowance.
    public func diskBytes(forTokens tokens: Int) -> Int? {
        guard tokens > 0 else { return nil }
        let pages = (tokens - 1) / pageTokens + 1
        let (pageBytes, pageOverflow) = pageTokens.multipliedReportingOverflow(by: diskBytesPerToken)
        let (bytes, overflow) = pages.multipliedReportingOverflow(by: pageBytes)
        return pageOverflow || overflow ? nil : bytes
    }

    /// Live history and staging only. Callers reserve model weights and
    /// non-attention workspace independently.
    public func residentBytes(forTokens tokens: Int) -> Int? {
        guard tokens > 0 else { return nil }
        let (growing, growthOverflow) = tokens.multipliedReportingOverflow(by: residentBytesPerToken)
        let (bytes, overflow) = fixedResidentBytes.addingReportingOverflow(growing)
        return growthOverflow || overflow ? nil : bytes
    }
}

public protocol ActiveKVSpillModel: LanguageModel {
    var activeKVSpillGeometry: ActiveKVSpillGeometry? { get }

    func newSpillCache(
        parameters: GenerateParameters?, directory: URL, pageTokens: Int, byteLimit: Int
    ) throws -> [KVCache]
}
#endif
