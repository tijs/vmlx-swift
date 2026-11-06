import MLX
import MLXNN
import XCTest
@testable import MLXLMCommon

final class SupplementalModelWeightsTests: XCTestCase {
    private final class Bank: Module, SupplementalModelWeights {
        let supplementalWeightBytes = 96 * 1024 * 1024 * 1024
        let supplementalParameterCount = 384 * 1024 * 1024 * 1024
    }
    private final class SharedOwner: Module {
        @ModuleInfo var banks: [Bank]
        init(_ bank: Bank) {
            _banks.wrappedValue = [bank, bank]
            super.init()
        }
    }

    func testOpaqueWeightsCountOnceWithoutChangingOrdinaryArrays() throws {
        try MLXMetalTestLock.withLock {
            let bank = Bank()
            let owner = SharedOwner(bank)
            XCTAssertTrue(owner.parameters().flattenedValues().isEmpty)
            let counts = owner.modelWeightAccounting()
            XCTAssertEqual(counts.parameterArrayBytes, 0)
            XCTAssertEqual(counts.supplementalMappedBytes, bank.supplementalWeightBytes)
            XCTAssertEqual(counts.logicalWeightBytes, bank.supplementalWeightBytes)
            XCTAssertEqual(owner.numParameters(), bank.supplementalParameterCount)
            XCTAssertEqual(bank.numParameters(), bank.supplementalParameterCount)
            let ordinary = Linear(8, 4, bias: false)
            let ordinaryCounts = ordinary.modelWeightAccounting()
            XCTAssertEqual(ordinaryCounts.parameterArrayBytes, ordinary.weight.nbytes)
            XCTAssertEqual(ordinaryCounts.supplementalMappedBytes, 0)
            XCTAssertEqual(ordinary.numParameters(), 32)
        }
    }

    func testMappedLogicalBytesDoNotBecomeWiredBudget() throws {
        XCTAssertThrowsError(try WiredMemoryUtils.validateAutomaticBudget(
            ModelWeightAccounting(parameterArrayBytes: 1024, supplementalMappedBytes: 96 * 1024 * 1024 * 1024)))
        XCTAssertNoThrow(try WiredMemoryUtils.validateAutomaticBudget(
            ModelWeightAccounting(parameterArrayBytes: 1024, supplementalMappedBytes: 0)))
        let sample = WiredMemoryMeasurement(weightBytes: 1024,
            supplementalMappedWeightBytes: 96 * 1024 * 1024 * 1024,
            kvBytes: 512, workspaceBytes: 256, peakActiveBytes: 1792,
            tokenCount: 8, prefillStepSize: 8)
        XCTAssertEqual(sample.totalBytes, 1792)
        XCTAssertEqual(sample.logicalWeightBytes, 1024 + 96 * 1024 * 1024 * 1024)
    }
    func testStandaloneQuantizedRootUsesRepresentedParameterCount() throws {
        try MLXMetalTestLock.withLock {
            let linear = QuantizedLinear(64, 4, bias: false, groupSize: 32, bits: 4)
            XCTAssertEqual(linear.numParameters(), 256)
            XCTAssertEqual(linear.numParameters(), linear.scales.size * linear.groupSize)
            let embedding = QuantizedEmbedding(embeddingCount: 8, dimensions: 64, groupSize: 32, bits: 4)
            XCTAssertEqual(embedding.numParameters(), 512)
            XCTAssertEqual(embedding.numParameters(), embedding.scales.size * embedding.groupSize)
        }
    }

}
