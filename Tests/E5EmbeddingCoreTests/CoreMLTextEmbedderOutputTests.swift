@testable import E5EmbeddingCore
import CoreML
import XCTest

final class CoreMLTextEmbedderOutputTests: XCTestCase {
    func testBuildsCoreMLInputProvider() throws {
        let input = CoreMLInput(
            inputIDs: [101, 102, 0],
            attentionMask: [1, 1, 0]
        )

        let provider = try CoreMLTextEmbeddingInputProvider(input: input)

        XCTAssertEqual(provider.featureNames, ["input_ids", "attention_mask"])
        XCTAssertEqual(
            provider.featureValue(for: "input_ids")?.multiArrayValue?.count,
            3
        )
        XCTAssertEqual(
            provider.featureValue(for: "attention_mask")?.multiArrayValue?[2].int32Value,
            0
        )
    }

    func testExtractsNamedEmbeddingOutput() throws {
        let provider = try FakeOutputProvider(
            features: ["embedding": MLFeatureValue(multiArray: Self.makeFloatArray([0.1, -0.2, 0.3]))]
        )

        let embedding = try CoreMLTextEmbedder.embeddingVector(
            from: provider,
            outputFeatureName: "embedding",
            expectedDimension: 3
        )

        XCTAssertEqual(embedding, [0.1, -0.2, 0.3])
    }

    func testExtractsFloat16EmbeddingOutput() throws {
        let provider = try FakeOutputProvider(
            features: [
                "embedding": MLFeatureValue(
                    multiArray: Self.makeArray([0.5, -0.25, 0.125], dataType: .float16)
                )
            ]
        )

        let embedding = try CoreMLTextEmbedder.embeddingVector(
            from: provider,
            outputFeatureName: "embedding",
            expectedDimension: 3
        )

        XCTAssertEqual(embedding, [0.5, -0.25, 0.125])
    }

    func testExtractsDoubleEmbeddingOutput() throws {
        let provider = try FakeOutputProvider(
            features: [
                "embedding": MLFeatureValue(
                    multiArray: Self.makeArray([0.25, -0.5], dataType: .double)
                )
            ]
        )

        let embedding = try CoreMLTextEmbedder.embeddingVector(
            from: provider,
            outputFeatureName: "embedding",
            expectedDimension: 2
        )

        XCTAssertEqual(embedding, [0.25, -0.5])
    }

    func testFallsBackToOnlyOutputFeature() throws {
        let provider = try FakeOutputProvider(
            features: ["Identity": MLFeatureValue(multiArray: Self.makeFloatArray([0.4, 0.5]))]
        )

        let embedding = try CoreMLTextEmbedder.embeddingVector(
            from: provider,
            outputFeatureName: "embedding",
            expectedDimension: 2
        )

        XCTAssertEqual(embedding, [0.4, 0.5])
    }

    func testRejectsUnexpectedEmbeddingDimension() throws {
        let provider = try FakeOutputProvider(
            features: ["embedding": MLFeatureValue(multiArray: Self.makeFloatArray([1, 2]))]
        )

        XCTAssertThrowsError(
            try CoreMLTextEmbedder.embeddingVector(
                from: provider,
                outputFeatureName: "embedding",
                expectedDimension: 3
            )
        ) { error in
            XCTAssertEqual(
                error as? EmbeddingError,
                .unexpectedEmbeddingDimension(expected: 3, actual: 2)
            )
        }
    }

    private static func makeFloatArray(_ values: [Float]) throws -> MLMultiArray {
        try makeArray(values.map(Double.init), dataType: .float32)
    }

    private static func makeArray(
        _ values: [Double],
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: values.count)],
            dataType: dataType
        )

        for (index, value) in values.enumerated() {
            array[index] = NSNumber(value: value)
        }

        return array
    }
}

private final class FakeOutputProvider: MLFeatureProvider {
    let features: [String: MLFeatureValue]

    init(features: [String: MLFeatureValue]) {
        self.features = features
    }

    var featureNames: Set<String> {
        Set(features.keys)
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        features[featureName]
    }
}
