import Foundation
import CoreML

public struct CoreMLTextEmbedder: TextEmbedder {
    public static let defaultModelName = "intfloat/multilingual-e5-small"
    public static let requiredTokenizerFiles = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]

    public let modelCandidates: [URL]
    public let tokenizerDirectory: URL
    public let maxSequenceLength: Int
    public let outputFeatureName: String
    public let expectedEmbeddingDimension: Int
    private let runtime: CoreMLTextEmbedderRuntime

    public init(
        repositoryRoot: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath),
        maxSequenceLength: Int = 128,
        outputFeatureName: String = "embedding",
        expectedEmbeddingDimension: Int = 384
    ) throws {
        try self.init(
            modelCandidates: [
                repositoryRoot.appendingPathComponent("Models/E5SmallEmbedding.mlpackage"),
                repositoryRoot.appendingPathComponent("Models/E5SmallEmbedding.mlmodelc")
            ],
            tokenizerDirectory: repositoryRoot.appendingPathComponent("Tokenizer"),
            maxSequenceLength: maxSequenceLength,
            outputFeatureName: outputFeatureName,
            expectedEmbeddingDimension: expectedEmbeddingDimension
        )
    }

    public init(
        modelCandidates: [URL],
        tokenizerDirectory: URL,
        maxSequenceLength: Int = 128,
        outputFeatureName: String = "embedding",
        expectedEmbeddingDimension: Int = 384
    ) throws {
        guard maxSequenceLength > 0 else {
            throw EmbeddingError.invalidMaxSequenceLength(maxSequenceLength)
        }

        self.modelCandidates = modelCandidates
        self.tokenizerDirectory = tokenizerDirectory
        self.maxSequenceLength = maxSequenceLength
        self.outputFeatureName = outputFeatureName
        self.expectedEmbeddingDimension = expectedEmbeddingDimension
        self.runtime = CoreMLTextEmbedderRuntime(
            modelCandidates: modelCandidates,
            tokenizerDirectory: tokenizerDirectory,
            maxSequenceLength: maxSequenceLength,
            outputFeatureName: outputFeatureName,
            expectedEmbeddingDimension: expectedEmbeddingDimension
        )
    }

    public func embed(_ text: String, purpose: EmbeddingPurpose) async throws -> [Float] {
        try await runtime.embed(text, purpose: purpose)
    }

    @discardableResult
    public func validateAssets(fileManager: FileManager = .default) throws -> URL {
        try Self.validateAssets(
            modelCandidates: modelCandidates,
            tokenizerDirectory: tokenizerDirectory,
            fileManager: fileManager
        )
    }

    static func validateAssets(
        modelCandidates: [URL],
        tokenizerDirectory: URL,
        fileManager: FileManager = .default
    ) throws -> URL {
        let modelURL = modelCandidates.first { candidate in
            fileManager.fileExists(atPath: candidate.path)
        }

        guard let modelURL else {
            throw EmbeddingError.modelAssetMissing(
                candidates: modelCandidates.map(\.path)
            )
        }

        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: tokenizerDirectory.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            throw EmbeddingError.tokenizerAssetMissing(path: tokenizerDirectory.path)
        }

        for filename in Self.requiredTokenizerFiles {
            let fileURL = tokenizerDirectory.appendingPathComponent(filename)
            guard fileManager.fileExists(atPath: fileURL.path) else {
                throw EmbeddingError.tokenizerFileMissing(path: fileURL.path)
            }
        }

        return modelURL
    }

    static func loadModel(from modelURL: URL) throws -> MLModel {
        do {
            let loadURL: URL
            switch modelURL.pathExtension {
            case "mlmodel", "mlpackage":
                loadURL = try MLModel.compileModel(at: modelURL)
            default:
                loadURL = modelURL
            }

            let configuration = MLModelConfiguration()
            return try MLModel(contentsOf: loadURL, configuration: configuration)
        } catch {
            throw EmbeddingError.coreMLModelLoadFailed(
                path: modelURL.path,
                reason: error.localizedDescription
            )
        }
    }

    static func embeddingVector(
        from outputProvider: MLFeatureProvider,
        outputFeatureName: String,
        expectedDimension: Int
    ) throws -> [Float] {
        let selectedFeatureName: String
        if outputProvider.featureNames.contains(outputFeatureName) {
            selectedFeatureName = outputFeatureName
        } else if outputProvider.featureNames.count == 1, let onlyFeatureName = outputProvider.featureNames.first {
            selectedFeatureName = onlyFeatureName
        } else {
            throw EmbeddingError.coreMLOutputMissing(
                name: outputFeatureName,
                availableOutputs: Array(outputProvider.featureNames)
            )
        }

        guard let multiArray = outputProvider.featureValue(for: selectedFeatureName)?.multiArrayValue else {
            throw EmbeddingError.coreMLOutputIsNotMultiArray(name: selectedFeatureName)
        }

        var embedding: [Float] = []
        embedding.reserveCapacity(multiArray.count)
        for index in 0..<multiArray.count {
            embedding.append(multiArray[index].floatValue)
        }

        guard embedding.count == expectedDimension else {
            throw EmbeddingError.unexpectedEmbeddingDimension(
                expected: expectedDimension,
                actual: embedding.count
            )
        }

        return embedding
    }
}

private actor CoreMLTextEmbedderRuntime {
    private let modelCandidates: [URL]
    private let tokenizerDirectory: URL
    private let maxSequenceLength: Int
    private let outputFeatureName: String
    private let expectedEmbeddingDimension: Int
    private var tokenizer: HuggingFaceTextTokenizer?
    private var model: MLModel?

    init(
        modelCandidates: [URL],
        tokenizerDirectory: URL,
        maxSequenceLength: Int,
        outputFeatureName: String,
        expectedEmbeddingDimension: Int
    ) {
        self.modelCandidates = modelCandidates
        self.tokenizerDirectory = tokenizerDirectory
        self.maxSequenceLength = maxSequenceLength
        self.outputFeatureName = outputFeatureName
        self.expectedEmbeddingDimension = expectedEmbeddingDimension
    }

    func embed(_ text: String, purpose: EmbeddingPurpose) async throws -> [Float] {
        guard !text.isEmpty else {
            throw EmbeddingError.emptyInput
        }

        let model = try cachedModel()
        let tokenizer = try await cachedTokenizer()
        let tokenizedInput = try await tokenizer.tokenize(text, purpose: purpose)
        let inputProvider = try CoreMLTextEmbeddingInputProvider(input: tokenizedInput.coreMLInput)

        let outputProvider: MLFeatureProvider
        do {
            outputProvider = try model.prediction(from: inputProvider)
        } catch {
            throw EmbeddingError.coreMLPredictionFailed(reason: error.localizedDescription)
        }

        return try CoreMLTextEmbedder.embeddingVector(
            from: outputProvider,
            outputFeatureName: outputFeatureName,
            expectedDimension: expectedEmbeddingDimension
        )
    }

    private func cachedTokenizer() async throws -> HuggingFaceTextTokenizer {
        if let tokenizer {
            return tokenizer
        }

        let loadedTokenizer = try await HuggingFaceTextTokenizer.load(
            from: tokenizerDirectory,
            maxSequenceLength: maxSequenceLength
        )
        tokenizer = loadedTokenizer
        return loadedTokenizer
    }

    private func cachedModel() throws -> MLModel {
        if let model {
            return model
        }

        let modelURL = try CoreMLTextEmbedder.validateAssets(
            modelCandidates: modelCandidates,
            tokenizerDirectory: tokenizerDirectory
        )
        let loadedModel = try CoreMLTextEmbedder.loadModel(from: modelURL)
        model = loadedModel
        return loadedModel
    }
}

final class CoreMLTextEmbeddingInputProvider: MLFeatureProvider {
    private let features: [String: MLFeatureValue]

    var featureNames: Set<String> {
        Set(features.keys)
    }

    init(input: CoreMLInput) throws {
        guard input.inputIDs.count == input.attentionMask.count else {
            throw EmbeddingError.coreMLInputLengthMismatch(
                inputIDs: input.inputIDs.count,
                attentionMask: input.attentionMask.count
            )
        }

        let inputIDsArray = try Self.makeMultiArray(values: input.inputIDs)
        let attentionMaskArray = try Self.makeMultiArray(values: input.attentionMask)

        features = [
            "input_ids": MLFeatureValue(multiArray: inputIDsArray),
            "attention_mask": MLFeatureValue(multiArray: attentionMaskArray)
        ]
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        features[featureName]
    }

    private static func makeMultiArray(values: [Int32]) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: values.count)],
            dataType: .int32
        )

        for (index, value) in values.enumerated() {
            array[index] = NSNumber(value: value)
        }

        return array
    }
}
