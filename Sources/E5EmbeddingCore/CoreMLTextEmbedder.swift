import Foundation
import CoreML

public struct CoreMLTextEmbedder: TextEmbedder {
    public static let defaultModelName = "intfloat/multilingual-e5-small"
    public static let requiredTokenizerFiles = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]

    public let assets: CoreMLTextEmbeddingAssets
    public let modelCandidates: [URL]
    public let tokenizerDirectory: URL
    public let tokenizerDirectoryCandidates: [URL]
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
            assets: .repositoryLayout(root: repositoryRoot),
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
        try self.init(
            modelCandidates: modelCandidates,
            tokenizerDirectoryCandidates: [tokenizerDirectory],
            maxSequenceLength: maxSequenceLength,
            outputFeatureName: outputFeatureName,
            expectedEmbeddingDimension: expectedEmbeddingDimension
        )
    }

    public init(
        modelCandidates: [URL],
        tokenizerDirectoryCandidates: [URL],
        maxSequenceLength: Int = 128,
        outputFeatureName: String = "embedding",
        expectedEmbeddingDimension: Int = 384
    ) throws {
        try self.init(
            assets: CoreMLTextEmbeddingAssets(
                modelCandidates: modelCandidates,
                tokenizerDirectoryCandidates: tokenizerDirectoryCandidates
            ),
            maxSequenceLength: maxSequenceLength,
            outputFeatureName: outputFeatureName,
            expectedEmbeddingDimension: expectedEmbeddingDimension
        )
    }

    public init(
        assets: CoreMLTextEmbeddingAssets,
        maxSequenceLength: Int = 128,
        outputFeatureName: String = "embedding",
        expectedEmbeddingDimension: Int = 384
    ) throws {
        guard maxSequenceLength > 0 else {
            throw EmbeddingError.invalidMaxSequenceLength(maxSequenceLength)
        }
        guard !assets.modelCandidates.isEmpty else {
            throw EmbeddingError.modelAssetMissing(candidates: [])
        }
        guard let tokenizerDirectory = assets.tokenizerDirectoryCandidates.first else {
            throw EmbeddingError.tokenizerAssetsMissing(candidates: [])
        }

        self.assets = assets
        self.modelCandidates = assets.modelCandidates
        self.tokenizerDirectory = tokenizerDirectory
        self.tokenizerDirectoryCandidates = assets.tokenizerDirectoryCandidates
        self.maxSequenceLength = maxSequenceLength
        self.outputFeatureName = outputFeatureName
        self.expectedEmbeddingDimension = expectedEmbeddingDimension
        self.runtime = CoreMLTextEmbedderRuntime(
            assets: assets,
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
        try resolvedAssets(fileManager: fileManager).modelURL
    }

    public func resolvedAssets(
        fileManager: FileManager = .default
    ) throws -> CoreMLTextEmbeddingResolvedAssets {
        try assets.resolve(fileManager: fileManager)
    }

    public func assetStatus(
        fileManager: FileManager = .default
    ) -> CoreMLTextEmbeddingAssetStatus {
        assets.status(fileManager: fileManager)
    }

    static func validateAssets(
        modelCandidates: [URL],
        tokenizerDirectory: URL,
        fileManager: FileManager = .default
    ) throws -> URL {
        try CoreMLTextEmbeddingAssets.resolve(
            modelCandidates: modelCandidates,
            tokenizerDirectoryCandidates: [tokenizerDirectory],
            fileManager: fileManager
        ).modelURL
    }

    static func resolveAssets(
        modelCandidates: [URL],
        tokenizerDirectoryCandidates: [URL],
        fileManager: FileManager = .default
    ) throws -> CoreMLTextEmbeddingResolvedAssets {
        try CoreMLTextEmbeddingAssets.resolve(
            modelCandidates: modelCandidates,
            tokenizerDirectoryCandidates: tokenizerDirectoryCandidates,
            fileManager: fileManager
        )
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
    private let assets: CoreMLTextEmbeddingAssets
    private let maxSequenceLength: Int
    private let outputFeatureName: String
    private let expectedEmbeddingDimension: Int
    private var resolvedAssets: CoreMLTextEmbeddingResolvedAssets?
    private var tokenizer: HuggingFaceTextTokenizer?
    private var model: SendableMLModel?

    init(
        assets: CoreMLTextEmbeddingAssets,
        maxSequenceLength: Int,
        outputFeatureName: String,
        expectedEmbeddingDimension: Int
    ) {
        self.assets = assets
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
#if os(macOS)
            if #available(macOS 14.0, *) {
                outputProvider = try await model.prediction(from: inputProvider)
            } else {
                outputProvider = try model.predictionSynchronously(from: inputProvider)
            }
#else
            outputProvider = try await model.prediction(from: inputProvider)
#endif
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

        let resolvedAssets = try cachedAssets()
        let loadedTokenizer = try await HuggingFaceTextTokenizer.load(
            from: resolvedAssets.tokenizerDirectory,
            maxSequenceLength: maxSequenceLength
        )
        tokenizer = loadedTokenizer
        return loadedTokenizer
    }

    private func cachedModel() throws -> SendableMLModel {
        if let model {
            return model
        }

        let resolvedAssets = try cachedAssets()
        let loadedModel = SendableMLModel(
            model: try CoreMLTextEmbedder.loadModel(from: resolvedAssets.modelURL)
        )
        model = loadedModel
        return loadedModel
    }

    private func cachedAssets() throws -> CoreMLTextEmbeddingResolvedAssets {
        if let resolvedAssets {
            return resolvedAssets
        }

        let loadedAssets = try assets.resolve()
        resolvedAssets = loadedAssets
        return loadedAssets
    }
}

private struct SendableMLModel: @unchecked Sendable {
    let model: MLModel
    private let predictionGate = CoreMLPredictionGate()

    @available(macOS 14.0, iOS 17.0, tvOS 17.0, watchOS 10.0, visionOS 1.0, *)
    func prediction(from inputProvider: CoreMLTextEmbeddingInputProvider) async throws -> MLFeatureProvider {
        return try await predictionGate.withLock {
            try await model.prediction(from: inputProvider)
        }
    }

    func predictionSynchronously(from inputProvider: CoreMLTextEmbeddingInputProvider) throws -> MLFeatureProvider {
        return try predictionGate.withLock {
            try model.prediction(from: inputProvider)
        }
    }
}

private final class CoreMLPredictionGate: @unchecked Sendable {
    private let condition = NSCondition()
    private var isAvailable = true
    private var asyncWaiters: [CheckedContinuation<Void, Never>] = []

    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        waitSynchronously()
        defer { signal() }
        return try body()
    }

    func withLock<T>(_ body: () async throws -> T) async rethrows -> T {
        await wait()
        defer { signal() }
        return try await body()
    }

    private func waitSynchronously() {
        condition.lock()
        while !isAvailable {
            condition.wait()
        }
        isAvailable = false
        condition.unlock()
    }

    private func wait() async {
        await withCheckedContinuation { continuation in
            condition.lock()
            if isAvailable {
                isAvailable = false
                condition.unlock()
                continuation.resume()
            } else {
                asyncWaiters.append(continuation)
                condition.unlock()
            }
        }
    }

    private func signal() {
        condition.lock()
        if asyncWaiters.isEmpty {
            isAvailable = true
            condition.signal()
            condition.unlock()
        } else {
            let continuation = asyncWaiters.removeFirst()
            condition.unlock()
            continuation.resume()
        }
    }
}

final class CoreMLTextEmbeddingInputProvider: MLFeatureProvider, @unchecked Sendable {
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
