import Foundation

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

    public init(
        repositoryRoot: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath),
        maxSequenceLength: Int = 128
    ) throws {
        try self.init(
            modelCandidates: [
                repositoryRoot.appendingPathComponent("Models/E5SmallEmbedding.mlpackage"),
                repositoryRoot.appendingPathComponent("Models/E5SmallEmbedding.mlmodelc")
            ],
            tokenizerDirectory: repositoryRoot.appendingPathComponent("Tokenizer"),
            maxSequenceLength: maxSequenceLength
        )
    }

    public init(
        modelCandidates: [URL],
        tokenizerDirectory: URL,
        maxSequenceLength: Int = 128
    ) throws {
        guard maxSequenceLength > 0 else {
            throw EmbeddingError.invalidMaxSequenceLength(maxSequenceLength)
        }

        self.modelCandidates = modelCandidates
        self.tokenizerDirectory = tokenizerDirectory
        self.maxSequenceLength = maxSequenceLength
    }

    public func embed(_ text: String, purpose: EmbeddingPurpose) async throws -> [Float] {
        guard !text.isEmpty else {
            throw EmbeddingError.emptyInput
        }

        try validateAssets()

        _ = purpose.applyPrefix(to: text)
        throw EmbeddingError.coreMLIntegrationUnavailable(
            "model and tokenizer assets are present, but tokenizer/Core ML prediction wiring is not implemented in this milestone"
        )
    }

    public func validateAssets(fileManager: FileManager = .default) throws {
        let modelExists = modelCandidates.contains { candidate in
            fileManager.fileExists(atPath: candidate.path)
        }

        guard modelExists else {
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
    }
}
