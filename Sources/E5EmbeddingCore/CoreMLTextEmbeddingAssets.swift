import Foundation

public struct CoreMLTextEmbeddingAssets: Equatable, Sendable {
    public let modelCandidates: [URL]
    public let tokenizerDirectoryCandidates: [URL]

    public init(modelCandidates: [URL], tokenizerDirectoryCandidates: [URL]) {
        self.modelCandidates = Self.unique(modelCandidates)
        self.tokenizerDirectoryCandidates = Self.unique(tokenizerDirectoryCandidates)
    }

    public init(modelCandidates: [URL], tokenizerDirectory: URL) {
        self.init(
            modelCandidates: modelCandidates,
            tokenizerDirectoryCandidates: [tokenizerDirectory]
        )
    }

    public static func repositoryLayout(
        root: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    ) -> CoreMLTextEmbeddingAssets {
        CoreMLTextEmbeddingAssets(
            modelCandidates: [
                root.appendingPathComponent("Models/E5SmallEmbedding.mlpackage"),
                root.appendingPathComponent("Models/E5SmallEmbedding.mlmodelc")
            ],
            tokenizerDirectory: root.appendingPathComponent("Tokenizer")
        )
    }

    public static func appBundle(
        _ bundle: Bundle = .main,
        modelResourceName: String = "E5SmallEmbedding",
        tokenizerDirectoryName: String = "Tokenizer",
        includeFlattenedTokenizerRoot: Bool = true
    ) -> CoreMLTextEmbeddingAssets {
        var modelCandidates: [URL] = []
        for pathExtension in ["mlmodelc", "mlpackage"] {
            if let resourceURL = bundle.url(
                forResource: modelResourceName,
                withExtension: pathExtension
            ) {
                modelCandidates.append(resourceURL)
            }
        }

        var tokenizerDirectoryCandidates: [URL] = []
        if let tokenizerURL = bundle.url(forResource: tokenizerDirectoryName, withExtension: nil) {
            tokenizerDirectoryCandidates.append(tokenizerURL)
        }

        if let resourceURL = bundle.resourceURL {
            modelCandidates.append(
                resourceURL.appendingPathComponent("\(modelResourceName).mlmodelc")
            )
            modelCandidates.append(
                resourceURL.appendingPathComponent("\(modelResourceName).mlpackage")
            )
            tokenizerDirectoryCandidates.append(
                resourceURL.appendingPathComponent(tokenizerDirectoryName, isDirectory: true)
            )

            if includeFlattenedTokenizerRoot {
                tokenizerDirectoryCandidates.append(resourceURL)
            }
        }

        return CoreMLTextEmbeddingAssets(
            modelCandidates: modelCandidates,
            tokenizerDirectoryCandidates: tokenizerDirectoryCandidates
        )
    }

    public func resolve(
        fileManager: FileManager = .default
    ) throws -> CoreMLTextEmbeddingResolvedAssets {
        try CoreMLTextEmbeddingAssets.resolve(
            modelCandidates: modelCandidates,
            tokenizerDirectoryCandidates: tokenizerDirectoryCandidates,
            fileManager: fileManager
        )
    }

    public func status(
        fileManager: FileManager = .default
    ) -> CoreMLTextEmbeddingAssetStatus {
        do {
            let resolvedAssets = try resolve(fileManager: fileManager)
            return CoreMLTextEmbeddingAssetStatus(
                isReady: true,
                modelURL: resolvedAssets.modelURL,
                tokenizerDirectory: resolvedAssets.tokenizerDirectory,
                modelSizeInBytes: Self.fileSizeInBytes(
                    at: resolvedAssets.modelURL,
                    fileManager: fileManager
                ),
                errorDescription: nil
            )
        } catch {
            return CoreMLTextEmbeddingAssetStatus(
                isReady: false,
                modelURL: firstExistingURL(in: modelCandidates, fileManager: fileManager),
                tokenizerDirectory: firstUsableTokenizerDirectory(fileManager: fileManager),
                modelSizeInBytes: nil,
                errorDescription: error.localizedDescription
            )
        }
    }

    static func resolve(
        modelCandidates: [URL],
        tokenizerDirectoryCandidates: [URL],
        fileManager: FileManager = .default
    ) throws -> CoreMLTextEmbeddingResolvedAssets {
        guard let modelURL = firstExistingURL(in: modelCandidates, fileManager: fileManager) else {
            throw EmbeddingError.modelAssetMissing(candidates: modelCandidates.map(\.path))
        }

        guard let tokenizerDirectory = firstUsableTokenizerDirectory(
            in: tokenizerDirectoryCandidates,
            fileManager: fileManager
        ) else {
            try throwTokenizerError(
                tokenizerDirectoryCandidates: tokenizerDirectoryCandidates,
                fileManager: fileManager
            )
        }

        return CoreMLTextEmbeddingResolvedAssets(
            modelURL: modelURL,
            tokenizerDirectory: tokenizerDirectory
        )
    }

    private static func firstExistingURL(
        in candidates: [URL],
        fileManager: FileManager
    ) -> URL? {
        candidates.first { fileManager.fileExists(atPath: $0.path) }
    }

    private func firstExistingURL(
        in candidates: [URL],
        fileManager: FileManager
    ) -> URL? {
        Self.firstExistingURL(in: candidates, fileManager: fileManager)
    }

    private static func firstUsableTokenizerDirectory(
        in candidates: [URL],
        fileManager: FileManager
    ) -> URL? {
        candidates.first { candidate in
            guard isDirectory(candidate, fileManager: fileManager) else {
                return false
            }

            return CoreMLTextEmbedder.requiredTokenizerFiles.allSatisfy { filename in
                fileManager.fileExists(
                    atPath: candidate.appendingPathComponent(filename).path
                )
            }
        }
    }

    private func firstUsableTokenizerDirectory(fileManager: FileManager) -> URL? {
        Self.firstUsableTokenizerDirectory(
            in: tokenizerDirectoryCandidates,
            fileManager: fileManager
        )
    }

    private static func throwTokenizerError(
        tokenizerDirectoryCandidates: [URL],
        fileManager: FileManager
    ) throws -> Never {
        if tokenizerDirectoryCandidates.count == 1, let onlyCandidate = tokenizerDirectoryCandidates.first {
            try validateTokenizerDirectory(onlyCandidate, fileManager: fileManager)
        }

        if let existingDirectory = tokenizerDirectoryCandidates.first(where: {
            isDirectory($0, fileManager: fileManager)
        }) {
            for filename in CoreMLTextEmbedder.requiredTokenizerFiles {
                let fileURL = existingDirectory.appendingPathComponent(filename)
                if !fileManager.fileExists(atPath: fileURL.path) {
                    throw EmbeddingError.tokenizerFileMissing(path: fileURL.path)
                }
            }
        }

        throw EmbeddingError.tokenizerAssetsMissing(
            candidates: tokenizerDirectoryCandidates.map(\.path)
        )
    }

    private static func validateTokenizerDirectory(
        _ tokenizerDirectory: URL,
        fileManager: FileManager
    ) throws {
        guard isDirectory(tokenizerDirectory, fileManager: fileManager) else {
            throw EmbeddingError.tokenizerAssetMissing(path: tokenizerDirectory.path)
        }

        for filename in CoreMLTextEmbedder.requiredTokenizerFiles {
            let fileURL = tokenizerDirectory.appendingPathComponent(filename)
            guard fileManager.fileExists(atPath: fileURL.path) else {
                throw EmbeddingError.tokenizerFileMissing(path: fileURL.path)
            }
        }
    }

    private static func isDirectory(_ url: URL, fileManager: FileManager) -> Bool {
        var isDirectory: ObjCBool = false
        return fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
            && isDirectory.boolValue
    }

    private static func fileSizeInBytes(at url: URL, fileManager: FileManager) -> Int64? {
        guard fileManager.fileExists(atPath: url.path) else {
            return nil
        }

        if isDirectory(url, fileManager: fileManager) {
            guard let enumerator = fileManager.enumerator(
                at: url,
                includingPropertiesForKeys: [.fileSizeKey],
                options: [.skipsHiddenFiles]
            ) else {
                return nil
            }

            var total: Int64 = 0
            for case let fileURL as URL in enumerator {
                guard let values = try? fileURL.resourceValues(forKeys: [.fileSizeKey]),
                      let fileSize = values.fileSize
                else {
                    continue
                }
                total += Int64(fileSize)
            }
            return total
        }

        guard let attributes = try? fileManager.attributesOfItem(atPath: url.path),
              let size = attributes[.size] as? NSNumber
        else {
            return nil
        }
        return size.int64Value
    }

    private static func unique(_ urls: [URL]) -> [URL] {
        var seenPaths: Set<String> = []
        var uniqueURLs: [URL] = []

        for url in urls {
            let path = url.standardizedFileURL.path
            if seenPaths.insert(path).inserted {
                uniqueURLs.append(url)
            }
        }

        return uniqueURLs
    }
}

public struct CoreMLTextEmbeddingResolvedAssets: Equatable, Sendable {
    public let modelURL: URL
    public let tokenizerDirectory: URL

    public init(modelURL: URL, tokenizerDirectory: URL) {
        self.modelURL = modelURL
        self.tokenizerDirectory = tokenizerDirectory
    }
}

public struct CoreMLTextEmbeddingAssetStatus: Equatable, Sendable {
    public let isReady: Bool
    public let modelURL: URL?
    public let tokenizerDirectory: URL?
    public let modelSizeInBytes: Int64?
    public let errorDescription: String?

    public init(
        isReady: Bool,
        modelURL: URL?,
        tokenizerDirectory: URL?,
        modelSizeInBytes: Int64?,
        errorDescription: String?
    ) {
        self.isReady = isReady
        self.modelURL = modelURL
        self.tokenizerDirectory = tokenizerDirectory
        self.modelSizeInBytes = modelSizeInBytes
        self.errorDescription = errorDescription
    }
}
