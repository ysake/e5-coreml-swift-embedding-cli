import E5EmbeddingCore
import Foundation
import XCTest

final class CoreMLTextEmbedderTests: XCTestCase {
    func testReportsMissingModelAsset() async throws {
        let missingModel = URL(fileURLWithPath: "/tmp/e5-missing-model.mlpackage")
        let missingTokenizer = URL(fileURLWithPath: "/tmp/e5-missing-tokenizer")
        let embedder = try CoreMLTextEmbedder(
            modelCandidates: [missingModel],
            tokenizerDirectory: missingTokenizer
        )

        do {
            _ = try await embedder.embed("test", purpose: .query)
            XCTFail("Expected missing model error.")
        } catch {
            XCTAssertEqual(
                error as? EmbeddingError,
                .modelAssetMissing(candidates: [missingModel.path])
            )
            XCTAssertTrue(error.localizedDescription.contains("Core ML model asset not found"))
        }
    }

    func testReportsMissingTokenizerDirectoryWhenModelExists() throws {
        let sandbox = try makeTemporaryDirectory()
        let model = sandbox.appendingPathComponent("Fake.mlmodelc")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)

        let tokenizer = sandbox.appendingPathComponent("MissingTokenizer")
        let embedder = try CoreMLTextEmbedder(
            modelCandidates: [model],
            tokenizerDirectory: tokenizer
        )

        XCTAssertThrowsError(try embedder.validateAssets()) { error in
            XCTAssertEqual(
                error as? EmbeddingError,
                .tokenizerAssetMissing(path: tokenizer.path)
            )
        }
    }

    func testReportsMissingTokenizerFile() throws {
        let sandbox = try makeTemporaryDirectory()
        let model = sandbox.appendingPathComponent("Fake.mlmodelc")
        let tokenizer = sandbox.appendingPathComponent("Tokenizer")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: tokenizer.appendingPathComponent("tokenizer.json"))
        try Data("{}".utf8).write(to: tokenizer.appendingPathComponent("tokenizer_config.json"))

        let embedder = try CoreMLTextEmbedder(
            modelCandidates: [model],
            tokenizerDirectory: tokenizer
        )
        let missingFile = tokenizer.appendingPathComponent("special_tokens_map.json")

        XCTAssertThrowsError(try embedder.validateAssets()) { error in
            XCTAssertEqual(
                error as? EmbeddingError,
                .tokenizerFileMissing(path: missingFile.path)
            )
        }
    }

    func testValidateAssetsReturnsResolvedModelURL() throws {
        let sandbox = try makeTemporaryDirectory()
        let model = sandbox.appendingPathComponent("Fake.mlmodelc")
        let tokenizer = sandbox.appendingPathComponent("Tokenizer")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)

        for filename in CoreMLTextEmbedder.requiredTokenizerFiles {
            try Data("{}".utf8).write(to: tokenizer.appendingPathComponent(filename))
        }

        let embedder = try CoreMLTextEmbedder(
            modelCandidates: [model],
            tokenizerDirectory: tokenizer
        )

        XCTAssertEqual(try embedder.validateAssets(), model)
    }

    func testResolvesTokenizerDirectoryFromMultipleCandidates() throws {
        let sandbox = try makeTemporaryDirectory()
        let model = sandbox.appendingPathComponent("Fake.mlmodelc")
        let missingTokenizer = sandbox.appendingPathComponent("MissingTokenizer")
        let tokenizer = sandbox.appendingPathComponent("Tokenizer")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)

        for filename in CoreMLTextEmbedder.requiredTokenizerFiles {
            try Data("{}".utf8).write(to: tokenizer.appendingPathComponent(filename))
        }

        let embedder = try CoreMLTextEmbedder(
            modelCandidates: [model],
            tokenizerDirectoryCandidates: [missingTokenizer, tokenizer]
        )

        let resolvedAssets = try embedder.resolvedAssets()

        XCTAssertEqual(resolvedAssets.modelURL, model)
        XCTAssertEqual(resolvedAssets.tokenizerDirectory, tokenizer)
    }

    func testResolvesFlattenedBundleTokenizerAssets() throws {
        let sandbox = try makeTemporaryDirectory()
        let model = sandbox.appendingPathComponent("E5SmallEmbedding.mlmodelc")
        let missingTokenizer = sandbox.appendingPathComponent("Tokenizer")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)

        for filename in CoreMLTextEmbedder.requiredTokenizerFiles {
            try Data("{}".utf8).write(to: sandbox.appendingPathComponent(filename))
        }

        let assets = CoreMLTextEmbeddingAssets(
            modelCandidates: [model],
            tokenizerDirectoryCandidates: [missingTokenizer, sandbox]
        )

        let resolvedAssets = try assets.resolve()

        XCTAssertEqual(resolvedAssets.modelURL, model)
        XCTAssertEqual(resolvedAssets.tokenizerDirectory, sandbox)
    }

    func testAssetStatusReportsReadyAssetsAndModelSize() throws {
        let sandbox = try makeTemporaryDirectory()
        let model = sandbox.appendingPathComponent("Fake.mlmodelc")
        let modelWeights = model.appendingPathComponent("weights.bin")
        let tokenizer = sandbox.appendingPathComponent("Tokenizer")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
        try Data([1, 2, 3, 4]).write(to: modelWeights)

        for filename in CoreMLTextEmbedder.requiredTokenizerFiles {
            try Data("{}".utf8).write(to: tokenizer.appendingPathComponent(filename))
        }

        let assets = CoreMLTextEmbeddingAssets(
            modelCandidates: [model],
            tokenizerDirectory: tokenizer
        )

        let status = assets.status()

        XCTAssertTrue(status.isReady)
        XCTAssertEqual(status.modelURL, model)
        XCTAssertEqual(status.tokenizerDirectory, tokenizer)
        XCTAssertEqual(status.modelSizeInBytes, 4)
        XCTAssertNil(status.errorDescription)
    }

    func testEmbedsWithRealAssetsWhenPresent() async throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let embedder = try CoreMLTextEmbedder(repositoryRoot: root)

        do {
            _ = try embedder.validateAssets()
        } catch {
            throw XCTSkip("Core ML model/tokenizer assets are not present: \(error.localizedDescription)")
        }

        let embedding = try await embedder.embed("テスト", purpose: .query)

        XCTAssertEqual(embedding.count, 384)
        XCTAssertFalse(embedding.contains { $0.isNaN || !$0.isFinite })
        XCTAssertEqual(CosineSimilarity.l2Norm(embedding), 1, accuracy: 0.01)
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("E5EmbeddingCoreTests")
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: directory)
        }
        return directory
    }
}
