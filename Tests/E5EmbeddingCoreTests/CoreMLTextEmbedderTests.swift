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
