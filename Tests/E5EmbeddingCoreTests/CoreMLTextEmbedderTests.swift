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
}
