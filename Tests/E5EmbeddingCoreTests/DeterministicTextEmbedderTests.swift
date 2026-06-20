import E5EmbeddingCore
import XCTest

final class DeterministicTextEmbedderTests: XCTestCase {
    func testReturnsNormalized384DimensionalVector() async throws {
        let embedder = DeterministicTextEmbedder()

        let embedding = try await embedder.embed("テスト", purpose: .query)

        XCTAssertEqual(embedding.count, 384)
        XCTAssertEqual(CosineSimilarity.l2Norm(embedding), 1, accuracy: 0.0001)
    }

    func testPurposeAffectsEmbedding() async throws {
        let embedder = DeterministicTextEmbedder()

        let query = try await embedder.embed("同じ本文", purpose: .query)
        let passage = try await embedder.embed("同じ本文", purpose: .passage)

        XCTAssertNotEqual(query, passage)
    }
}
