import E5EmbeddingCore
import XCTest

final class KeywordGraphBuilderTests: XCTestCase {
    func testBuildsDeduplicatedTopKEdges() throws {
        let records = [
            StoredEmbedding(id: "1", text: "car storage", purpose: .passage, model: "test", embedding: [1, 0]),
            StoredEmbedding(id: "2", text: "roof box", purpose: .passage, model: "test", embedding: [0.9, 0.1]),
            StoredEmbedding(id: "3", text: "quantum mechanics", purpose: .passage, model: "test", embedding: [0, 1])
        ]

        let edges = try KeywordGraphBuilder.exactTopKEdges(
            records: records,
            topK: 1,
            threshold: 0.5
        )

        XCTAssertEqual(edges.count, 1)
        XCTAssertEqual(edges[0].sourceID, "1")
        XCTAssertEqual(edges[0].targetID, "2")
        XCTAssertEqual(edges[0].score, 0.9, accuracy: 0.0001)
    }

    func testRejectsMismatchedEmbeddingDimensions() {
        let records = [
            StoredEmbedding(id: "1", text: "a", purpose: .passage, model: "test", embedding: [1, 0]),
            StoredEmbedding(id: "2", text: "b", purpose: .passage, model: "test", embedding: [1])
        ]

        XCTAssertThrowsError(
            try KeywordGraphBuilder.exactTopKEdges(records: records, topK: 1, threshold: 0)
        ) { error in
            XCTAssertEqual(
                error as? EmbeddingError,
                .vectorLengthMismatch(left: 2, right: 1)
            )
        }
    }

    func testRejectsInvalidTopK() {
        let record = StoredEmbedding(
            id: "1",
            text: "a",
            purpose: .passage,
            model: "test",
            embedding: [1]
        )

        XCTAssertThrowsError(
            try KeywordGraphBuilder.exactTopKEdges(records: [record], topK: 0, threshold: 0)
        ) { error in
            XCTAssertEqual(error as? EmbeddingError, .invalidTopK(0))
        }
    }
}
