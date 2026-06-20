import E5EmbeddingCore
import XCTest

final class CosineSimilarityTests: XCTestCase {
    func testDotProduct() {
        let score = CosineSimilarity.dot([1, 2, 3], [4, 5, 6])
        XCTAssertEqual(score, 32)
    }

    func testCheckedDotRejectsMismatchedDimensions() {
        XCTAssertThrowsError(try CosineSimilarity.checkedDot([1, 2], [1])) { error in
            XCTAssertEqual(
                error as? EmbeddingError,
                .vectorLengthMismatch(left: 2, right: 1)
            )
        }
    }

    func testL2Normalization() {
        let normalized = CosineSimilarity.l2Normalized([3, 4])
        XCTAssertEqual(normalized[0], 0.6, accuracy: 0.0001)
        XCTAssertEqual(normalized[1], 0.8, accuracy: 0.0001)
    }
}
