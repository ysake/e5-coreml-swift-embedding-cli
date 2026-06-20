import E5EmbeddingCore
import XCTest

final class EmbeddingPurposeTests: XCTestCase {
    func testAppliesQueryPrefix() {
        XCTAssertEqual(
            EmbeddingPurpose.query.applyPrefix(to: "テスト"),
            "query: テスト"
        )
    }

    func testAppliesPassagePrefix() {
        XCTAssertEqual(
            EmbeddingPurpose.passage.applyPrefix(to: "車内収納を増やす"),
            "passage: 車内収納を増やす"
        )
    }

    func testParsesPurposeArgument() throws {
        XCTAssertEqual(try EmbeddingPurpose(argument: "query"), .query)
        XCTAssertEqual(try EmbeddingPurpose(argument: "passage"), .passage)
    }

    func testRejectsInvalidPurposeArgument() {
        XCTAssertThrowsError(try EmbeddingPurpose(argument: "document")) { error in
            XCTAssertEqual(error as? EmbeddingError, .invalidPurpose("document"))
        }
    }
}
