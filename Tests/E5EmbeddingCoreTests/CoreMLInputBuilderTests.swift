import E5EmbeddingCore
import XCTest

final class CoreMLInputBuilderTests: XCTestCase {
    func testPadsInputIDsAndBuildsAttentionMask() throws {
        let builder = try CoreMLInputBuilder(maxSequenceLength: 5, padTokenID: 0)

        let input = builder.buildInputIDs(from: [101, 200, 102])

        XCTAssertEqual(input.inputIDs, [101, 200, 102, 0, 0])
        XCTAssertEqual(input.attentionMask, [1, 1, 1, 0, 0])
    }

    func testTruncatesInputIDs() throws {
        let builder = try CoreMLInputBuilder(maxSequenceLength: 3, padTokenID: 0)

        let input = builder.buildInputIDs(from: [1, 2, 3, 4])

        XCTAssertEqual(input.inputIDs, [1, 2, 3])
        XCTAssertEqual(input.attentionMask, [1, 1, 1])
    }

    func testRejectsInvalidMaxSequenceLength() {
        XCTAssertThrowsError(try CoreMLInputBuilder(maxSequenceLength: 0)) { error in
            XCTAssertEqual(error as? EmbeddingError, .invalidMaxSequenceLength(0))
        }
    }
}
