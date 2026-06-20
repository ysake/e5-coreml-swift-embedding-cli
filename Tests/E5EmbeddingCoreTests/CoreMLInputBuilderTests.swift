import E5EmbeddingCore
import XCTest

final class CoreMLInputBuilderTests: XCTestCase {
    func testPadsInputIDsAndBuildsAttentionMask() throws {
        let builder = try CoreMLInputBuilder(maxSequenceLength: 5)

        let input = builder.buildInputIDs(from: [101, 200, 102])

        XCTAssertEqual(input.inputIDs, [101, 200, 102, 1, 1])
        XCTAssertEqual(input.attentionMask, [1, 1, 1, 0, 0])
    }

    func testPreservesTerminalSpecialTokenWhenTruncatingInputIDs() throws {
        let builder = try CoreMLInputBuilder(maxSequenceLength: 4)

        let input = builder.buildInputIDs(from: [0, 10, 20, 30, 2])

        XCTAssertEqual(input.inputIDs, [0, 10, 20, 2])
        XCTAssertEqual(input.attentionMask, [1, 1, 1, 1])
    }

    func testCanDisableTerminalTokenPreservationWhenTruncatingInputIDs() throws {
        let builder = try CoreMLInputBuilder(
            maxSequenceLength: 3,
            padTokenID: 0,
            preserveTerminalTokenWhenTruncated: false
        )

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
