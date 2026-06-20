import E5EmbeddingCore
import XCTest

final class EmbeddingResponseTests: XCTestCase {
    func testEncodesJSONOutputStructure() throws {
        let response = EmbeddingResponse(
            model: "intfloat/multilingual-e5-small",
            purpose: .query,
            embedding: [0.1, -0.2]
        )

        let data = try JSONEncoder().encode(response)
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        XCTAssertEqual(object?["model"] as? String, "intfloat/multilingual-e5-small")
        XCTAssertEqual(object?["purpose"] as? String, "query")
        XCTAssertEqual(object?["dimension"] as? Int, 2)
        XCTAssertEqual((object?["embedding"] as? [Double])?.count, 2)
    }
}
