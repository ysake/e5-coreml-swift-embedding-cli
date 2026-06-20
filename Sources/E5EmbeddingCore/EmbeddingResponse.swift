public struct EmbeddingResponse: Codable, Equatable, Sendable {
    public let model: String
    public let purpose: EmbeddingPurpose
    public let dimension: Int
    public let embedding: [Float]

    public init(model: String, purpose: EmbeddingPurpose, embedding: [Float]) {
        self.model = model
        self.purpose = purpose
        self.dimension = embedding.count
        self.embedding = embedding
    }
}
