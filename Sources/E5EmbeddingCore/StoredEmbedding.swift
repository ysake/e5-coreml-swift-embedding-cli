public struct StoredEmbedding: Codable, Equatable, Sendable {
    public let id: String
    public let text: String
    public let purpose: EmbeddingPurpose
    public let model: String
    public let dimension: Int
    public let embedding: [Float]

    public init(
        id: String,
        text: String,
        purpose: EmbeddingPurpose,
        model: String,
        embedding: [Float]
    ) {
        self.id = id
        self.text = text
        self.purpose = purpose
        self.model = model
        self.dimension = embedding.count
        self.embedding = embedding
    }
}
