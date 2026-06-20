public protocol TextEmbedder: Sendable {
    func embed(_ text: String, purpose: EmbeddingPurpose) async throws -> [Float]
}
