public enum EmbeddingPurpose: String, CaseIterable, Codable, Sendable {
    case query
    case passage

    public init(argument: String) throws {
        guard let purpose = Self(rawValue: argument) else {
            throw EmbeddingError.invalidPurpose(argument)
        }
        self = purpose
    }

    public func applyPrefix(to text: String) -> String {
        switch self {
        case .query:
            return "query: \(text)"
        case .passage:
            return "passage: \(text)"
        }
    }
}
