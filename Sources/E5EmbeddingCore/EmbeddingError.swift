import Foundation

public enum EmbeddingError: Error, Equatable, Sendable {
    case emptyInput
    case invalidPurpose(String)
    case invalidMaxSequenceLength(Int)
    case vectorLengthMismatch(left: Int, right: Int)
    case modelAssetMissing(candidates: [String])
    case tokenizerAssetMissing(path: String)
    case tokenizerFileMissing(path: String)
    case coreMLIntegrationUnavailable(String)
}

extension EmbeddingError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .emptyInput:
            return "Input text must not be empty."
        case .invalidPurpose(let value):
            return "Invalid purpose '\(value)'. Expected 'query' or 'passage'."
        case .invalidMaxSequenceLength(let value):
            return "Max sequence length must be greater than zero, got \(value)."
        case .vectorLengthMismatch(let left, let right):
            return "Vector lengths must match, got \(left) and \(right)."
        case .modelAssetMissing(let candidates):
            let joined = candidates.joined(separator: ", ")
            return "Core ML model asset not found. Checked: \(joined)."
        case .tokenizerAssetMissing(let path):
            return "Tokenizer directory not found at \(path)."
        case .tokenizerFileMissing(let path):
            return "Tokenizer file not found at \(path)."
        case .coreMLIntegrationUnavailable(let reason):
            return "Core ML embedding integration is not available yet: \(reason)"
        }
    }
}
