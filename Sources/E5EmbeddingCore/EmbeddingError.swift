import Foundation

public enum EmbeddingError: Error, Equatable, Sendable {
    case emptyInput
    case invalidPurpose(String)
    case invalidMaxSequenceLength(Int)
    case vectorLengthMismatch(left: Int, right: Int)
    case modelAssetMissing(candidates: [String])
    case tokenizerAssetMissing(path: String)
    case tokenizerAssetsMissing(candidates: [String])
    case tokenizerFileMissing(path: String)
    case tokenIDOutOfInt32Range(Int)
    case coreMLInputLengthMismatch(inputIDs: Int, attentionMask: Int)
    case coreMLModelLoadFailed(path: String, reason: String)
    case coreMLPredictionFailed(reason: String)
    case coreMLOutputMissing(name: String, availableOutputs: [String])
    case coreMLOutputIsNotMultiArray(name: String)
    case unexpectedEmbeddingDimension(expected: Int, actual: Int)
    case emptyEmbeddingRecords
    case invalidTopK(Int)
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
        case .tokenizerAssetsMissing(let candidates):
            let joined = candidates.joined(separator: ", ")
            return "Tokenizer assets not found. Checked: \(joined)."
        case .tokenizerFileMissing(let path):
            return "Tokenizer file not found at \(path)."
        case .tokenIDOutOfInt32Range(let tokenID):
            return "Tokenizer produced token ID \(tokenID), which does not fit in Int32."
        case .coreMLInputLengthMismatch(let inputIDs, let attentionMask):
            return "Core ML input_ids and attention_mask must have equal length, got \(inputIDs) and \(attentionMask)."
        case .coreMLModelLoadFailed(let path, let reason):
            return "Failed to load Core ML model at \(path): \(reason)"
        case .coreMLPredictionFailed(let reason):
            return "Core ML prediction failed: \(reason)"
        case .coreMLOutputMissing(let name, let availableOutputs):
            let joinedOutputs = availableOutputs.sorted().joined(separator: ", ")
            return "Core ML output '\(name)' not found. Available outputs: \(joinedOutputs)."
        case .coreMLOutputIsNotMultiArray(let name):
            return "Core ML output '\(name)' is not an MLMultiArray."
        case .unexpectedEmbeddingDimension(let expected, let actual):
            return "Expected embedding dimension \(expected), got \(actual)."
        case .emptyEmbeddingRecords:
            return "At least one embedding record is required."
        case .invalidTopK(let value):
            return "top-k must be greater than zero, got \(value)."
        }
    }
}
