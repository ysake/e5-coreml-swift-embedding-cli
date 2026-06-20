public struct CoreMLInput: Equatable, Sendable {
    public let inputIDs: [Int32]
    public let attentionMask: [Int32]

    public init(inputIDs: [Int32], attentionMask: [Int32]) {
        self.inputIDs = inputIDs
        self.attentionMask = attentionMask
    }
}

public struct CoreMLInputBuilder: Sendable {
    public let maxSequenceLength: Int
    public let padTokenID: Int32

    public init(maxSequenceLength: Int = 128, padTokenID: Int32 = 0) throws {
        guard maxSequenceLength > 0 else {
            throw EmbeddingError.invalidMaxSequenceLength(maxSequenceLength)
        }

        self.maxSequenceLength = maxSequenceLength
        self.padTokenID = padTokenID
    }

    public func buildInputIDs(from tokenIDs: [Int32]) -> CoreMLInput {
        let clipped = Array(tokenIDs.prefix(maxSequenceLength))
        let paddingCount = maxSequenceLength - clipped.count
        let inputIDs = clipped + Array(repeating: padTokenID, count: paddingCount)
        let attentionMask = Array(repeating: Int32(1), count: clipped.count)
            + Array(repeating: Int32(0), count: paddingCount)

        return CoreMLInput(inputIDs: inputIDs, attentionMask: attentionMask)
    }
}
