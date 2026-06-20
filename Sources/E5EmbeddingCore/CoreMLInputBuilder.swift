public struct CoreMLInput: Equatable, Sendable {
    public let inputIDs: [Int32]
    public let attentionMask: [Int32]

    public init(inputIDs: [Int32], attentionMask: [Int32]) {
        self.inputIDs = inputIDs
        self.attentionMask = attentionMask
    }
}

public struct CoreMLInputBuilder: Sendable {
    public static let e5PadTokenID: Int32 = 1

    public let maxSequenceLength: Int
    public let padTokenID: Int32
    public let preserveTerminalTokenWhenTruncated: Bool

    public init(
        maxSequenceLength: Int = 128,
        padTokenID: Int32 = Self.e5PadTokenID,
        preserveTerminalTokenWhenTruncated: Bool = true
    ) throws {
        guard maxSequenceLength > 0 else {
            throw EmbeddingError.invalidMaxSequenceLength(maxSequenceLength)
        }

        self.maxSequenceLength = maxSequenceLength
        self.padTokenID = padTokenID
        self.preserveTerminalTokenWhenTruncated = preserveTerminalTokenWhenTruncated
    }

    public func buildInputIDs(from tokenIDs: [Int32]) -> CoreMLInput {
        let clipped = clippedTokenIDs(from: tokenIDs)
        let paddingCount = maxSequenceLength - clipped.count
        let inputIDs = clipped + Array(repeating: padTokenID, count: paddingCount)
        let attentionMask = Array(repeating: Int32(1), count: clipped.count)
            + Array(repeating: Int32(0), count: paddingCount)

        return CoreMLInput(inputIDs: inputIDs, attentionMask: attentionMask)
    }

    private func clippedTokenIDs(from tokenIDs: [Int32]) -> [Int32] {
        guard tokenIDs.count > maxSequenceLength else {
            return tokenIDs
        }

        guard preserveTerminalTokenWhenTruncated, let terminalTokenID = tokenIDs.last else {
            return Array(tokenIDs.prefix(maxSequenceLength))
        }

        if maxSequenceLength == 1 {
            return [terminalTokenID]
        }

        return Array(tokenIDs.prefix(maxSequenceLength - 1)) + [terminalTokenID]
    }
}
