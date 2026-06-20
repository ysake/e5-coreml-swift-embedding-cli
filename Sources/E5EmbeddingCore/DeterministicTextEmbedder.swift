import Foundation

public struct DeterministicTextEmbedder: TextEmbedder {
    public let dimension: Int

    public init(dimension: Int = 384) {
        self.dimension = dimension
    }

    public func embed(_ text: String, purpose: EmbeddingPurpose) async throws -> [Float] {
        guard !text.isEmpty else {
            throw EmbeddingError.emptyInput
        }

        let prefixedText = purpose.applyPrefix(to: text)
        var state = Self.fnv1a64(prefixedText)
        var values: [Float] = []
        values.reserveCapacity(dimension)

        for index in 0..<dimension {
            state = Self.nextRandom(state &+ UInt64(index))
            let upperBits = UInt32(truncatingIfNeeded: state >> 32)
            let unit = Float(upperBits) / Float(UInt32.max)
            values.append(unit * 2 - 1)
        }

        return CosineSimilarity.l2Normalized(values)
    }

    private static func fnv1a64(_ text: String) -> UInt64 {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x100000001b3
        }
        return hash
    }

    private static func nextRandom(_ state: UInt64) -> UInt64 {
        state &* 6364136223846793005 &+ 1442695040888963407
    }
}
