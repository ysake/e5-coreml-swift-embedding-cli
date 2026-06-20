import Foundation

public struct CosineSimilarity: Sendable {
    public static func dot(_ lhs: [Float], _ rhs: [Float]) -> Float {
        precondition(
            lhs.count == rhs.count,
            "CosineSimilarity.dot requires vectors with equal length."
        )

        return zip(lhs, rhs).reduce(Float.zero) { partial, pair in
            partial + pair.0 * pair.1
        }
    }

    public static func checkedDot(_ lhs: [Float], _ rhs: [Float]) throws -> Float {
        guard lhs.count == rhs.count else {
            throw EmbeddingError.vectorLengthMismatch(left: lhs.count, right: rhs.count)
        }

        return dot(lhs, rhs)
    }

    public static func l2Norm(_ vector: [Float]) -> Float {
        let squaredSum = vector.reduce(Double.zero) { partial, value in
            let doubleValue = Double(value)
            return partial + doubleValue * doubleValue
        }

        return Float(sqrt(squaredSum))
    }

    public static func l2Normalized(_ vector: [Float]) -> [Float] {
        let norm = l2Norm(vector)
        guard norm > 0 else {
            return vector
        }

        return vector.map { $0 / norm }
    }
}
