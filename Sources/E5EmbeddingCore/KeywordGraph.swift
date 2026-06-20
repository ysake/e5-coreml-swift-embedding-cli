public struct KeywordGraphEdge: Codable, Equatable, Sendable {
    public let sourceID: String
    public let sourceText: String
    public let targetID: String
    public let targetText: String
    public let score: Float

    public init(
        sourceID: String,
        sourceText: String,
        targetID: String,
        targetText: String,
        score: Float
    ) {
        self.sourceID = sourceID
        self.sourceText = sourceText
        self.targetID = targetID
        self.targetText = targetText
        self.score = score
    }
}

public enum KeywordGraphBuilder {
    public static func exactTopKEdges(
        records: [StoredEmbedding],
        topK: Int,
        threshold: Float
    ) throws -> [KeywordGraphEdge] {
        guard !records.isEmpty else {
            throw EmbeddingError.emptyEmbeddingRecords
        }
        guard topK > 0 else {
            throw EmbeddingError.invalidTopK(topK)
        }

        var bestByPair: [PairKey: KeywordGraphEdge] = [:]

        for sourceIndex in records.indices {
            let source = records[sourceIndex]
            var candidates: [(targetIndex: Int, score: Float)] = []
            candidates.reserveCapacity(max(0, records.count - 1))

            for targetIndex in records.indices where targetIndex != sourceIndex {
                let target = records[targetIndex]
                let score = try CosineSimilarity.checkedDot(source.embedding, target.embedding)
                guard score >= threshold else {
                    continue
                }
                candidates.append((targetIndex, score))
            }

            candidates.sort { lhs, rhs in
                if lhs.score != rhs.score {
                    return lhs.score > rhs.score
                }
                return records[lhs.targetIndex].id < records[rhs.targetIndex].id
            }

            for candidate in candidates.prefix(topK) {
                let firstIndex = min(sourceIndex, candidate.targetIndex)
                let secondIndex = max(sourceIndex, candidate.targetIndex)
                let key = PairKey(firstIndex: firstIndex, secondIndex: secondIndex)
                let first = records[firstIndex]
                let second = records[secondIndex]
                let edge = KeywordGraphEdge(
                    sourceID: first.id,
                    sourceText: first.text,
                    targetID: second.id,
                    targetText: second.text,
                    score: candidate.score
                )

                if let existing = bestByPair[key], existing.score >= edge.score {
                    continue
                }
                bestByPair[key] = edge
            }
        }

        return bestByPair.values.sorted { lhs, rhs in
            if lhs.score != rhs.score {
                return lhs.score > rhs.score
            }
            if lhs.sourceID != rhs.sourceID {
                return lhs.sourceID < rhs.sourceID
            }
            return lhs.targetID < rhs.targetID
        }
    }
}

private struct PairKey: Hashable {
    let firstIndex: Int
    let secondIndex: Int
}
