import Darwin
import E5EmbeddingCore
import Foundation

@main
struct E5EmbedSimilarityCommand {
    static func main() async {
        do {
            let options = try SimilarityCLIOptions.parse(Array(CommandLine.arguments.dropFirst()))
            let embedder = try options.makeEmbedder()

            let queryEmbedding = try await embedder.embed(options.query, purpose: .query)
            let passageEmbedding = try await embedder.embed(options.passage, purpose: .passage)
            let score = try CosineSimilarity.checkedDot(queryEmbedding, passageEmbedding)
            let response = SimilarityResponse(
                model: options.modelName,
                query: options.query,
                passage: options.passage,
                queryDimension: queryEmbedding.count,
                passageDimension: passageEmbedding.count,
                score: score
            )

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(response)
            guard let json = String(data: data, encoding: .utf8) else {
                throw SimilarityCLIError.outputEncodingFailed
            }
            print(json)
        } catch SimilarityCLIError.helpRequested {
            print(SimilarityCLIOptions.usage)
        } catch {
            fputs("e5-embed-similarity: \(error.localizedDescription)\n", stderr)
            fputs("\n\(SimilarityCLIOptions.usage)\n", stderr)
            exit(1)
        }
    }
}

private enum SimilarityBackend: String {
    case coreML = "coreml"
    case deterministic
}

private struct SimilarityCLIOptions {
    static let usage = """
    Usage:
      swift run e5-embed-similarity [options] --query <text> --passage <text>

    Options:
      --backend coreml|deterministic       Default: coreml
      --model <path>                       Core ML .mlpackage/.mlmodelc path
      --tokenizer <path>                   Tokenizer assets directory
      --max-length <n>                     Default: 128
      --model-name <name>                  JSON model field
    """

    let backend: SimilarityBackend
    let query: String
    let passage: String
    let modelPath: String?
    let tokenizerPath: String?
    let maxSequenceLength: Int
    let modelNameOverride: String?

    var modelName: String {
        if let modelNameOverride {
            return modelNameOverride
        }

        switch backend {
        case .coreML:
            return CoreMLTextEmbedder.defaultModelName
        case .deterministic:
            return "development/deterministic-e5-placeholder"
        }
    }

    static func parse(_ arguments: [String]) throws -> SimilarityCLIOptions {
        var backend = SimilarityBackend.coreML
        var query: String?
        var passage: String?
        var modelPath: String?
        var tokenizerPath: String?
        var maxSequenceLength = 128
        var modelNameOverride: String?
        var index = 0

        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "-h", "--help":
                throw SimilarityCLIError.helpRequested
            case "--backend":
                index += 1
                guard index < arguments.count else {
                    throw SimilarityCLIError.missingValue(option: "--backend")
                }
                guard let parsedBackend = SimilarityBackend(rawValue: arguments[index]) else {
                    throw SimilarityCLIError.invalidBackend(arguments[index])
                }
                backend = parsedBackend
            case "--query":
                index += 1
                guard index < arguments.count else {
                    throw SimilarityCLIError.missingValue(option: "--query")
                }
                query = arguments[index]
            case "--passage":
                index += 1
                guard index < arguments.count else {
                    throw SimilarityCLIError.missingValue(option: "--passage")
                }
                passage = arguments[index]
            case "--model":
                index += 1
                guard index < arguments.count else {
                    throw SimilarityCLIError.missingValue(option: "--model")
                }
                modelPath = arguments[index]
            case "--tokenizer":
                index += 1
                guard index < arguments.count else {
                    throw SimilarityCLIError.missingValue(option: "--tokenizer")
                }
                tokenizerPath = arguments[index]
            case "--max-length":
                index += 1
                guard index < arguments.count else {
                    throw SimilarityCLIError.missingValue(option: "--max-length")
                }
                guard let parsedMaxLength = Int(arguments[index]), parsedMaxLength > 0 else {
                    throw SimilarityCLIError.invalidMaxLength(arguments[index])
                }
                maxSequenceLength = parsedMaxLength
            case "--model-name":
                index += 1
                guard index < arguments.count else {
                    throw SimilarityCLIError.missingValue(option: "--model-name")
                }
                modelNameOverride = arguments[index]
            default:
                throw SimilarityCLIError.unexpectedArgument(argument)
            }

            index += 1
        }

        guard let query, !query.isEmpty else {
            throw SimilarityCLIError.missingRequiredOption("--query")
        }
        guard let passage, !passage.isEmpty else {
            throw SimilarityCLIError.missingRequiredOption("--passage")
        }

        return SimilarityCLIOptions(
            backend: backend,
            query: query,
            passage: passage,
            modelPath: modelPath,
            tokenizerPath: tokenizerPath,
            maxSequenceLength: maxSequenceLength,
            modelNameOverride: modelNameOverride
        )
    }

    func makeEmbedder() throws -> any TextEmbedder {
        switch backend {
        case .coreML:
            let repositoryRoot = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            let modelCandidates: [URL]
            if let modelPath {
                modelCandidates = [Self.fileURL(from: modelPath, relativeTo: repositoryRoot)]
            } else {
                modelCandidates = [
                    repositoryRoot.appendingPathComponent("Models/E5SmallEmbedding.mlpackage"),
                    repositoryRoot.appendingPathComponent("Models/E5SmallEmbedding.mlmodelc")
                ]
            }

            let tokenizerDirectory = Self.fileURL(
                from: tokenizerPath ?? "Tokenizer",
                relativeTo: repositoryRoot
            )

            return try CoreMLTextEmbedder(
                modelCandidates: modelCandidates,
                tokenizerDirectory: tokenizerDirectory,
                maxSequenceLength: maxSequenceLength
            )
        case .deterministic:
            return DeterministicTextEmbedder()
        }
    }

    private static func fileURL(from path: String, relativeTo root: URL) -> URL {
        if path.hasPrefix("/") {
            return URL(fileURLWithPath: path).standardizedFileURL
        }

        return root.appendingPathComponent(path).standardizedFileURL
    }
}

private struct SimilarityResponse: Codable {
    let model: String
    let query: String
    let passage: String
    let queryDimension: Int
    let passageDimension: Int
    let score: Float
}

private enum SimilarityCLIError: Error {
    case helpRequested
    case missingValue(option: String)
    case missingRequiredOption(String)
    case unexpectedArgument(String)
    case invalidBackend(String)
    case invalidMaxLength(String)
    case outputEncodingFailed
}

extension SimilarityCLIError: LocalizedError {
    var errorDescription: String? {
        switch self {
        case .helpRequested:
            return nil
        case .missingValue(let option):
            return "Missing value for \(option)."
        case .missingRequiredOption(let option):
            return "Missing required option \(option)."
        case .unexpectedArgument(let value):
            return "Unexpected argument '\(value)'."
        case .invalidBackend(let value):
            return "Invalid backend '\(value)'. Expected 'coreml' or 'deterministic'."
        case .invalidMaxLength(let value):
            return "Invalid max length '\(value)'. Expected a positive integer."
        case .outputEncodingFailed:
            return "Failed to encode JSON output as UTF-8."
        }
    }
}
