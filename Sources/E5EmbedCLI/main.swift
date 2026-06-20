import Darwin
import E5EmbeddingCore
import Foundation

@main
struct E5EmbedCommand {
    static func main() async {
        do {
            let options = try CLIOptions.parse(Array(CommandLine.arguments.dropFirst()))
            let embedder: any TextEmbedder = try options.makeEmbedder()
            let embedding = try await embedder.embed(options.text, purpose: options.purpose)
            let response = EmbeddingResponse(
                model: options.modelName,
                purpose: options.purpose,
                embedding: embedding
            )

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

            let data = try encoder.encode(response)
            guard let json = String(data: data, encoding: .utf8) else {
                throw CLIError.outputEncodingFailed
            }
            print(json)
        } catch CLIError.helpRequested {
            print(CLIOptions.usage)
        } catch {
            fputs("e5-embed: \(error.localizedDescription)\n", stderr)
            fputs("\n\(CLIOptions.usage)\n", stderr)
            exit(1)
        }
    }
}

private enum Backend: String {
    case coreML = "coreml"
    case deterministic
}

private struct CLIOptions {
    static let usage = """
    Usage:
      swift run e5-embed [options] <text>

    Options:
      --purpose query|passage              Default: query
      --backend coreml|deterministic       Default: coreml
      --model <path>                       Core ML .mlpackage/.mlmodelc path
      --tokenizer <path>                   Tokenizer assets directory
      --max-length <n>                     Default: 128
      --model-name <name>                  JSON model field
    """

    let purpose: EmbeddingPurpose
    let backend: Backend
    let text: String
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

    static func parse(_ arguments: [String]) throws -> CLIOptions {
        var purpose = EmbeddingPurpose.query
        var backend = Backend.coreML
        var modelPath: String?
        var tokenizerPath: String?
        var maxSequenceLength = 128
        var modelNameOverride: String?
        var textParts: [String] = []
        var index = 0

        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "-h", "--help":
                throw CLIError.helpRequested
            case "--purpose":
                index += 1
                guard index < arguments.count else {
                    throw CLIError.missingValue(option: "--purpose")
                }
                purpose = try EmbeddingPurpose(argument: arguments[index])
            case "--backend":
                index += 1
                guard index < arguments.count else {
                    throw CLIError.missingValue(option: "--backend")
                }
                guard let parsedBackend = Backend(rawValue: arguments[index]) else {
                    throw CLIError.invalidBackend(arguments[index])
                }
                backend = parsedBackend
            case "--model":
                index += 1
                guard index < arguments.count else {
                    throw CLIError.missingValue(option: "--model")
                }
                modelPath = arguments[index]
            case "--tokenizer":
                index += 1
                guard index < arguments.count else {
                    throw CLIError.missingValue(option: "--tokenizer")
                }
                tokenizerPath = arguments[index]
            case "--max-length":
                index += 1
                guard index < arguments.count else {
                    throw CLIError.missingValue(option: "--max-length")
                }
                guard let parsedMaxLength = Int(arguments[index]), parsedMaxLength > 0 else {
                    throw CLIError.invalidMaxLength(arguments[index])
                }
                maxSequenceLength = parsedMaxLength
            case "--model-name":
                index += 1
                guard index < arguments.count else {
                    throw CLIError.missingValue(option: "--model-name")
                }
                modelNameOverride = arguments[index]
            default:
                textParts.append(argument)
            }

            index += 1
        }

        let text = textParts.joined(separator: " ")
        guard !text.isEmpty else {
            throw EmbeddingError.emptyInput
        }

        return CLIOptions(
            purpose: purpose,
            backend: backend,
            text: text,
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

private enum CLIError: Error {
    case helpRequested
    case missingValue(option: String)
    case invalidBackend(String)
    case invalidMaxLength(String)
    case outputEncodingFailed
}

extension CLIError: LocalizedError {
    var errorDescription: String? {
        switch self {
        case .helpRequested:
            return nil
        case .missingValue(let option):
            return "Missing value for \(option)."
        case .invalidBackend(let value):
            return "Invalid backend '\(value)'. Expected 'coreml' or 'deterministic'."
        case .invalidMaxLength(let value):
            return "Invalid max length '\(value)'. Expected a positive integer."
        case .outputEncodingFailed:
            return "Failed to encode JSON output as UTF-8."
        }
    }
}
