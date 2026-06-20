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
      swift run e5-embed [--purpose query|passage] [--backend coreml|deterministic] <text>

    Defaults:
      --purpose query
      --backend coreml
    """

    let purpose: EmbeddingPurpose
    let backend: Backend
    let text: String

    var modelName: String {
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
            default:
                textParts.append(argument)
            }

            index += 1
        }

        let text = textParts.joined(separator: " ")
        guard !text.isEmpty else {
            throw EmbeddingError.emptyInput
        }

        return CLIOptions(purpose: purpose, backend: backend, text: text)
    }

    func makeEmbedder() throws -> any TextEmbedder {
        switch backend {
        case .coreML:
            return try CoreMLTextEmbedder()
        case .deterministic:
            return DeterministicTextEmbedder()
        }
    }
}

private enum CLIError: Error {
    case helpRequested
    case missingValue(option: String)
    case invalidBackend(String)
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
        case .outputEncodingFailed:
            return "Failed to encode JSON output as UTF-8."
        }
    }
}
