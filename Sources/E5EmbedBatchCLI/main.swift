import Darwin
import E5EmbeddingCore
import Foundation

@main
struct E5EmbedBatchCommand {
    static func main() async {
        do {
            let options = try BatchOptions.parse(Array(CommandLine.arguments.dropFirst()))
            let embedder = try options.makeEmbedder()
            let keywords = try options.loadKeywords()
            guard !keywords.isEmpty else {
                throw BatchError.noInputKeywords
            }

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]

            var outputLines: [String] = []
            outputLines.reserveCapacity(keywords.count)

            for keyword in keywords {
                let embedding = try await embedder.embed(keyword.text, purpose: options.purpose)
                let record = StoredEmbedding(
                    id: keyword.id,
                    text: keyword.text,
                    purpose: options.purpose,
                    model: options.modelName,
                    embedding: embedding
                )
                let data = try encoder.encode(record)
                guard let line = String(data: data, encoding: .utf8) else {
                    throw BatchError.outputEncodingFailed
                }
                outputLines.append(line)
            }

            try options.writeOutput(outputLines.joined(separator: "\n") + "\n")
        } catch BatchError.helpRequested {
            print(BatchOptions.usage)
        } catch {
            fputs("e5-embed-batch: \(error.localizedDescription)\n", stderr)
            fputs("\n\(BatchOptions.usage)\n", stderr)
            exit(1)
        }
    }
}

private enum BatchBackend: String {
    case coreML = "coreml"
    case deterministic
}

private struct BatchKeyword {
    let id: String
    let text: String
}

private struct BatchOptions {
    static let usage = """
    Usage:
      swift run e5-embed-batch [options] --input <keywords.txt> --output <embeddings.jsonl>

    Options:
      --input <path>                      One keyword per line. Use '-' for stdin.
      --output <path>                     JSONL output path. Use '-' for stdout.
      --purpose query|passage             Default: passage
      --backend coreml|deterministic      Default: coreml
      --model <path>                      Core ML .mlpackage/.mlmodelc path
      --tokenizer <path>                  Tokenizer assets directory
      --max-length <n>                    Default: 128
      --model-name <name>                 JSON model field
    """

    let inputPath: String
    let outputPath: String
    let purpose: EmbeddingPurpose
    let backend: BatchBackend
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

    static func parse(_ arguments: [String]) throws -> BatchOptions {
        var inputPath: String?
        var outputPath: String?
        var purpose = EmbeddingPurpose.passage
        var backend = BatchBackend.coreML
        var modelPath: String?
        var tokenizerPath: String?
        var maxSequenceLength = 128
        var modelNameOverride: String?
        var index = 0

        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "-h", "--help":
                throw BatchError.helpRequested
            case "--input":
                index += 1
                guard index < arguments.count else {
                    throw BatchError.missingValue(option: "--input")
                }
                inputPath = arguments[index]
            case "--output":
                index += 1
                guard index < arguments.count else {
                    throw BatchError.missingValue(option: "--output")
                }
                outputPath = arguments[index]
            case "--purpose":
                index += 1
                guard index < arguments.count else {
                    throw BatchError.missingValue(option: "--purpose")
                }
                purpose = try EmbeddingPurpose(argument: arguments[index])
            case "--backend":
                index += 1
                guard index < arguments.count else {
                    throw BatchError.missingValue(option: "--backend")
                }
                guard let parsedBackend = BatchBackend(rawValue: arguments[index]) else {
                    throw BatchError.invalidBackend(arguments[index])
                }
                backend = parsedBackend
            case "--model":
                index += 1
                guard index < arguments.count else {
                    throw BatchError.missingValue(option: "--model")
                }
                modelPath = arguments[index]
            case "--tokenizer":
                index += 1
                guard index < arguments.count else {
                    throw BatchError.missingValue(option: "--tokenizer")
                }
                tokenizerPath = arguments[index]
            case "--max-length":
                index += 1
                guard index < arguments.count else {
                    throw BatchError.missingValue(option: "--max-length")
                }
                guard let parsedMaxLength = Int(arguments[index]), parsedMaxLength > 0 else {
                    throw BatchError.invalidMaxLength(arguments[index])
                }
                maxSequenceLength = parsedMaxLength
            case "--model-name":
                index += 1
                guard index < arguments.count else {
                    throw BatchError.missingValue(option: "--model-name")
                }
                modelNameOverride = arguments[index]
            default:
                throw BatchError.unexpectedArgument(argument)
            }

            index += 1
        }

        guard let inputPath else {
            throw BatchError.missingRequiredOption("--input")
        }
        guard let outputPath else {
            throw BatchError.missingRequiredOption("--output")
        }

        return BatchOptions(
            inputPath: inputPath,
            outputPath: outputPath,
            purpose: purpose,
            backend: backend,
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

    func loadKeywords() throws -> [BatchKeyword] {
        let content: String
        if inputPath == "-" {
            let data = FileHandle.standardInput.readDataToEndOfFile()
            guard let standardInput = String(data: data, encoding: .utf8) else {
                throw BatchError.inputEncodingFailed
            }
            content = standardInput
        } else {
            content = try String(contentsOf: Self.fileURL(from: inputPath), encoding: .utf8)
        }

        return content.components(separatedBy: .newlines).enumerated().compactMap { offset, line in
            let text = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else {
                return nil
            }
            return BatchKeyword(id: String(offset + 1), text: text)
        }
    }

    func writeOutput(_ output: String) throws {
        if outputPath == "-" {
            print(output, terminator: "")
            return
        }

        let outputURL = Self.fileURL(from: outputPath)
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try output.write(to: outputURL, atomically: true, encoding: .utf8)
    }

    private static func fileURL(from path: String) -> URL {
        fileURL(from: path, relativeTo: URL(fileURLWithPath: FileManager.default.currentDirectoryPath))
    }

    private static func fileURL(from path: String, relativeTo root: URL) -> URL {
        if path.hasPrefix("/") {
            return URL(fileURLWithPath: path).standardizedFileURL
        }

        return root.appendingPathComponent(path).standardizedFileURL
    }
}

private enum BatchError: Error {
    case helpRequested
    case missingValue(option: String)
    case missingRequiredOption(String)
    case unexpectedArgument(String)
    case invalidBackend(String)
    case invalidMaxLength(String)
    case inputEncodingFailed
    case outputEncodingFailed
    case noInputKeywords
}

extension BatchError: LocalizedError {
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
        case .inputEncodingFailed:
            return "Failed to read input as UTF-8."
        case .outputEncodingFailed:
            return "Failed to encode JSONL output as UTF-8."
        case .noInputKeywords:
            return "Input must contain at least one non-empty keyword."
        }
    }
}
