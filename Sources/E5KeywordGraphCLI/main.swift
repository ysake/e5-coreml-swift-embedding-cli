import Darwin
import E5EmbeddingCore
import Foundation

@main
struct E5KeywordGraphCommand {
    static func main() {
        do {
            let options = try GraphOptions.parse(Array(CommandLine.arguments.dropFirst()))
            let records = try options.loadRecords()
            let edges = try KeywordGraphBuilder.exactTopKEdges(
                records: records,
                topK: options.topK,
                threshold: options.threshold
            )
            let output = try options.render(records: records, edges: edges)
            try options.writeOutput(output)
        } catch GraphError.helpRequested {
            print(GraphOptions.usage)
        } catch {
            fputs("e5-keyword-graph: \(error.localizedDescription)\n", stderr)
            fputs("\n\(GraphOptions.usage)\n", stderr)
            exit(1)
        }
    }
}

private enum GraphFormat: String {
    case csv
    case graphml
    case json
}

private struct GraphOptions {
    static let usage = """
    Usage:
      swift run e5-keyword-graph [options] --input <embeddings.jsonl> --output <graph-file>

    Options:
      --input <path>                      JSONL from e5-embed-batch. Use '-' for stdin.
      --output <path>                     Output file path. Use '-' for stdout.
      --format csv|graphml|json           Default: csv
      --top-k <n>                         Default: 10
      --threshold <score>                 Default: 0.0
    """

    let inputPath: String
    let outputPath: String
    let format: GraphFormat
    let topK: Int
    let threshold: Float

    static func parse(_ arguments: [String]) throws -> GraphOptions {
        var inputPath: String?
        var outputPath: String?
        var format = GraphFormat.csv
        var topK = 10
        var threshold = Float.zero
        var index = 0

        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "-h", "--help":
                throw GraphError.helpRequested
            case "--input":
                index += 1
                guard index < arguments.count else {
                    throw GraphError.missingValue(option: "--input")
                }
                inputPath = arguments[index]
            case "--output":
                index += 1
                guard index < arguments.count else {
                    throw GraphError.missingValue(option: "--output")
                }
                outputPath = arguments[index]
            case "--format":
                index += 1
                guard index < arguments.count else {
                    throw GraphError.missingValue(option: "--format")
                }
                guard let parsedFormat = GraphFormat(rawValue: arguments[index]) else {
                    throw GraphError.invalidFormat(arguments[index])
                }
                format = parsedFormat
            case "--top-k":
                index += 1
                guard index < arguments.count else {
                    throw GraphError.missingValue(option: "--top-k")
                }
                guard let parsedTopK = Int(arguments[index]), parsedTopK > 0 else {
                    throw GraphError.invalidTopK(arguments[index])
                }
                topK = parsedTopK
            case "--threshold":
                index += 1
                guard index < arguments.count else {
                    throw GraphError.missingValue(option: "--threshold")
                }
                guard let parsedThreshold = Float(arguments[index]) else {
                    throw GraphError.invalidThreshold(arguments[index])
                }
                threshold = parsedThreshold
            default:
                throw GraphError.unexpectedArgument(argument)
            }

            index += 1
        }

        guard let inputPath else {
            throw GraphError.missingRequiredOption("--input")
        }
        guard let outputPath else {
            throw GraphError.missingRequiredOption("--output")
        }

        return GraphOptions(
            inputPath: inputPath,
            outputPath: outputPath,
            format: format,
            topK: topK,
            threshold: threshold
        )
    }

    func loadRecords() throws -> [StoredEmbedding] {
        let content: String
        if inputPath == "-" {
            let data = FileHandle.standardInput.readDataToEndOfFile()
            guard let standardInput = String(data: data, encoding: .utf8) else {
                throw GraphError.inputEncodingFailed
            }
            content = standardInput
        } else {
            content = try String(contentsOf: Self.fileURL(from: inputPath), encoding: .utf8)
        }

        let decoder = JSONDecoder()
        var records: [StoredEmbedding] = []

        for (offset, line) in content.components(separatedBy: .newlines).enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else {
                continue
            }

            guard let data = trimmed.data(using: .utf8) else {
                throw GraphError.inputEncodingFailed
            }

            do {
                records.append(try decoder.decode(StoredEmbedding.self, from: data))
            } catch {
                throw GraphError.invalidJSONLLine(line: offset + 1, reason: error.localizedDescription)
            }
        }

        return records
    }

    func render(records: [StoredEmbedding], edges: [KeywordGraphEdge]) throws -> String {
        switch format {
        case .csv:
            return renderCSV(edges: edges)
        case .graphml:
            return renderGraphML(records: records, edges: edges)
        case .json:
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let nodes = records.map { record in
                GraphJSONNode(
                    id: record.id,
                    text: record.text,
                    purpose: record.purpose,
                    model: record.model,
                    dimension: record.dimension
                )
            }
            let data = try encoder.encode(GraphJSONOutput(nodes: nodes, edges: edges))
            guard let output = String(data: data, encoding: .utf8) else {
                throw GraphError.outputEncodingFailed
            }
            return output + "\n"
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

    private func renderCSV(edges: [KeywordGraphEdge]) -> String {
        var lines = ["source_id,source_text,target_id,target_text,score"]
        lines.reserveCapacity(edges.count + 1)

        for edge in edges {
            lines.append([
                csv(edge.sourceID),
                csv(edge.sourceText),
                csv(edge.targetID),
                csv(edge.targetText),
                String(edge.score)
            ].joined(separator: ","))
        }

        return lines.joined(separator: "\n") + "\n"
    }

    private func renderGraphML(records: [StoredEmbedding], edges: [KeywordGraphEdge]) -> String {
        var lines: [String] = [
            #"<?xml version="1.0" encoding="UTF-8"?>"#,
            #"<graphml xmlns="http://graphml.graphdrawing.org/xmlns">"#,
            #"  <key id="label" for="node" attr.name="label" attr.type="string"/>"#,
            #"  <key id="score" for="edge" attr.name="score" attr.type="double"/>"#,
            #"  <graph id="keyword_graph" edgedefault="undirected">"#
        ]
        lines.reserveCapacity(records.count + edges.count + 8)

        for record in records {
            lines.append(#"    <node id="\#(xml(record.id))"><data key="label">\#(xml(record.text))</data></node>"#)
        }

        for (index, edge) in edges.enumerated() {
            lines.append(#"    <edge id="e\#(index + 1)" source="\#(xml(edge.sourceID))" target="\#(xml(edge.targetID))"><data key="score">\#(edge.score)</data></edge>"#)
        }

        lines.append("  </graph>")
        lines.append("</graphml>")
        return lines.joined(separator: "\n") + "\n"
    }

    private func csv(_ value: String) -> String {
        let escaped = value.replacingOccurrences(of: "\"", with: "\"\"")
        return "\"\(escaped)\""
    }

    private func xml(_ value: String) -> String {
        value
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "\"", with: "&quot;")
            .replacingOccurrences(of: "'", with: "&apos;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
    }

    private static func fileURL(from path: String) -> URL {
        if path.hasPrefix("/") {
            return URL(fileURLWithPath: path).standardizedFileURL
        }

        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        return root.appendingPathComponent(path).standardizedFileURL
    }
}

private struct GraphJSONNode: Codable {
    let id: String
    let text: String
    let purpose: EmbeddingPurpose
    let model: String
    let dimension: Int
}

private struct GraphJSONOutput: Codable {
    let nodes: [GraphJSONNode]
    let edges: [KeywordGraphEdge]
}

private enum GraphError: Error {
    case helpRequested
    case missingValue(option: String)
    case missingRequiredOption(String)
    case unexpectedArgument(String)
    case invalidFormat(String)
    case invalidTopK(String)
    case invalidThreshold(String)
    case inputEncodingFailed
    case outputEncodingFailed
    case invalidJSONLLine(line: Int, reason: String)
}

extension GraphError: LocalizedError {
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
        case .invalidFormat(let value):
            return "Invalid format '\(value)'. Expected 'csv', 'graphml', or 'json'."
        case .invalidTopK(let value):
            return "Invalid top-k '\(value)'. Expected a positive integer."
        case .invalidThreshold(let value):
            return "Invalid threshold '\(value)'. Expected a numeric score."
        case .inputEncodingFailed:
            return "Failed to read input as UTF-8."
        case .outputEncodingFailed:
            return "Failed to encode output as UTF-8."
        case .invalidJSONLLine(let line, let reason):
            return "Invalid JSONL at line \(line): \(reason)"
        }
    }
}
