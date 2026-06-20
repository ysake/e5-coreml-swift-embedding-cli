# Agent Handoff: E5 Core ML Swift Embedding CLI

## Background

This repository is intended to become a Swift Package Manager command-line sample that generates embedding vectors from keywords or sentences on Apple platforms.

The long-term goal is to reuse the same embedding layer in iOS and visionOS apps. The first milestone is a macOS command-line tool built with SwiftPM.

## Product goal

Create a working CLI:

```bash
swift run e5-embed "車内の収納を増やしたい"
```

It should output JSON containing:

- model name
- purpose
- embedding dimension
- embedding vector

## Technical direction

Use:

- Swift Package Manager
- Core ML
- Hugging Face `swift-transformers` for tokenization
- `intfloat/multilingual-e5-small` as the initial embedding model

## Important E5 behavior

E5 models expect prefixes:

```text
query: <text>
passage: <text>
```

The CLI should expose this as:

```bash
swift run e5-embed --purpose query "..."
swift run e5-embed --purpose passage "..."
```

Default purpose should be `query`.

## First implementation milestone

Create a minimal SwiftPM package with:

```text
Package.swift
Sources/E5EmbeddingCore/
Sources/E5EmbedCLI/
Tests/E5EmbeddingCoreTests/
```

Suggested modules:

```swift
public enum EmbeddingPurpose {
    case query
    case passage

    public func applyPrefix(to text: String) -> String
}
```

```swift
public protocol TextEmbedder {
    func embed(_ text: String, purpose: EmbeddingPurpose) async throws -> [Float]
}
```

```swift
public struct CosineSimilarity {
    public static func dot(_ lhs: [Float], _ rhs: [Float]) -> Float
}
```

## Core ML integration

Implement a `CoreMLTextEmbedder` that:

1. Loads `E5SmallEmbedding.mlmodelc` or `.mlpackage`.
2. Tokenizes text.
3. Builds `input_ids` and `attention_mask`.
4. Runs Core ML prediction.
5. Returns `[Float]`.

Initial max sequence length:

```text
128
```

Make it configurable later.

## Model conversion script

Add:

```text
scripts/convert_e5_small_to_coreml.py
```

The script should wrap the PyTorch model so the Core ML output is already:

```text
mean pooled + L2 normalized
```

This keeps Swift-side logic simple.

## CLI behavior

### Basic embedding

```bash
swift run e5-embed "車内の収納を増やしたい"
```

### Passage embedding

```bash
swift run e5-embed --purpose passage "車内収納を増やすには、天井ネットやラゲッジ収納を使う。"
```

### Pretty output

Default JSON is fine.

Optional later:

```bash
swift run e5-embed --format raw "..."
swift run e5-embed --format json "..."
```

## Tests

Add tests for:

- `EmbeddingPurpose.applyPrefix`
- dot product calculation
- vector dimension check, if model asset exists
- graceful error when model asset is missing
- graceful error when tokenizer asset is missing

## Suggested acceptance criteria

- `swift build` passes.
- `swift test` passes.
- CLI accepts Japanese input.
- CLI outputs JSON.
- Embedding dimension is 384 for `multilingual-e5-small`.
- Similarity demo shows related Japanese sentences have higher score than unrelated ones.

## Nice-to-have follow-up issues

- Add model conversion script.
- Add GitHub Actions for `swift build` and `swift test`.
- Add similarity demo CLI.
- Add benchmark command.
- Add iOS sample target.
- Add visionOS sample target.
- Add support for `multilingual-e5-base`.
