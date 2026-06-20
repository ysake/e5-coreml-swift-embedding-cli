# Implementation Plan: Generate E5 Embeddings with a SwiftPM CLI

## Purpose

Implement a Swift Package Manager command-line tool that generates embedding vectors from text or keywords.

The first target is a macOS CLI. The embedding logic should be implemented as a library target so it can later be reused from iOS and visionOS apps.

## First goal

Make this command work:

```bash
swift run e5-embed "車内の収納を増やしたい"
```

Expected output:

```json
{
  "model": "intfloat/multilingual-e5-small",
  "purpose": "query",
  "dimension": 384,
  "embedding": [0.0123, -0.0456]
}
```

## Suggested target structure

```text
E5EmbeddingCore
  - library target for embedding logic

E5EmbedCLI
  - command-line executable target

E5EmbeddingCoreTests
  - unit tests
```

## Suggested file structure

```text
Package.swift
Sources/
  E5EmbeddingCore/
    EmbeddingPurpose.swift
    TextEmbedder.swift
    CoreMLTextEmbedder.swift
    CoreMLInputBuilder.swift
    CosineSimilarity.swift
    EmbeddingError.swift
  E5EmbedCLI/
    main.swift
Tests/
  E5EmbeddingCoreTests/
    EmbeddingPurposeTests.swift
    CosineSimilarityTests.swift
scripts/
  convert_e5_small_to_coreml.py
Models/
  E5SmallEmbedding.mlpackage
Tokenizer/
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
```

## Implementation steps

### 1. Create the SwiftPM package

```bash
swift package init --type executable
```

Then split it into a library target and a CLI target.

### 2. Add `EmbeddingPurpose`

```swift
public enum EmbeddingPurpose: String, Sendable {
    case query
    case passage

    public func applyPrefix(to text: String) -> String {
        switch self {
        case .query:
            return "query: \(text)"
        case .passage:
            return "passage: \(text)"
        }
    }
}
```

### 3. Add `TextEmbedder`

```swift
public protocol TextEmbedder: Sendable {
    func embed(_ text: String, purpose: EmbeddingPurpose) async throws -> [Float]
}
```

### 4. Implement tokenizer integration

Use Hugging Face `swift-transformers` to load tokenizer assets bundled in the repository.

Swift-side tokenizer output should become:

```text
input_ids: [Int32]
attention_mask: [Int32]
```

Initial max length should be 128.

### 5. Implement Core ML integration

Load `MLModel` and build `MLMultiArray` inputs.

Inputs:

```text
input_ids: shape [1, 128], Int32
attention_mask: shape [1, 128], Int32
```

Output:

```text
embedding: shape [1, 384], Float32 or Float16
```

### 6. Implement CLI

Keep the first implementation simple. A full argument parser is optional.

Supported commands:

```bash
swift run e5-embed "..."
swift run e5-embed --purpose query "..."
swift run e5-embed --purpose passage "..."
```

Output should be JSON.

### 7. Add model conversion script

Add:

```text
scripts/convert_e5_small_to_coreml.py
```

The conversion script should include:

- mean pooling
- L2 normalization

This keeps the Swift implementation simple.

### 8. Add tests

Minimum tests:

- `EmbeddingPurpose.applyPrefix`
- dot product
- JSON output structure
- readable errors when model/tokenizer assets are missing

## Implementation notes

- E5 prefixes are important: `query:` and `passage:`.
- Do not use the same prefix for both query and passage embeddings.
- Core ML model and tokenizer assets should come from the same Hugging Face model.
- `.mlpackage` can be large; consider Git LFS or keeping only generation steps in git.
- Validate the CLI first. iOS and visionOS samples should be follow-up work.

## Acceptance criteria

- `swift build` passes.
- `swift test` passes.
- `swift run e5-embed "テスト"` returns JSON.
- Embedding dimension is 384.
- `--purpose query` and `--purpose passage` work.
- README includes setup and usage examples.
