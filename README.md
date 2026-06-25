# E5 Core ML Swift

[日本語版](README.ja.md)

Swift Package Manager library and command-line tools for generating text embeddings locally with a Core ML converted E5 model.

This repository is a proof of concept for building a reusable embedding layer that can be shared with visionOS apps while keeping macOS CLI tools for local validation.

## Current status

Implemented:

- SwiftPM package with `E5EmbeddingCore`
- `e5-embed` JSON embedding CLI
- `e5-embed-similarity` JSON similarity CLI
- `e5-embed-batch` JSONL batch embedding CLI
- `e5-keyword-graph` exact top-k keyword graph CLI
- E5 `query:` / `passage:` prefix handling
- local tokenizer loading with Hugging Face `swift-transformers`
- Core ML input creation and prediction wiring
- visionOS package platform support
- app-bundle asset lookup for Core ML model and tokenizer files
- conversion script at `scripts/convert_e5_small_to_coreml.py`
- unit tests for pure Swift logic, Core ML input/output handling, and missing-asset errors

Not committed:

- generated Core ML model artifacts under `Models/`
- generated tokenizer assets under `Tokenizer/`

Generate those locally with the conversion script, or decide to track them later with Git LFS.

## Why not `NLContextualEmbedding`?

On Apple platforms, `NLContextualEmbedding` should usually be the first option to try for local text embeddings. It is provided by the NaturalLanguage framework and avoids bundling a custom model with the app.

This package exists because the project needed Japanese semantic search on visionOS, and `NLContextualEmbedding(language: .japanese)` did not work reliably in local visionOS testing.

Observed behavior:

- `NLContextualEmbedding(language: .english)` worked on visionOS.
- `NLContextualEmbedding(language: .japanese)` worked on iOS.
- `NLContextualEmbedding(language: .japanese)` failed on visionOS 26.x.
- The same issue was still observed on visionOS 27 beta.
- Feedback has been filed with Apple.

Because Japanese embeddings were needed on visionOS, relying only on `NLContextualEmbedding` was not enough. This repository explores an app-controlled fallback path: converting a known multilingual E5 embedding model to Core ML, bundling the model and tokenizer with the app, and running embedding locally.

This is heavier than using the system framework, but it gives the app control over:

- the exact embedding model
- Japanese and multilingual behavior
- E5 `query:` / `passage:` search-compatible embeddings
- offline execution
- compatibility with server-side or Python-generated E5 vectors

Use `NLContextualEmbedding` first when it works for the target language and platform. Use this package when the app needs E5-compatible embeddings, stable Japanese behavior on visionOS, or matching vectors across app-side and server-side pipelines.

## Setup

Build and test the Swift package:

```bash
swift build
swift test
```

Generate the Core ML model and tokenizer assets:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements-convert.txt
python scripts/convert_e5_small_to_coreml.py --validate
```

The conversion script defaults to `FLOAT32`. Use that default for visionOS integration; BrainCopy visionOS Simulator testing saw `FLOAT16` converted models return zero vectors.

The script writes:

```text
Models/E5SmallEmbedding.mlpackage
Tokenizer/
```

The model package can be large and is ignored by git by default.

## Usage

Embedding:

```bash
swift run e5-embed "Find more storage space inside my car"
swift run e5-embed --purpose passage "Cargo organizers and roof boxes can increase available storage in a vehicle."
```

Custom asset paths:

```bash
swift run e5-embed \
  --model Models/E5SmallEmbedding.mlpackage \
  --tokenizer Tokenizer \
  --max-length 128 \
  "Find more storage space inside my car"
```

Similarity:

```bash
swift run e5-embed-similarity \
  --query "Find more storage space inside my car" \
  --passage "Cargo organizers and roof boxes can increase available storage in a vehicle."
```

Batch embedding and keyword graph:

```bash
swift run e5-embed-batch \
  --input keywords.txt \
  --output embeddings.jsonl

swift run e5-keyword-graph \
  --input embeddings.jsonl \
  --output edges.csv \
  --top-k 10 \
  --threshold 0.82

swift run e5-keyword-graph \
  --input embeddings.jsonl \
  --output graph.dot \
  --format dot \
  --top-k 10 \
  --threshold 0.82
```

Development smoke test without model assets:

```bash
swift run e5-embed --backend deterministic "Smoke test"
swift run e5-embed-similarity --backend deterministic \
  --query "Find more storage space inside my car" \
  --passage "Cargo organizers and roof boxes can increase available storage in a vehicle."
```

For full command and option details, see [`docs/cli-usage.md`](docs/cli-usage.md).

## visionOS app integration

For detailed setup, asset packaging, and runtime behavior, see [`docs/visionos-app-integration.md`](docs/visionos-app-integration.md).

`E5EmbeddingCore` does not download the E5 model at app runtime. Generate the Core ML model and tokenizer files before building the app, then bundle those generated assets with the app target.

Add the package by URL and depend on the library product:

```swift
.package(url: "https://github.com/ysake/e5-coreml-swift", branch: "main")
```

```swift
.product(name: "E5EmbeddingCore", package: "e5-coreml-swift")
```

Add the generated assets to the app target:

```text
E5SmallEmbedding.mlpackage
Tokenizer/
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
```

Xcode may compile the model into `E5SmallEmbedding.mlmodelc` in the app bundle. The tokenizer files may also be flattened into the bundle resource root. `CoreMLTextEmbeddingAssets.appBundle()` checks both layouts:

```swift
import E5EmbeddingCore
import Foundation

let assets = CoreMLTextEmbeddingAssets.appBundle(.main)
let embedder = try CoreMLTextEmbedder(assets: assets)

let status = embedder.assetStatus()
guard status.isReady else {
    throw NSError(
        domain: "E5Embedding",
        code: 1,
        userInfo: [NSLocalizedDescriptionKey: status.errorDescription ?? "Missing E5 assets"]
    )
}

let queryEmbedding = try await embedder.embed(
    "車内の収納を増やしたい",
    purpose: .query
)
```

Tokenizer inputs are built for E5/XLM-R style models: `<pad>` uses token ID `1`, padding positions get `attention_mask = 0`, and truncation preserves the terminal special token when possible. Core ML output extraction accepts `float16`, `float32`, and `double` `MLMultiArray` values and returns `[Float]`.

## Target model

Use `intfloat/multilingual-e5-small` first.

Reasons:

- multilingual model, including Japanese
- relatively small and suitable for a CLI proof of concept
- 384-dimensional output
- good enough to validate local semantic search behavior before moving to larger models

## Scope

This repository covers:

- Swift Package Manager CLI
- reusable `E5EmbeddingCore` library product
- visionOS app package consumption
- local tokenizer execution
- Core ML model inference
- E5-style `query:` / `passage:` prefixes
- normalized embedding vector output
- simple similarity calculation
- batch keyword embedding
- exact top-k keyword graph generation

This repository does not currently target:

- iOS app UI
- visionOS app UI
- vector database integration
- production model distribution
- remote embedding API

## Architecture

```text
Input text
  -> E5 prefix: query: ... / passage: ...
  -> Tokenizer: input_ids + attention_mask
  -> Core ML model
  -> L2-normalized embedding
  -> JSON output / similarity search
```

## Model conversion

A Python conversion script is available under:

```text
scripts/convert_e5_small_to_coreml.py
```

The script:

1. Loads `intfloat/multilingual-e5-small`.
2. Wraps the encoder with mean pooling.
3. Applies L2 normalization.
4. Converts to Core ML `.mlpackage`.
5. Saves as `Models/E5SmallEmbedding.mlpackage`.

Tokenizer files should come from the same Hugging Face model repository as the converted model.

Expected files:

```text
Tokenizer/
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
```

## Documentation

- [`docs/index.md`](docs/index.md): documentation index
- [`docs/index.ja.md`](docs/index.ja.md): Japanese documentation index
