# visionOS App Integration

This package does not download the E5 model at app runtime. A visionOS app must bundle the converted Core ML model and tokenizer assets with the app.

## Asset lifecycle

The expected flow is:

1. During development or release preparation, run the conversion script on a Mac.
2. The script downloads `intfloat/multilingual-e5-small` through Python/Hugging Face tooling, converts it to Core ML, and writes tokenizer files.
3. Add the generated model and tokenizer files to the app target as resources.
4. At app runtime, `E5EmbeddingCore` loads assets from the app bundle and performs local tokenization and Core ML inference.

`E5EmbeddingCore` itself does not contact Hugging Face, download model weights, or create model files at runtime.

## Generate assets

From this repository:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements-convert.txt
python scripts/convert_e5_small_to_coreml.py --validate
```

The script writes:

```text
Models/E5SmallEmbedding.mlpackage
Tokenizer/
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
```

The conversion script defaults to `FLOAT32`. Use that default for visionOS integration; BrainCopy visionOS Simulator testing saw `FLOAT16` converted models return zero vectors with L2 norm `0.0000`.

## Add assets to the app target

Add these generated assets to the visionOS app target:

```text
E5SmallEmbedding.mlpackage
Tokenizer/
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
```

Xcode may compile `E5SmallEmbedding.mlpackage` into `E5SmallEmbedding.mlmodelc` in the app bundle. Depending on how resources are added, tokenizer JSON files may stay under `Tokenizer/` or be flattened into the app bundle resource root.

`CoreMLTextEmbeddingAssets.appBundle()` checks these layouts:

```text
E5SmallEmbedding.mlmodelc
E5SmallEmbedding.mlpackage
Tokenizer/tokenizer.json
Tokenizer/tokenizer_config.json
Tokenizer/special_tokens_map.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

## Add the package

Until the repository is renamed, use the repository URL identity in the product dependency:

```swift
.package(url: "https://github.com/ysake/e5-coreml-swift-embedding-cli", branch: "main")
```

```swift
.product(name: "E5EmbeddingCore", package: "e5-coreml-swift-embedding-cli")
```

## Load from the app bundle

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

let embedding = try await embedder.embed(
    "車内の収納を増やしたい",
    purpose: .query
)
```

`assetStatus()` is intended for diagnostics and UI readiness reporting. It reports whether assets are ready, which model/tokenizer paths were selected, an approximate model size, and a missing-asset error message when resolution fails.

## Runtime behavior

- Tokenization runs locally through `swift-transformers` and local tokenizer JSON files.
- Core ML inference runs locally through the bundled model.
- E5 prefixes are applied in Swift with `query:` or `passage:`.
- Padding uses the E5/XLM-R `<pad>` token ID `1`.
- Padding positions use `attention_mask = 0`.
- Truncation preserves the terminal special token when possible.
- Core ML output extraction accepts `float16`, `float32`, and `double` `MLMultiArray` values and returns `[Float]`.

## Size and distribution notes

The generated `E5SmallEmbedding.mlpackage` was about 448 MB in the BrainCopy PoC, and tokenizer assets were about 16 MB. Bundling those assets increases app size. This package currently assumes bundled assets; on-demand downloads, asset packs, and remote model distribution are outside the current scope.
