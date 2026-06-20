# E5 Core ML Swift

Swift Package Manager library and command-line sample for generating text embeddings locally with a Core ML converted E5 model.

This repository is a proof-of-concept for building a reusable embedding layer that can be shared with visionOS apps while keeping macOS CLI tools for local validation.

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
.package(url: "https://github.com/ysake/e5-coreml-swift-embedding-cli", branch: "main")
```

```swift
.product(name: "E5EmbeddingCore", package: "e5-coreml-swift-embedding-cli")
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

## Goal

Build a minimal Swift CLI that accepts multilingual text and returns an embedding vector.

```bash
swift run e5-embed "Find more storage space inside my car"
```

Expected output:

```json
{
  "model": "intfloat/multilingual-e5-small",
  "purpose": "query",
  "dimension": 384,
  "embedding": [0.0123, -0.0456, "... 382 more values"]
}
```

The `embedding` array contains `dimension` values. It is shortened in this README for readability.

## Initial target model

Use `intfloat/multilingual-e5-small` first.

Reasons:

- multilingual model, including Japanese
- relatively small and suitable for a CLI proof-of-concept
- 384-dimensional output
- good enough to validate local semantic search behavior before moving to larger models

## Scope

This repository should cover:

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

## Non-goals

This repository does not initially target:

- iOS app UI
- visionOS app UI
- vector database integration
- production model distribution
- remote embedding API

## Architecture

```text
Input text
  ↓
E5 prefix
  - query: ...
  - passage: ...
  ↓
Tokenizer
  - input_ids
  - attention_mask
  ↓
Core ML model
  ↓
L2-normalized embedding
  ↓
JSON output / similarity search
```

## Suggested package structure

```text
.
├── Package.swift
├── README.md
├── docs/
│   ├── agent-handoff.md
│   └── agent-handoff.ja.md
├── Models/
│   └── E5SmallEmbedding.mlpackage
├── Tokenizer/
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── Sources/
│   ├── E5EmbeddingCore/
│   │   ├── E5Embedder.swift
│   │   ├── EmbeddingPurpose.swift
│   │   ├── CoreMLEmbeddingModel.swift
│   │   └── CosineSimilarity.swift
│   └── E5EmbedCLI/
│       └── main.swift
└── Tests/
    └── E5EmbeddingCoreTests/
        └── E5EmbeddingCoreTests.swift
```

## CLI design

For command and option details, see [`docs/cli-usage.md`](docs/cli-usage.md).

## Implementation notes

- Use `intfloat/multilingual-e5-small` first.
- The Core ML model should output a single normalized vector.
- Mean pooling and L2 normalization should preferably be included in the converted Core ML model.
- Swift should handle tokenization, Core ML invocation, output formatting, and similarity calculation.
- Use `Float` for embedding values.
- If vectors are already L2-normalized, dot product can be used as cosine similarity.

## Model conversion

A Python conversion script is available under:

```text
scripts/convert_e5_small_to_coreml.py
```

The script:

1. Load `intfloat/multilingual-e5-small`.
2. Wrap the encoder with mean pooling.
3. Apply L2 normalization.
4. Convert to Core ML `.mlpackage`.
5. Save as `Models/E5SmallEmbedding.mlpackage`.

## Tokenizer assets

Tokenizer files should come from the same Hugging Face model repository as the converted model.

Expected files:

```text
Tokenizer/
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
```

## Acceptance criteria

- `swift build` succeeds.
- `swift test` succeeds.
- `swift run e5-embed "test"` returns JSON.
- Output vector dimension is 384.
- `query:` and `passage:` prefixes are handled by the CLI.
- Similar Japanese texts produce higher similarity than unrelated texts.

---

# E5 Core ML Swift（日本語）

Core MLに変換したE5系embeddingモデルを使って、Swift Package Managerのlibraryとコマンドラインツールからローカルで文章ベクトルを生成するためのリポジトリです。

macOS CLIでローカル検証しながら、`E5EmbeddingCore` を visionOS アプリから再利用できる構成にしています。

## 現在の状態

実装済み:

- `E5EmbeddingCore` を持つ SwiftPM package
- JSONを返す `e5-embed` embedding CLI
- JSONを返す `e5-embed-similarity` 類似度CLI
- JSONLを返す `e5-embed-batch` batch embedding CLI
- exact top-kでedgeを作る `e5-keyword-graph` keyword graph CLI
- E5の `query:` / `passage:` prefix処理
- Hugging Face `swift-transformers` によるローカルtokenizer読み込み
- Core ML入力生成とprediction呼び出し
- visionOS package platform 対応
- app bundle内のCore ML model / tokenizer asset lookup
- `scripts/convert_e5_small_to_coreml.py` の変換スクリプト
- pure Swift logic、Core ML入出力、asset未配置エラーのunit test

コミットしていないもの:

- `Models/` 配下の生成済みCore ML model artifact
- `Tokenizer/` 配下の生成済みtokenizer assets

これらは変換スクリプトでローカル生成します。リポジトリで管理する場合はGit LFSの利用を検討してください。

## セットアップ

Swift packageをビルド・テストします。

```bash
swift build
swift test
```

Core MLモデルとtokenizer assetsを生成します。

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements-convert.txt
python scripts/convert_e5_small_to_coreml.py --validate
```

変換スクリプトは `FLOAT32` を標準にしています。BrainCopy の visionOS Simulator 検証では `FLOAT16` 変換モデルがゼロベクトルを返したため、visionOS 組み込みではこの標準から始めてください。

スクリプトは以下を書き出します。

```text
Models/E5SmallEmbedding.mlpackage
Tokenizer/
```

model packageは大きくなるため、デフォルトではgit管理から除外しています。

## 使い方

Embedding:

```bash
swift run e5-embed "車内の収納を増やしたい"
swift run e5-embed --purpose passage "セレナの荷室容量を増やすには、車内収納やルーフボックスを検討する。"
```

asset pathを明示する場合:

```bash
swift run e5-embed \
  --model Models/E5SmallEmbedding.mlpackage \
  --tokenizer Tokenizer \
  --max-length 128 \
  "テスト"
```

類似度:

```bash
swift run e5-embed-similarity \
  --query "車内の収納を増やしたい" \
  --passage "セレナの荷物積載量を増やす方法"
```

Batch embedding と keyword graph:

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

model assetsなしの開発用smoke test:

```bash
swift run e5-embed --backend deterministic "テスト"
swift run e5-embed-similarity --backend deterministic \
  --query "車内の収納を増やしたい" \
  --passage "セレナの荷物積載量を増やす方法"
```

コマンドとオプションの詳細は [`docs/cli-usage.ja.md`](docs/cli-usage.ja.md) を参照してください。

## visionOSアプリ組み込み

詳細な setup、asset 同梱手順、runtime behavior は [`docs/visionos-app-integration.ja.md`](docs/visionos-app-integration.ja.md) を参照してください。

`E5EmbeddingCore` はアプリ実行時に E5 model をダウンロードしません。アプリを build する前に Core ML model と tokenizer files を生成し、その生成済み assets を app target に同梱します。

Package URLを追加し、library productに依存します。

```swift
.package(url: "https://github.com/ysake/e5-coreml-swift-embedding-cli", branch: "main")
```

```swift
.product(name: "E5EmbeddingCore", package: "e5-coreml-swift-embedding-cli")
```

生成済みassetsをアプリtargetに追加します。

```text
E5SmallEmbedding.mlpackage
Tokenizer/
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
```

Xcodeはmodelを `E5SmallEmbedding.mlmodelc` としてbundleへ配置する場合があります。tokenizer filesがbundle rootにflattenされる場合もあります。`CoreMLTextEmbeddingAssets.appBundle()` は両方のlayoutを探します。

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

Tokenizer入力はE5/XLM-R系に合わせています。`<pad>` は token ID `1`、padding位置は `attention_mask = 0`、truncation時は可能な限り終端special tokenを保持します。Core ML出力は `float16` / `float32` / `double` の `MLMultiArray` を `[Float]` として読み出します。

## 目的

日本語または多言語のテキストを入力し、embeddingベクトルを返す最小CLIを作ります。

```bash
swift run e5-embed "車内の収納を増やしたい"
```

期待する出力例:

```json
{
  "model": "intfloat/multilingual-e5-small",
  "purpose": "query",
  "dimension": 384,
  "embedding": [0.0123, -0.0456, "... 残り382個の値"]
}
```

`embedding` 配列には `dimension` 個の値が入ります。このREADMEでは読みやすさのために省略しています。

## 最初に使うモデル

まずは `intfloat/multilingual-e5-small` を使います。

理由:

- 日本語を含む多言語モデル
- CLIでのPoCに向いた比較的小さなモデル
- 出力が384次元
- より大きなモデルへ進む前に、ローカル意味検索の挙動を検証しやすい

## 対象範囲

このリポジトリで扱うこと:

- Swift Package ManagerのCLI
- 再利用可能な `E5EmbeddingCore` library product
- visionOSアプリからのSwiftPM dependency利用
- ローカルtokenizer実行
- Core MLモデル推論
- E5形式の `query:` / `passage:` prefix
- 正規化済みembeddingベクトル出力
- シンプルな類似度計算
- keywordのbatch embedding
- exact top-kによるkeyword graph生成

## 対象外

初期段階では以下は扱いません。

- iOSアプリUI
- visionOSアプリUI
- ベクトルDB連携
- 本番向けモデル配布設計
- リモートembedding API

## アーキテクチャ

```text
入力テキスト
  ↓
E5 prefix付与
  - query: ...
  - passage: ...
  ↓
tokenizer
  - input_ids
  - attention_mask
  ↓
Core ML model
  ↓
L2正規化済みembedding
  ↓
JSON出力 / 類似検索
```

## 推奨ディレクトリ構成

```text
.
├── Package.swift
├── README.md
├── docs/
│   ├── agent-handoff.md
│   └── agent-handoff.ja.md
├── Models/
│   └── E5SmallEmbedding.mlpackage
├── Tokenizer/
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── Sources/
│   ├── E5EmbeddingCore/
│   │   ├── E5Embedder.swift
│   │   ├── EmbeddingPurpose.swift
│   │   ├── CoreMLEmbeddingModel.swift
│   │   └── CosineSimilarity.swift
│   └── E5EmbedCLI/
│       └── main.swift
└── Tests/
    └── E5EmbeddingCoreTests/
        └── E5EmbeddingCoreTests.swift
```

## CLI設計

コマンドとオプションの詳細は [`docs/cli-usage.ja.md`](docs/cli-usage.ja.md) を参照してください。

## 実装メモ

- 最初は `intfloat/multilingual-e5-small` を使う。
- Core MLモデルの出力は単一の正規化済みベクトルにする。
- mean poolingとL2 normalizationは、できればCore ML変換後モデルに含める。
- Swift側はtokenization、Core ML呼び出し、出力整形、類似度計算に集中する。
- embedding値は `Float` で扱う。
- ベクトルがL2正規化済みなら、dot productをcosine similarityとして扱える。

## モデル変換

Python変換スクリプトは以下にあります。

```text
scripts/convert_e5_small_to_coreml.py
```

スクリプトの役割:

1. `intfloat/multilingual-e5-small` を読み込む。
2. encoderをmean pooling付きでwrapする。
3. L2 normalizationを適用する。
4. Core ML `.mlpackage` に変換する。
5. `Models/E5SmallEmbedding.mlpackage` として保存する。

## tokenizer assets

tokenizerファイルは、Core ML変換元と同じHugging Faceモデルリポジトリから取得します。

想定ファイル:

```text
Tokenizer/
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
```

## 完了条件

- `swift build` が成功する。
- `swift test` が成功する。
- `swift run e5-embed "テスト"` がJSONを返す。
- 出力ベクトル次元が384である。
- CLI側で `query:` / `passage:` prefixを扱える。
- 関連する日本語文同士の類似度が、無関係な文より高くなる。
