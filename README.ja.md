# E5 Core ML Swift

[English README](README.md)

Core ML に変換した E5 系 embedding モデルを使って、Swift Package Manager の library とコマンドラインツールからローカルで文章ベクトルを生成するためのリポジトリです。

macOS CLI でローカル検証しながら、`E5EmbeddingCore` を visionOS アプリから再利用できる構成にしています。

## 現在の状態

実装済み:

- `E5EmbeddingCore` を持つ SwiftPM package
- JSON を返す `e5-embed` embedding CLI
- JSON を返す `e5-embed-similarity` 類似度 CLI
- JSONL を返す `e5-embed-batch` batch embedding CLI
- exact top-k で edge を作る `e5-keyword-graph` keyword graph CLI
- E5 の `query:` / `passage:` prefix 処理
- Hugging Face `swift-transformers` によるローカル tokenizer 読み込み
- Core ML 入力生成と prediction 呼び出し
- visionOS package platform 対応
- app bundle 内の Core ML model / tokenizer asset lookup
- `scripts/convert_e5_small_to_coreml.py` の変換スクリプト
- pure Swift logic、Core ML 入出力、asset 未配置エラーの unit test

コミットしていないもの:

- `Models/` 配下の生成済み Core ML model artifact
- `Tokenizer/` 配下の生成済み tokenizer assets

これらは変換スクリプトでローカル生成します。リポジトリで管理する場合は Git LFS の利用を検討してください。

## なぜ `NLContextualEmbedding` ではなく E5 / Core ML なのか

Apple プラットフォームでローカル embedding を使う場合、通常はまず NaturalLanguage framework の `NLContextualEmbedding` を検討します。OS 標準 API で使えるため、独自モデルをアプリに同梱せずに済むからです。

ただし、この package は、visionOS で `NLContextualEmbedding(language: .japanese)` が期待通りに動作しないケースがあったことをきっかけに作っています。

観測した挙動:

- visionOS では `NLContextualEmbedding(language: .english)` は動作する。
- iOS では `NLContextualEmbedding(language: .japanese)` も動作する。
- visionOS 26.x では `NLContextualEmbedding(language: .japanese)` が失敗する。
- visionOS 27 beta でも同じ問題が残っていることを確認した。
- Apple には Feedback 済み。

この project では visionOS 上で日本語の semantic search を行いたかったため、`NLContextualEmbedding` だけに依存する構成では不十分でした。

そこでアプリ側で制御できる fallback path として、多言語 embedding model である E5 を Core ML に変換し、model と tokenizer をアプリに同梱して、visionOS 上でローカル inference できる構成を検証しています。

OS 標準 API を使うより app size や asset 管理の負担は増えますが、代わりに以下をアプリ側で制御できます。

- 使用する正確な embedding model
- 日本語・多言語での embedding 挙動
- E5 の `query:` / `passage:` prefix による検索互換 embedding
- オフライン実行
- server-side や Python 側で生成した E5 embedding との互換性

対象の言語・platform で `NLContextualEmbedding` が動く場合は、まずそちらを使うのが自然です。この package は、E5 互換の embedding が必要な場合、visionOS で日本語 embedding を安定して扱いたい場合、app 側と server-side pipeline で同じ embedding 空間を使いたい場合に使います。

## セットアップ

Swift package をビルド・テストします。

```bash
swift build
swift test
```

Core ML モデルと tokenizer assets を生成します。

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

model package は大きくなるため、デフォルトでは git 管理から除外しています。

## 使い方

Embedding:

```bash
swift run e5-embed "車内の収納を増やしたい"
swift run e5-embed --purpose passage "セレナの荷室容量を増やすには、車内収納やルーフボックスを検討する。"
```

asset path を明示する場合:

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

model assets なしの開発用 smoke test:

```bash
swift run e5-embed --backend deterministic "テスト"
swift run e5-embed-similarity --backend deterministic \
  --query "車内の収納を増やしたい" \
  --passage "セレナの荷物積載量を増やす方法"
```

コマンドとオプションの詳細は [`docs/cli-usage.ja.md`](docs/cli-usage.ja.md) を参照してください。

## visionOS アプリ組み込み

詳細な setup、asset 同梱手順、runtime behavior は [`docs/visionos-app-integration.ja.md`](docs/visionos-app-integration.ja.md) を参照してください。

`E5EmbeddingCore` はアプリ実行時に E5 model をダウンロードしません。アプリを build する前に Core ML model と tokenizer files を生成し、その生成済み assets を app target に同梱します。

Package URL を追加し、library product に依存します。

```swift
.package(url: "https://github.com/ysake/e5-coreml-swift", branch: "main")
```

```swift
.product(name: "E5EmbeddingCore", package: "e5-coreml-swift")
```

生成済み assets をアプリ target に追加します。

```text
E5SmallEmbedding.mlpackage
Tokenizer/
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
```

Xcode は model を `E5SmallEmbedding.mlmodelc` として bundle へ配置する場合があります。tokenizer files が bundle root に flatten される場合もあります。`CoreMLTextEmbeddingAssets.appBundle()` は両方の layout を探します。

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

Tokenizer 入力は E5/XLM-R 系に合わせています。`<pad>` は token ID `1`、padding 位置は `attention_mask = 0`、truncation 時は可能な限り終端 special token を保持します。Core ML 出力は `float16` / `float32` / `double` の `MLMultiArray` を `[Float]` として読み出します。

## 対象モデル

まずは `intfloat/multilingual-e5-small` を使います。

理由:

- 日本語を含む多言語モデル
- CLI での proof of concept に向いた比較的小さなモデル
- 出力が 384 次元
- より大きなモデルへ進む前に、ローカル意味検索の挙動を検証しやすい

## 対象範囲

このリポジトリで扱うこと:

- Swift Package Manager の CLI
- 再利用可能な `E5EmbeddingCore` library product
- visionOS アプリからの SwiftPM dependency 利用
- ローカル tokenizer 実行
- Core ML モデル推論
- E5 形式の `query:` / `passage:` prefix
- 正規化済み embedding ベクトル出力
- シンプルな類似度計算
- keyword の batch embedding
- exact top-k による keyword graph 生成

現時点では以下は扱いません。

- iOS アプリ UI
- visionOS アプリ UI
- ベクトル DB 連携
- 本番向けモデル配布設計
- リモート embedding API

## アーキテクチャ

```text
入力テキスト
  -> E5 prefix: query: ... / passage: ...
  -> Tokenizer: input_ids + attention_mask
  -> Core ML model
  -> L2 正規化済み embedding
  -> JSON 出力 / 類似検索
```

## モデル変換

Python 変換スクリプトは以下にあります。

```text
scripts/convert_e5_small_to_coreml.py
```

スクリプトの役割:

1. `intfloat/multilingual-e5-small` を読み込む。
2. encoder を mean pooling 付きで wrap する。
3. L2 normalization を適用する。
4. Core ML `.mlpackage` に変換する。
5. `Models/E5SmallEmbedding.mlpackage` として保存する。

tokenizer ファイルは、Core ML 変換元と同じ Hugging Face モデルリポジトリから取得します。

想定ファイル:

```text
Tokenizer/
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
```

## ドキュメント

- [`docs/index.ja.md`](docs/index.ja.md): 日本語版ドキュメント一覧
- [`docs/index.md`](docs/index.md): 英語版ドキュメント一覧
