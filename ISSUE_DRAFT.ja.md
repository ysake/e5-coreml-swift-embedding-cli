# SwiftPM CLIサンプルでローカルE5 Core ML embeddingsを実装する

## 概要

Core MLに変換したE5モデルを使って、日本語または多言語テキストからembeddingベクトルを生成するSwiftPM CLIサンプルを実装する。

## ゴール

以下のコマンドが動く状態にする。

```bash
swift run e5-embed "車内の収納を増やしたい"
```

期待する出力:

```json
{
  "model": "intfloat/multilingual-e5-small",
  "purpose": "query",
  "dimension": 384,
  "embedding": []
}
```

## 要件

- SwiftPM CLI packageを作る。
- Core MLでローカル推論する。
- tokenizationにはHugging Face `swift-transformers` を使う。
- 初期モデルは `intfloat/multilingual-e5-small` とする。
- E5 prefixに対応する。
  - `query: <text>`
  - `passage: <text>`
- デフォルトpurposeは `query` とする。
- `[Float]` のembedding vectorを返す。
- CLIからJSONを出力する。
- シンプルなcosine similarity / dot product helperを追加する。

## 想定コマンド

```bash
swift run e5-embed "車内の収納を増やしたい"
swift run e5-embed --purpose query "車内の収納を増やしたい"
swift run e5-embed --purpose passage "車内収納を増やすには、天井ネットやラゲッジ収納を使う。"
```

## 推奨構成

```text
Sources/
  E5EmbeddingCore/
    E5Embedder.swift
    EmbeddingPurpose.swift
    CoreMLEmbeddingModel.swift
    CosineSimilarity.swift
  E5EmbedCLI/
    main.swift
Tests/
  E5EmbeddingCoreTests/
    E5EmbeddingCoreTests.swift
Models/
  E5SmallEmbedding.mlpackage
Tokenizer/
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
scripts/
  convert_e5_small_to_coreml.py
docs/
  agent-handoff.md
  agent-handoff.ja.md
```

## 完了条件

- [ ] `swift build` が通る
- [ ] `swift test` が通る
- [ ] CLIが日本語テキストを受け取れる
- [ ] CLIがJSONを出力する
- [ ] 出力ベクトル次元が384である
- [ ] `--purpose query` / `--purpose passage` が動く
- [ ] model/tokenizer assetがない場合に読みやすいエラーを出す
- [ ] READMEにセットアップ手順と使い方がある
- [ ] `docs/agent-handoff.md` と `docs/agent-handoff.ja.md` にエージェント向け実装方針がある

## メモ

Core MLモデル側にmean poolingとL2 normalizationを含めることで、Swift側の実装をシンプルに保つ。
