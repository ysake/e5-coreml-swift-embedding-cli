# エージェント引き継ぎ: E5 Core ML Swift Embedding CLI

## 背景

このリポジトリは、Appleプラットフォーム上でキーワードや文章からembeddingベクトルを生成するSwift Package Managerのコマンドラインサンプルを作るためのものです。

長期的には、ここで作ったembedding層をiOS / visionOSアプリでも再利用することを想定しています。最初のマイルストーンは、macOS上で動くSwiftPM製CLIです。

## プロダクト目標

以下のようなCLIを作ります。

```bash
swift run e5-embed "車内の収納を増やしたい"
```

出力はJSONで、以下を含めます。

- モデル名
- 用途
- embedding次元数
- embeddingベクトル

## 技術方針

使用予定:

- Swift Package Manager
- Core ML
- tokenizationにはHugging Face `swift-transformers`
- 初期embeddingモデルは `intfloat/multilingual-e5-small`

## E5モデルの重要な挙動

E5モデルでは、入力テキストに以下のprefixを付けることが重要です。

```text
query: <text>
passage: <text>
```

CLIでは以下のように扱えるようにします。

```bash
swift run e5-embed --purpose query "..."
swift run e5-embed --purpose passage "..."
```

デフォルトは `query` とします。

## 最初の実装マイルストーン

最小構成のSwiftPM packageを作ります。

```text
Package.swift
Sources/E5EmbeddingCore/
Sources/E5EmbedCLI/
Tests/E5EmbeddingCoreTests/
```

推奨モジュール:

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

## Core ML連携

`CoreMLTextEmbedder` を実装します。

役割:

1. `E5SmallEmbedding.mlmodelc` または `.mlpackage` を読み込む。
2. テキストをtokenizeする。
3. `input_ids` と `attention_mask` を作る。
4. Core ML predictionを実行する。
5. `[Float]` を返す。

初期の最大sequence length:

```text
128
```

後で設定可能にします。

## モデル変換スクリプト

以下を追加します。

```text
scripts/convert_e5_small_to_coreml.py
```

このスクリプトでは、PyTorchモデルをwrapしてCore MLモデルの出力が最初から以下になるようにします。

```text
mean pooled + L2 normalized
```

これによりSwift側の実装をシンプルにできます。

## CLIの挙動

### 基本のembedding

```bash
swift run e5-embed "車内の収納を増やしたい"
```

### passage embedding

```bash
swift run e5-embed --purpose passage "車内収納を増やすには、天井ネットやラゲッジ収納を使う。"
```

### 出力形式

デフォルトはJSONで十分です。

将来的なオプション:

```bash
swift run e5-embed --format raw "..."
swift run e5-embed --format json "..."
```

## テスト

追加するテスト:

- `EmbeddingPurpose.applyPrefix`
- dot product計算
- モデルassetがある場合のベクトル次元チェック
- モデルassetがない場合の読みやすいエラー
- tokenizer assetがない場合の読みやすいエラー

## 完了条件

- `swift build` が通る。
- `swift test` が通る。
- CLIが日本語入力を受け取れる。
- CLIがJSONを出力する。
- `multilingual-e5-small` のembedding次元が384である。
- 類似度デモで、関連する日本語文のスコアが無関係な文より高くなる。

## 追加Issue候補

- モデル変換スクリプトを追加する。
- `swift build` / `swift test` 用のGitHub Actionsを追加する。
- 類似度デモCLIを追加する。
- ベンチマークコマンドを追加する。
- iOSサンプルtargetを追加する。
- visionOSサンプルtargetを追加する。
- `multilingual-e5-base` に対応する。
