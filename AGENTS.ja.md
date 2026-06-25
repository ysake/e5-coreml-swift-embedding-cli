# AGENTS.ja.md

このリポジトリは、エージェントによる実装引き継ぎを前提にしています。

## 最初に読むもの

作業前に以下を読んでください。

1. `README.ja.md`
2. `docs/agent-handoff.ja.md`
3. `docs/implementation-plan.ja.md`
4. `docs/model-conversion-notes.ja.md`

英語版が必要な場合は、`README.md` と対応する `.md` ファイルを参照してください。

## 開発方針

- 最初のマイルストーンは小さく保つ。
- iOS / visionOSサンプルより先に、SwiftPM CLIを動かす。
- Core ML推論はprotocolの裏に隠す。
- tokenizer assetsとmodel assetsは明確に分ける。
- pure Swift logicのテストを先に追加する。
- Git LFSを使う方針が決まるまでは、大きな生成済みモデルartifactをcommitしない。

## コマンド

実装後は以下を実行します。

```bash
swift build
swift test
swift run e5-embed "テスト"
```
