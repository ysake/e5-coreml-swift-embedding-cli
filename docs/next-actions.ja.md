# 次のアクション

完了:

1. SwiftPM packageを初期化する。
2. library targetとCLI targetに分ける。
3. `EmbeddingPurpose` と `TextEmbedder` protocolを実装する。
4. pure Swift部分のunit testを追加する。
5. ローカルtokenizer assetsとの連携を追加する。
6. Core ML model loadingとpredictionを追加する。
7. Python変換スクリプトを追加する。
8. embedding CLIからJSON出力できるようにする。
9. 類似度デモCLIを追加する。
10. モデルasset未配置時に読みやすいエラーを返す。

次:

1. `torch`、`transformers`、`coremltools`、`numpy` がある環境で `scripts/convert_e5_small_to_coreml.py` を実行する。
2. 生成されたmodel artifactsをローカル運用にするかGit LFSで管理するか決める。
3. assets配置後に `swift run e5-embed "テスト"` で実embeddingを検証する。
4. `swift run e5-embed-similarity` で意味的な順位付けを検証する。
5. `swift build` / `swift test` のCIを追加する。
