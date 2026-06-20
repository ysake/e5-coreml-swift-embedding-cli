# 次のアクション

完了:

1. SwiftPM packageを初期化する。
2. library targetとCLI targetに分ける。
3. `EmbeddingPurpose` と `TextEmbedder` protocolを実装する。
4. pure Swift部分のunit testを追加する。
5. 明示的なdeterministic開発backendでCLIからJSON出力できるようにする。
6. モデルasset未配置時に読みやすいエラーを返す。

次:

1. tokenizer連携を追加する。
2. Core ML model loadingとpredictionを追加する。
3. Python変換スクリプトを追加する。
4. 類似度デモを追加する。
5. asset配置後に `swift run e5-embed "テスト"` が実モデルのJSONを返すようにする。
