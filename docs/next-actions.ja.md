# 次のアクション

1. SwiftPM packageを初期化する。
2. library targetとCLI targetに分ける。
3. `EmbeddingPurpose` と `TextEmbedder` protocolを実装する。
4. pure Swift部分のunit testを追加する。
5. tokenizer連携を追加する。
6. Core ML model loaderを追加する。
7. Python変換スクリプトを追加する。
8. CLIからJSON出力できるようにする。

まずはモデルassetが未配置でも、読みやすいエラーを返すところまで作るとよいです。
