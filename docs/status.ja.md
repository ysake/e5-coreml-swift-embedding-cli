# ステータス

このリポジトリには、最初のSwiftPM実装マイルストーンが入りました。

- `E5EmbeddingCore` library target
- `e5-embed` CLI target
- `E5EmbeddingCoreTests` test target
- E5の `query:` / `passage:` prefix処理
- cosine/dot product helper
- Core ML入力のpadding / attention mask helper
- 今後のCore ML backend向けの読みやすいasset未配置エラー

Core MLモデルとtokenizerはまだ実推論には接続していません。次の作業は、`TextEmbedder` の背後にtokenizer連携とCore ML predictionを追加することです。
