# ステータス

このリポジトリには、SwiftPM CLI実装マイルストーンが入りました。

- `E5EmbeddingCore` library target
- `e5-embed` CLI target
- `e5-embed-similarity` CLI target
- `E5EmbeddingCoreTests` test target
- E5の `query:` / `passage:` prefix処理
- `swift-transformers` によるローカルtokenizer読み込み
- Core ML model loadingとprediction
- cosine/dot product helper
- Core ML入力のpadding / attention mask helper
- `intfloat/multilingual-e5-small` 向けPython変換スクリプト
- 読みやすいasset未配置エラー
- model/tokenizer filesがある場合だけ走るintegration test

生成されたCore ML model artifactは意図的にコミットしていません。`scripts/convert_e5_small_to_coreml.py` で `Models/E5SmallEmbedding.mlpackage` と `Tokenizer/` をローカル生成するか、リポジトリで管理する判断になった場合はGit LFSを使ってください。
