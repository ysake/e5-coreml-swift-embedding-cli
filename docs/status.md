# Status

This repository now has the SwiftPM CLI implementation milestone:

- `E5EmbeddingCore` library target
- `e5-embed` CLI target
- `e5-embed-similarity` CLI target
- `E5EmbeddingCoreTests` test target
- E5 `query:` / `passage:` prefix handling
- local tokenizer loading through `swift-transformers`
- Core ML model loading and prediction
- cosine/dot-product helpers
- Core ML input padding/masking helper
- Python conversion script for `intfloat/multilingual-e5-small`
- readable missing-asset errors
- asset-gated integration test that runs only when model/tokenizer files are present

The generated Core ML model artifact is intentionally not committed. Use `scripts/convert_e5_small_to_coreml.py` to create `Models/E5SmallEmbedding.mlpackage` and `Tokenizer/` locally, or store generated artifacts with Git LFS if the repository later decides to track them.
