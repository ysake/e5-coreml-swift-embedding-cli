# Status

This repository now has the first SwiftPM implementation milestone:

- `E5EmbeddingCore` library target
- `e5-embed` CLI target
- `E5EmbeddingCoreTests` test target
- E5 `query:` / `passage:` prefix handling
- cosine/dot-product helpers
- Core ML input padding/masking helper
- readable missing-asset errors for the future Core ML backend

The Core ML model and tokenizer are not wired yet. The next task is to add tokenizer integration and real Core ML prediction behind `TextEmbedder`.
