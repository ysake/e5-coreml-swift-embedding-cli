# Next Actions

Completed:

1. Initialize the SwiftPM package.
2. Split into a library target and a CLI target.
3. Implement `EmbeddingPurpose` and `TextEmbedder`.
4. Add unit tests for pure Swift logic.
5. Output JSON from the CLI with the explicit deterministic development backend.
6. Return readable errors when model assets are not yet included.

Next:

1. Add tokenizer integration.
2. Add Core ML model loading and prediction.
3. Add the Python conversion script.
4. Add the similarity demo.
5. Make `swift run e5-embed "テスト"` return real model JSON once assets are present.
