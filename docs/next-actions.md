# Next Actions

Completed:

1. Initialize the SwiftPM package.
2. Split into a library target and a CLI target.
3. Implement `EmbeddingPurpose` and `TextEmbedder`.
4. Add unit tests for pure Swift logic.
5. Add tokenizer integration with local tokenizer assets.
6. Add Core ML model loading and prediction.
7. Add the Python conversion script.
8. Add JSON output from the embedding CLI.
9. Add the similarity demo CLI.
10. Return readable errors when model assets are not yet included.

Next:

1. Run `scripts/convert_e5_small_to_coreml.py` in an environment with `torch`, `transformers`, `coremltools`, and `numpy`.
2. Decide whether generated model artifacts stay local or move to Git LFS.
3. Validate real embeddings with `swift run e5-embed "テスト"` after assets are present.
4. Validate semantic ranking with `swift run e5-embed-similarity`.
5. Add CI for `swift build` and `swift test`.
