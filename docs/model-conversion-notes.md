# Model Conversion Notes: Convert multilingual-e5-small to Core ML

## Direction

The first model target is `intfloat/multilingual-e5-small` converted to Core ML.

To keep the Swift implementation simple, the converted Core ML model should output an embedding that is already:

```text
mean pooled + L2 normalized
```

In other words, Swift should only need to pass tokenizer outputs to Core ML and receive a normalized embedding vector.

## Inputs

Expected Core ML model inputs:

```text
input_ids: Int32, shape [1, 128]
attention_mask: Int32, shape [1, 128]
```

## Output

```text
embedding: Float32 or Float16, shape [1, 384]
```

`multilingual-e5-small` has a 384-dimensional embedding output.

## Python script location

```text
scripts/convert_e5_small_to_coreml.py
```

## Script outline

1. Load `intfloat/multilingual-e5-small` with Hugging Face Transformers.
2. Wrap the encoder with a PyTorch module.
3. Apply attention-mask-aware mean pooling to `last_hidden_state`.
4. Apply L2 normalization.
5. Trace the module with `torch.jit.trace`.
6. Convert to `.mlpackage` with `coremltools.convert`.
7. Save as `Models/E5SmallEmbedding.mlpackage`.

## Notes

- Tokenizer assets and the Core ML model must come from the same Hugging Face model.
- Initial max sequence length can be 128.
- Long-text support should be a follow-up task.
- `.mlpackage` may be large; consider Git LFS or only storing generation instructions in git.
- Use `FLOAT32` as the default `compute_precision` for app integration. BrainCopy visionOS Simulator testing saw `FLOAT16` converted models return zero vectors with L2 norm `0.0000`, while `FLOAT32` produced usable 384-dimensional embeddings.
- `FLOAT16` can still be tested explicitly with `--compute-precision FLOAT16` for macOS or device-specific experiments.
- If Swift expects the output name `embedding`, the conversion script should fix that output name.

## Minimal validation

Compare PyTorch output and Core ML output for the same text.

Check that:

- Output dimension is 384.
- L2 norm is approximately 1.0.
- Cosine similarity between PyTorch and Core ML output is high enough.
- Japanese input does not produce NaN or empty vectors.

## E5 prefix

E5 models expect input prefixes:

```text
query: 車内の収納を増やしたい
passage: セレナの荷室容量を増やすには、車内収納やルーフボックスを検討する。
```

Prefix handling should happen on the Swift side.
