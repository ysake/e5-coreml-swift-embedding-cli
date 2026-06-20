# モデル変換メモ: multilingual-e5-small を Core ML に変換する

## 方針

最初は `intfloat/multilingual-e5-small` をCore MLに変換します。

Swift側の実装をシンプルにするため、変換済みCore MLモデルの出力は以下の状態にします。

```text
mean pooled + L2 normalized
```

つまり、Swift側ではtokenizerの結果をCore MLに渡すだけで、正規化済みembeddingベクトルを取得できる状態を目指します。

## 入力

Core MLモデルの入力は以下を想定します。

```text
input_ids: Int32, shape [1, 128]
attention_mask: Int32, shape [1, 128]
```

## 出力

```text
embedding: Float32 or Float16, shape [1, 384]
```

`multilingual-e5-small` のembedding次元は384です。

## Pythonスクリプトの配置場所

```text
scripts/convert_e5_small_to_coreml.py
```

## 変換スクリプトの概要

1. Hugging Face Transformersで `intfloat/multilingual-e5-small` を読み込む。
2. encoderをwrapper moduleで包む。
3. `last_hidden_state` に対してattention mask付きmean poolingを行う。
4. L2 normalizationを行う。
5. `torch.jit.trace` する。
6. `coremltools.convert` で `.mlpackage` に変換する。
7. `Models/E5SmallEmbedding.mlpackage` に保存する。

## 注意点

- tokenizer assetsとCore MLモデルは、必ず同じHugging Faceモデル由来にする。
- max sequence lengthは最初は128でよい。
- 長文対応は後続課題にする。
- `.mlpackage` はサイズが大きくなる可能性があるため、Git LFSを使うか、生成手順だけをgit管理するか検討する。
- アプリ組み込み向けの `compute_precision` は `FLOAT32` を標準にする。BrainCopy の visionOS Simulator 検証では、`FLOAT16` 変換モデルが L2 norm `0.0000` のゼロベクトルを返し、`FLOAT32` 変換モデルでは 384 次元 embedding が得られた。
- macOS や実機ごとの検証で `FLOAT16` を試す場合は、明示的に `--compute-precision FLOAT16` を指定する。
- Swift側で出力名 `embedding` を前提にするなら、変換スクリプト側でも出力名を固定する。

## 最小検証

Python側で同じテキストに対して、PyTorch出力とCore ML出力が近いことを確認します。

確認する観点:

- 出力次元が384である。
- L2 normがほぼ1.0である。
- PyTorch出力とCore ML出力のcosine similarityが十分高い。
- 日本語入力でNaNや空ベクトルにならない。

## E5 prefix

E5モデルでは入力にprefixを付けます。

```text
query: 車内の収納を増やしたい
passage: セレナの荷室容量を増やすには、車内収納やルーフボックスを検討する。
```

このprefix付与はSwift側で行う想定です。
