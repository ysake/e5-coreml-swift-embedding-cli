# CLI の使い方

このドキュメントでは、この package が提供するコマンドラインツールの使い方を説明します。

## 前提

デフォルトの Core ML backend を使う前に、ローカルの Core ML モデルと tokenizer assets を生成します。

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements-convert.txt
python scripts/convert_e5_small_to_coreml.py --validate
```

変換スクリプトの標準は `FLOAT32` です。visionOS アプリ組み込みではこの標準から始めてください。macOS や実機ごとの挙動を明示的に試す場合だけ `--compute-precision FLOAT16` を指定します。

生成される assets:

```text
Models/E5SmallEmbedding.mlpackage
Tokenizer/
```

生成された assets は git 管理対象外です。

## `e5-embed`

1つの embedding vector を生成し、JSON を出力します。

```bash
swift run e5-embed [options] <text>
```

### オプション

| Option | Values | Default | Description |
| --- | --- | --- | --- |
| `--purpose` | `query`, `passage` | `query` | E5 の入力 prefix を指定します。検索クエリは `query`、文書や候補テキストは `passage` を使います。 |
| `--backend` | `coreml`, `deterministic` | `coreml` | embedding backend を選びます。`deterministic` は開発用 smoke test 専用です。 |
| `--model` | path | `Models/E5SmallEmbedding.mlpackage` または `.mlmodelc` | Core ML model の path です。 |
| `--tokenizer` | path | `Tokenizer` | tokenizer assets のディレクトリです。 |
| `--max-length` | positive integer | `128` | padding / truncation に使う token sequence length です。変換済み Core ML model と一致させる必要があります。 |
| `--model-name` | string | `intfloat/multilingual-e5-small` | JSON output の `model` field を上書きします。 |

### Query Embedding

```bash
swift run e5-embed "車内の収納を増やしたい"
```

以下と同じ意味です。

```bash
swift run e5-embed --purpose query "車内の収納を増やしたい"
```

### Passage Embedding

```bash
swift run e5-embed \
  --purpose passage \
  "セレナの荷室容量を増やすには、車内収納やルーフボックスを検討する。"
```

### Asset Path を明示する

```bash
swift run e5-embed \
  --model Models/E5SmallEmbedding.mlpackage \
  --tokenizer Tokenizer \
  --max-length 128 \
  "テスト"
```

### 開発用 Smoke Test

model assets がない状態で CLI の JSON 出力だけ確認したい場合に使います。この vector は deterministic ですが、意味検索用の embedding ではありません。

```bash
swift run e5-embed --backend deterministic "テスト"
```

## `e5-embed-similarity`

query と passage を embedding し、dot product を JSON で出力します。変換済み E5 model は L2-normalized vector を返すため、dot product は cosine similarity として扱えます。

```bash
swift run e5-embed-similarity [options] --query <text> --passage <text>
```

### オプション

| Option | Values | Default | Description |
| --- | --- | --- | --- |
| `--query` | text | required | query text です。CLI は `purpose = query` で embedding します。 |
| `--passage` | text | required | 候補テキストです。CLI は `purpose = passage` で embedding します。 |
| `--backend` | `coreml`, `deterministic` | `coreml` | embedding backend を選びます。 |
| `--model` | path | `Models/E5SmallEmbedding.mlpackage` または `.mlmodelc` | Core ML model の path です。 |
| `--tokenizer` | path | `Tokenizer` | tokenizer assets のディレクトリです。 |
| `--max-length` | positive integer | `128` | padding / truncation に使う token sequence length です。 |
| `--model-name` | string | `intfloat/multilingual-e5-small` | JSON output の `model` field を上書きします。 |

### 類似度の例

```bash
swift run e5-embed-similarity \
  --query "車内の収納を増やしたい" \
  --passage "セレナの荷物積載量を増やす方法"
```

### 無関係な passage と比較する

```bash
swift run e5-embed-similarity \
  --query "車内の収納を増やしたい" \
  --passage "量子力学における波動関数の解釈を説明する"
```

## `e5-embed-batch`

テキストファイル内の複数 keyword を embedding し、JSONL を出力します。入力の非空行が1つの `StoredEmbedding` record になります。

```bash
swift run e5-embed-batch [options] --input <keywords.txt> --output <embeddings.jsonl>
```

### オプション

| Option | Values | Default | Description |
| --- | --- | --- | --- |
| `--input` | path or `-` | required | 入力テキストファイルです。1行1 keyword。`-` で stdin を使います。 |
| `--output` | path or `-` | required | JSONL output path です。`-` で stdout を使います。 |
| `--purpose` | `query`, `passage` | `passage` | 全入力行に使う purpose です。keyword catalog は通常 `passage` を使います。 |
| `--backend` | `coreml`, `deterministic` | `coreml` | embedding backend を選びます。 |
| `--model` | path | `Models/E5SmallEmbedding.mlpackage` または `.mlmodelc` | Core ML model の path です。 |
| `--tokenizer` | path | `Tokenizer` | tokenizer assets のディレクトリです。 |
| `--max-length` | positive integer | `128` | padding / truncation に使う token sequence length です。 |
| `--model-name` | string | `intfloat/multilingual-e5-small` | JSONL output の `model` field を上書きします。 |

### 例

`keywords.txt`:

```text
車内収納
ルーフボックス
シートバック収納
量子力学
```

Command:

```bash
swift run e5-embed-batch \
  --input keywords.txt \
  --output embeddings.jsonl
```

出力は1行1 JSON recordです。`embedding` field には384個の値が入りますが、ここでは省略しています。

```json
{"dimension":384,"embedding":[0.0123,-0.0456,"... 残り382個の値"],"id":"1","model":"intfloat/multilingual-e5-small","purpose":"passage","text":"車内収納"}
```

## `e5-keyword-graph`

`e5-embed-batch` の JSONL output から exact top-k graph を作ります。vector 同士は dot product で比較します。変換済み model は L2-normalized vector を返すため、dot product は cosine similarity として扱えます。

```bash
swift run e5-keyword-graph [options] --input <embeddings.jsonl> --output <graph-file>
```

これは ANN ではなく exact search です。依存が増えずシンプルですが、keyword 数が増えると全件比較のコストが大きくなります。

### オプション

| Option | Values | Default | Description |
| --- | --- | --- | --- |
| `--input` | path or `-` | required | `e5-embed-batch` が出した JSONL です。`-` で stdin を使います。 |
| `--output` | path or `-` | required | output file path です。`-` で stdout を使います。 |
| `--format` | `csv`, `dot`, `graphml`, `json` | `csv` | graph output format です。 |
| `--top-k` | positive integer | `10` | edge deduplication 前に各 keyword について残す近傍数です。 |
| `--threshold` | numeric score | `0.0` | edge にする最小 similarity score です。 |

### CSV 例

```bash
swift run e5-keyword-graph \
  --input embeddings.jsonl \
  --output edges.csv \
  --top-k 10 \
  --threshold 0.82
```

CSV columns:

```text
source_id,source_text,target_id,target_text,score
```

### GraphML 例

```bash
swift run e5-keyword-graph \
  --input embeddings.jsonl \
  --output graph.graphml \
  --format graphml \
  --top-k 10 \
  --threshold 0.82
```

GraphML は Gephi や Cytoscape などで開けます。

### DOT / Graphviz 例

```bash
swift run e5-keyword-graph \
  --input embeddings.jsonl \
  --output graph.dot \
  --format dot \
  --top-k 10 \
  --threshold 0.82

sfdp -Tsvg graph.dot -o graph.svg
```

DOT output は Graphviz の command-line rendering 向けです。keyword の関係 graph には `sfdp` または `neato` が使いやすいです。
macOS で `sfdp` がない場合は `brew install graphviz` で Graphviz を入れます。

### JSON 例

```bash
swift run e5-keyword-graph \
  --input embeddings.jsonl \
  --output graph.json \
  --format json \
  --top-k 10 \
  --threshold 0.82
```

JSON output には `nodes` と `edges` が含まれます。

## 出力メモ

`e5-embed` は以下を返します。

```json
{
  "dimension": 384,
  "embedding": [0.0123, -0.0456, "... 残り382個の値"],
  "model": "intfloat/multilingual-e5-small",
  "purpose": "query"
}
```

`embedding` 配列には正確に `dimension` 個の値が入ります。この例では読みやすさのために省略しています。

`e5-embed-similarity` は以下を返します。

```json
{
  "model": "intfloat/multilingual-e5-small",
  "query": "車内の収納を増やしたい",
  "passage": "セレナの荷物積載量を増やす方法",
  "queryDimension": 384,
  "passageDimension": 384,
  "score": 0.851
}
```

score は相対的に比較する値です。単一の絶対しきい値を固定するより、同じ query に対する複数 passage の順位付けとして扱う方が堅実です。
