# CLI Usage

This document describes the command-line tools provided by this package.

## Prerequisites

Generate the local Core ML model and tokenizer assets before using the default Core ML backend.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements-convert.txt
python scripts/convert_e5_small_to_coreml.py --validate
```

The conversion script defaults to `FLOAT32`, which is the recommended starting point for visionOS apps. Use `--compute-precision FLOAT16` only when explicitly testing macOS or device-specific behavior.

Expected local assets:

```text
Models/E5SmallEmbedding.mlpackage
Tokenizer/
```

The generated assets are ignored by git.

## `e5-embed`

Generates one embedding vector and prints JSON.

```bash
swift run e5-embed [options] <text>
```

### Options

| Option | Values | Default | Description |
| --- | --- | --- | --- |
| `--purpose` | `query`, `passage` | `query` | Applies the E5 input prefix. Use `query` for search queries and `passage` for documents or candidate text. |
| `--backend` | `coreml`, `deterministic` | `coreml` | Selects the embedding backend. `deterministic` is only for development smoke tests. |
| `--model` | path | `Models/E5SmallEmbedding.mlpackage` or `.mlmodelc` | Core ML model path. |
| `--tokenizer` | path | `Tokenizer` | Directory containing tokenizer assets. |
| `--max-length` | positive integer | `128` | Token sequence length used for padding and truncation. It must match the converted Core ML model. |
| `--model-name` | string | `intfloat/multilingual-e5-small` | Overrides the `model` field in JSON output. |

### Query Embedding

```bash
swift run e5-embed "Find more storage space inside my car"
```

Equivalent:

```bash
swift run e5-embed --purpose query "Find more storage space inside my car"
```

### Passage Embedding

```bash
swift run e5-embed \
  --purpose passage \
  "Use seat-back organizers, cargo boxes, or roof storage to increase vehicle storage capacity."
```

### Custom Asset Paths

```bash
swift run e5-embed \
  --model Models/E5SmallEmbedding.mlpackage \
  --tokenizer Tokenizer \
  --max-length 128 \
  "Find more storage space inside my car"
```

### Development Smoke Test

Use this when model assets are not present. The vector is deterministic but not semantically meaningful.

```bash
swift run e5-embed --backend deterministic "Smoke test"
```

## `e5-embed-similarity`

Embeds a query and a passage, then prints their dot product as JSON. Because the converted E5 model returns L2-normalized vectors, the dot product is cosine similarity.

```bash
swift run e5-embed-similarity [options] --query <text> --passage <text>
```

### Options

| Option | Values | Default | Description |
| --- | --- | --- | --- |
| `--query` | text | required | Query text. The CLI embeds it with `purpose = query`. |
| `--passage` | text | required | Candidate text. The CLI embeds it with `purpose = passage`. |
| `--backend` | `coreml`, `deterministic` | `coreml` | Selects the embedding backend. |
| `--model` | path | `Models/E5SmallEmbedding.mlpackage` or `.mlmodelc` | Core ML model path. |
| `--tokenizer` | path | `Tokenizer` | Directory containing tokenizer assets. |
| `--max-length` | positive integer | `128` | Token sequence length used for padding and truncation. |
| `--model-name` | string | `intfloat/multilingual-e5-small` | Overrides the `model` field in JSON output. |

### Similarity Example

```bash
swift run e5-embed-similarity \
  --query "Find more storage space inside my car" \
  --passage "Cargo organizers and roof boxes can increase available storage in a vehicle."
```

### Compare Against an Unrelated Passage

```bash
swift run e5-embed-similarity \
  --query "Find more storage space inside my car" \
  --passage "Explain the interpretation of wave functions in quantum mechanics."
```

## `e5-embed-batch`

Embeds many keywords from a text file and writes JSONL. Each non-empty input line becomes one `StoredEmbedding` record.

```bash
swift run e5-embed-batch [options] --input <keywords.txt> --output <embeddings.jsonl>
```

### Options

| Option | Values | Default | Description |
| --- | --- | --- | --- |
| `--input` | path or `-` | required | Input text file. One keyword per line. Use `-` for stdin. |
| `--output` | path or `-` | required | JSONL output path. Use `-` for stdout. |
| `--purpose` | `query`, `passage` | `passage` | Purpose used for every input line. Keyword catalogs usually use `passage`. |
| `--backend` | `coreml`, `deterministic` | `coreml` | Selects the embedding backend. |
| `--model` | path | `Models/E5SmallEmbedding.mlpackage` or `.mlmodelc` | Core ML model path. |
| `--tokenizer` | path | `Tokenizer` | Directory containing tokenizer assets. |
| `--max-length` | positive integer | `128` | Token sequence length used for padding and truncation. |
| `--model-name` | string | `intfloat/multilingual-e5-small` | Overrides the `model` field in JSONL output. |

### Example

`keywords.txt`:

```text
car storage
roof box
seat organizer
quantum mechanics
```

Command:

```bash
swift run e5-embed-batch \
  --input keywords.txt \
  --output embeddings.jsonl
```

Each output line is one JSON record. The `embedding` field contains 384 values and is shown truncated here.

```json
{"dimension":384,"embedding":[0.0123,-0.0456,"... 382 more values"],"id":"1","model":"intfloat/multilingual-e5-small","purpose":"passage","text":"car storage"}
```

## `e5-keyword-graph`

Builds an exact top-k graph from `e5-embed-batch` JSONL output. It compares vectors with dot product. Because the converted model returns L2-normalized vectors, the dot product is cosine similarity.

```bash
swift run e5-keyword-graph [options] --input <embeddings.jsonl> --output <graph-file>
```

This is exact search, not ANN. It is simple and dependency-free, but pairwise comparison cost grows quickly as keyword count increases.

### Options

| Option | Values | Default | Description |
| --- | --- | --- | --- |
| `--input` | path or `-` | required | JSONL produced by `e5-embed-batch`. Use `-` for stdin. |
| `--output` | path or `-` | required | Output file path. Use `-` for stdout. |
| `--format` | `csv`, `dot`, `graphml`, `json` | `csv` | Graph output format. |
| `--top-k` | positive integer | `10` | Number of nearest neighbors retained per keyword before edge deduplication. |
| `--threshold` | numeric score | `0.0` | Minimum similarity score for an edge. |

### CSV Example

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

### GraphML Example

```bash
swift run e5-keyword-graph \
  --input embeddings.jsonl \
  --output graph.graphml \
  --format graphml \
  --top-k 10 \
  --threshold 0.82
```

GraphML can be opened by tools such as Gephi and Cytoscape.

### DOT / Graphviz Example

```bash
swift run e5-keyword-graph \
  --input embeddings.jsonl \
  --output graph.dot \
  --format dot \
  --top-k 10 \
  --threshold 0.82

sfdp -Tsvg graph.dot -o graph.svg
```

DOT output is intended for Graphviz command-line rendering. Use `sfdp` or `neato` for keyword relationship graphs.
On macOS, install Graphviz with `brew install graphviz` if `sfdp` is not available.

### JSON Example

```bash
swift run e5-keyword-graph \
  --input embeddings.jsonl \
  --output graph.json \
  --format json \
  --top-k 10 \
  --threshold 0.82
```

The JSON output contains `nodes` and `edges`.

## Output Notes

`e5-embed` returns:

```json
{
  "dimension": 384,
  "embedding": [0.0123, -0.0456, "... 382 more values"],
  "model": "intfloat/multilingual-e5-small",
  "purpose": "query"
}
```

The `embedding` array contains exactly `dimension` values. The example is truncated for readability.

`e5-embed-similarity` returns:

```json
{
  "model": "intfloat/multilingual-e5-small",
  "query": "Find more storage space inside my car",
  "passage": "Cargo organizers and roof boxes can increase available storage in a vehicle.",
  "queryDimension": 384,
  "passageDimension": 384,
  "score": 0.851
}
```

Scores are relative. Compare multiple passages for the same query instead of treating one absolute threshold as universal.
