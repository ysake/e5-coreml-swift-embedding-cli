#!/usr/bin/env python3
"""Convert intfloat/multilingual-e5-small to a Core ML embedding model.

The converted model accepts tokenized E5 inputs:

  input_ids: Int32, shape [1, max_length]
  attention_mask: Int32, shape [1, max_length]

and returns:

  embedding: Float32/Float16, shape [1, 384]

The PyTorch wrapper performs attention-mask-aware mean pooling followed by
L2 normalization, so Swift only needs to tokenize text and run prediction.
FLOAT32 is the default because FLOAT16 models have produced zero vectors in
visionOS Simulator testing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_ID = "intfloat/multilingual-e5-small"
DEFAULT_MAX_LENGTH = 128


class E5EmbeddingWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.to(torch.long)
        attention_mask = attention_mask.to(torch.long)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        mask = mask.to(token_embeddings.dtype)

        pooled = torch.sum(token_embeddings * mask, dim=1)
        denominator = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = pooled / denominator
        return F.normalize(pooled, p=2, dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert multilingual-e5-small to Core ML."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Models/E5SmallEmbedding.mlpackage"),
    )
    parser.add_argument(
        "--tokenizer-output",
        type=Path,
        default=Path("Tokenizer"),
    )
    parser.add_argument(
        "--compute-precision",
        choices=("FLOAT16", "FLOAT32"),
        default="FLOAT32",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run a small PyTorch/Core ML parity check after conversion.",
    )
    return parser.parse_args()


def encoded_example(tokenizer: AutoTokenizer, max_length: int) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        "query: 車内の収納を増やしたい",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    return {
        "input_ids": encoded["input_ids"].to(torch.int32),
        "attention_mask": encoded["attention_mask"].to(torch.int32),
    }


def convert(args: argparse.Namespace) -> None:
    import coremltools as ct

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id)
    model.eval()

    wrapper = E5EmbeddingWrapper(model).eval()
    example = encoded_example(tokenizer, args.max_length)

    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (example["input_ids"], example["attention_mask"]),
            strict=False,
        )

    precision = (
        ct.precision.FLOAT16
        if args.compute_precision == "FLOAT16"
        else ct.precision.FLOAT32
    )

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=precision,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=(1, args.max_length),
                dtype=np.int32,
            ),
            ct.TensorType(
                name="attention_mask",
                shape=(1, args.max_length),
                dtype=np.int32,
            ),
        ],
        outputs=[
            ct.TensorType(name="embedding"),
        ],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(args.output)

    args.tokenizer_output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.tokenizer_output)

    print(f"Saved Core ML model: {args.output}")
    print(f"Saved tokenizer assets: {args.tokenizer_output}")

    if args.validate:
        validate(wrapper, mlmodel, tokenizer, args.max_length)


def validate(
    wrapper: E5EmbeddingWrapper,
    mlmodel: object,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> None:
    example = encoded_example(tokenizer, max_length)

    with torch.no_grad():
        torch_embedding = wrapper(
            example["input_ids"],
            example["attention_mask"],
        ).detach().cpu().numpy()

    coreml_output = mlmodel.predict(
        {
            "input_ids": example["input_ids"].numpy().astype(np.int32),
            "attention_mask": example["attention_mask"].numpy().astype(np.int32),
        }
    )
    coreml_embedding = np.asarray(coreml_output["embedding"])
    cosine = float(
        np.dot(torch_embedding.flatten(), coreml_embedding.flatten())
        / (
            np.linalg.norm(torch_embedding.flatten())
            * np.linalg.norm(coreml_embedding.flatten())
        )
    )

    print(f"PyTorch shape: {torch_embedding.shape}")
    print(f"Core ML shape: {coreml_embedding.shape}")
    print(f"Core ML L2 norm: {np.linalg.norm(coreml_embedding):.6f}")
    print(f"PyTorch/Core ML cosine similarity: {cosine:.6f}")


def main() -> None:
    args = parse_args()
    if args.max_length <= 0:
        raise SystemExit("--max-length must be greater than zero")
    convert(args)


if __name__ == "__main__":
    main()
