"""
Utility script to download HuggingFace checkpoints (model + tokenizer)
and store them in a local directory.
"""

import argparse
from pathlib import Path

from transformers import AutoModel, AutoTokenizer
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Download HuggingFace checkpoints")
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace model identifier (e.g., bert-base-uncased, bert-large-uncased)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store the downloaded checkpoint",
    )
    parser.add_argument(
        "--safe-serialization",
        action="store_true",
        help="Save weights in safetensors format",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="If set, do not download the tokenizer files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = args.output_dir or f"./hf/pretrained_bert_checkpoints/{args.model_name}"
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model: {args.model_name}")
    model = AutoModel.from_pretrained(args.model_name)
    model.save_pretrained(
        output_path,
        safe_serialization=args.safe_serialization,
    )

    if not args.skip_tokenizer:
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        tokenizer.save_pretrained(output_path)

    with torch.no_grad():
        preview = model.embeddings.word_embeddings.weight[0, :5]
        print("Embedding preview (first token, first 5 dims):")
        print(preview)

    print(f"Checkpoint saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
