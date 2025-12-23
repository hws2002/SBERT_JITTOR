"""
Download Hugging Face tokenizer files for offline use.
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ALLOW_PATTERNS = [
    "tokenizer.json",
    "vocab.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
]


def _parse_args():
    parser = argparse.ArgumentParser(description="Download tokenizer files for offline use.")
    parser.add_argument(
        "--repo_id",
        default="bert-large-uncased",
        help="Hugging Face model repo id (e.g., bert-large-uncased).",
    )
    parser.add_argument(
        "--local_dir",
        default=None,
        help="Destination directory (default: ./hf/tokenizer/<repo_id>).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an existing directory.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    local_dir = args.local_dir or f"./hf/tokenizer/{args.repo_id}"
    target = Path(local_dir)
    if target.exists() and any(target.iterdir()) and not args.force:
        raise SystemExit(
            f"Refusing to overwrite existing directory: {target}. "
            "Use --force to override or pick a new --local_dir."
        )
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=local_dir,
        allow_patterns=ALLOW_PATTERNS,
    )


if __name__ == "__main__":
    main()
