"""
Example: download a Jittor SBERT checkpoint from Hugging Face and run a short MR test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from training.mr.train_mr import train as train_mr


def _find_checkpoint(repo_dir: Path) -> Path:
    candidates = sorted(repo_dir.glob("*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No .pkl checkpoint found in {repo_dir}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True, help="HF repo id, e.g. user/roberta-base-nli-mean-tokens")
    parser.add_argument("--data_dir", default="./data", help="Directory containing MR dataset")
    parser.add_argument("--output_dir", default="./checkpoints/mr_debug", help="Output directory for MR run")
    parser.add_argument("--epochs", type=int, default=1, help="Number of MR epochs (debug)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")
    args = parser.parse_args()

    repo_dir = Path(snapshot_download(repo_id=args.repo_id))
    checkpoint_path = _find_checkpoint(repo_dir)

    train_args = argparse.Namespace(
        base_model=str(repo_dir),
        tokenizer_dir=str(repo_dir),
        encoder_checkpoint=str(checkpoint_path),
        jittor_checkpoint=None,
        pooling="mean",
        data_dir=args.data_dir,
        cache_dir=None,
        overwrite_cache=False,
        tokenize_batch_size=1024,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        epochs=args.epochs,
        lr=2e-5,
        warmup_ratio=0.1,
        max_length=128,
        num_workers=4,
        use_cuda=args.use_cuda,
        num_labels=2,
        train_encoder=False,
        log_steps=100,
        eval_steps=500,
        output_dir=args.output_dir,
        run_name=None,
    )

    train_mr(train_args)


if __name__ == "__main__":
    main()
