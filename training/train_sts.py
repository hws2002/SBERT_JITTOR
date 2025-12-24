"""
Train SBERT on STS-style regression with cosine-similarity MSE loss.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import jittor as jt
from jittor import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses import RegressionLoss
from model.sbert_model import SBERTJittor
from utils.data_loader import collate_sts, prepare_sts_dataset


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _jt_array(data, dtype: str):
    return jt.array(np.asarray(data, dtype=dtype))


def _to_jittor_batch(batch: Dict[str, Iterable]) -> Dict[str, jt.Var]:
    out: Dict[str, jt.Var] = {
        "input_ids_a": _jt_array(batch["input_ids_a"], "int32"),
        "attention_mask_a": _jt_array(batch["attention_mask_a"], "float32"),
        "input_ids_b": _jt_array(batch["input_ids_b"], "int32"),
        "attention_mask_b": _jt_array(batch["attention_mask_b"], "float32"),
        "scores": _jt_array(batch["scores"], "float32"),
    }

    if "token_type_ids_a" in batch:
        out["token_type_ids_a"] = _jt_array(batch["token_type_ids_a"], "int32")
    if "token_type_ids_b" in batch:
        out["token_type_ids_b"] = _jt_array(batch["token_type_ids_b"], "int32")

    return out


def resolve_tokenizer_source(args) -> str:
    candidate = os.path.join(args.hf_tokenizer_dir, args.base_model)
    if os.path.isdir(candidate):
        logger.info(f"Using local tokenizer: {candidate}")
        return candidate
    return args.base_model


def resolve_encoder_checkpoint(args) -> str | None:
    if args.encoder_checkpoint:
        return args.encoder_checkpoint
    candidate = os.path.join(args.hf_checkpoint_dir, args.base_model, "pytorch_model.bin")
    if os.path.isfile(candidate):
        logger.info(f"Using encoder checkpoint: {candidate}")
        return candidate
    return None


def evaluate(model, dataloader):
    from scipy.stats import pearsonr, spearmanr

    model.eval()
    all_predictions = []
    all_scores = []

    with jt.no_grad():
        for batch in dataloader:
            jt_batch = _to_jittor_batch(batch)
            emb_a = model.encode(
                jt_batch["input_ids_a"],
                jt_batch["attention_mask_a"],
                jt_batch.get("token_type_ids_a", None),
            )
            emb_b = model.encode(
                jt_batch["input_ids_b"],
                jt_batch["attention_mask_b"],
                jt_batch.get("token_type_ids_b", None),
            )

            emb_a_np = emb_a.numpy()
            emb_b_np = emb_b.numpy()
            denom = np.linalg.norm(emb_a_np, axis=1) * np.linalg.norm(emb_b_np, axis=1) + 1e-9
            sim = np.sum(emb_a_np * emb_b_np, axis=1) / denom

            all_predictions.extend(sim.tolist())
            all_scores.extend(np.asarray(batch["scores"]).tolist())

    pearson_corr, _ = pearsonr(all_predictions, all_scores)
    spearman_corr, _ = spearmanr(all_predictions, all_scores)
    model.train()
    return {
        "pearson": pearson_corr * 100,
        "spearman": spearman_corr * 100,
    }


def save_checkpoint(model, optimizer, iteration, epoch, args, name="checkpoint"):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if name == "best":
        checkpoint_path = output_dir / "best.pkl"
    elif name == "checkpoint":
        checkpoint_path = output_dir / "checkpoint_latest.pkl"
    else:
        checkpoint_path = output_dir / f"{name}_step{iteration}.pkl"

    checkpoint = {
        "iteration": iteration,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "base_model": args.base_model,
        "pooling": args.pooling,
    }

    jt.save(checkpoint, str(checkpoint_path))
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def train(args):
    logger.info("=" * 70)
    logger.info("SBERT STS Regression Training")
    logger.info("=" * 70)
    logger.info(f"Model: {args.base_model}")
    logger.info(f"Pooling: {args.pooling}")
    logger.info(f"Train dataset: {args.train_dataset} ({args.train_split})")
    logger.info(f"Eval dataset: {args.eval_dataset} ({args.eval_split})")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Eval steps: {args.eval_steps}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info("=" * 70)

    if args.use_cuda and jt.has_cuda:
        jt.flags.use_cuda = 1
        logger.info("Using CUDA")
    else:
        jt.flags.use_cuda = 0
        logger.info("Using CPU")

    tokenizer_source = resolve_tokenizer_source(args)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    cache_dir = args.cache_dir or os.path.join(args.data_dir, "_cache")

    train_dataset = prepare_sts_dataset(
        data_dir=args.data_dir,
        dataset_name=args.train_dataset,
        split=args.train_split,
        tokenizer=tokenizer,
        max_length=args.max_length,
        cache_dir=cache_dir,
        overwrite_cache=args.overwrite_cache,
        tokenize_batch_size=args.tokenize_batch_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_sts,
    )

    eval_dataset = prepare_sts_dataset(
        data_dir=args.data_dir,
        dataset_name=args.eval_dataset,
        split=args.eval_split,
        tokenizer=tokenizer,
        max_length=args.max_length,
        cache_dir=cache_dir,
        overwrite_cache=args.overwrite_cache,
        tokenize_batch_size=args.tokenize_batch_size,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sts,
    )

    model = SBERTJittor(
        encoder_name=args.base_model,
        pooling=args.pooling,
        head_type="none",
        checkpoint_path=resolve_encoder_checkpoint(args),
    )
    logger.info(f"Model embedding dimension: {model.output_dim}")

    optimizer = nn.Adam(model.parameters(), lr=args.lr)
    loss_fn = RegressionLoss()

    total_steps = args.epochs * len(train_loader)
    if total_steps == 0:
        raise RuntimeError("No training data found. Check data_dir/datasets arguments.")
    logger.info(f"Total training steps: {total_steps}")

    global_step = 0
    best_spearman = -1.0

    for epoch in range(args.epochs):
        for batch in train_loader:
            jt_batch = _to_jittor_batch(batch)
            emb_a = model.encode(
                jt_batch["input_ids_a"],
                jt_batch["attention_mask_a"],
                jt_batch.get("token_type_ids_a", None),
            )
            emb_b = model.encode(
                jt_batch["input_ids_b"],
                jt_batch["attention_mask_b"],
                jt_batch.get("token_type_ids_b", None),
            )

            targets = jt_batch["scores"]
            if args.normalize_scores:
                targets = targets / args.score_scale
            loss = loss_fn(emb_a, emb_b, targets)
            optimizer.step(loss)

            global_step += 1
            if global_step % args.log_steps == 0:
                logger.info(f"Step {global_step}/{total_steps} | Loss: {loss.item():.4f}")

            if global_step % args.eval_steps == 0:
                scores = evaluate(model, eval_loader)
                logger.info(
                    f"Eval - Pearson: {scores['pearson']:.2f}, Spearman: {scores['spearman']:.2f}"
                )
                if scores["spearman"] > best_spearman:
                    best_spearman = scores["spearman"]
                    save_checkpoint(model, optimizer, global_step, epoch + 1, args, name="best")

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_checkpoint(model, optimizer, global_step, epoch + 1, args, name="checkpoint")

    logger.info("Training completed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SBERT on STS regression tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("base_model", nargs="?", default="bert-base-uncased",
                        help="Base encoder model (bert-base-uncased, bert-large-uncased, roberta-base, roberta-large)")
    parser.add_argument("--pooling", default="mean",
                        choices=["mean", "cls", "max"],
                        help="Pooling strategy")
    parser.add_argument("--encoder_checkpoint", type=str, default=None,
                        help="Optional pretrained encoder checkpoint (.bin/.pt)")
    parser.add_argument("--hf_checkpoint_dir", type=str, default="./hf/pretrained_bert_checkpoints",
                        help="Base directory containing pretrained HF checkpoints")
    parser.add_argument("--hf_tokenizer_dir", type=str, default="./hf/tokenizer",
                        help="Base directory containing local tokenizers")

    parser.add_argument("--data_dir", default="./data",
                        help="Directory containing datasets")
    parser.add_argument("--cache_dir", type=str, default="./data/tokenized",
                        help="Cache directory for tokenized datasets")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite existing cached datasets")
    parser.add_argument("--tokenize_batch_size", type=int, default=1024,
                        help="Batch size used during dataset tokenization")

    parser.add_argument("--train_dataset", type=str, default="STS-B",
                        help="Dataset used for training")
    parser.add_argument("--train_split", type=str, default="train",
                        help="Split used for training (STS-B supports train/validation/test)")
    parser.add_argument("--eval_dataset", type=str, default="STS-B",
                        help="Dataset used for evaluation")
    parser.add_argument("--eval_split", type=str, default="validation",
                        help="Split used for evaluation (STS-B supports validation/test)")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Evaluation batch size")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--log_steps", type=int, default=100,
                        help="Log metrics every N steps")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every N steps (0 to disable)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes")
    parser.add_argument("--use_cuda", action="store_true",
                        help="Use CUDA if available")
    parser.add_argument("--normalize_scores", action=argparse.BooleanOptionalAction, default=True,
                        help="Normalize STS scores by score_scale before MSE loss")
    parser.add_argument("--score_scale", type=float, default=5.0,
                        help="Scale factor for STS scores (default: 5.0)")

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for checkpoints")

    args = parser.parse_args()

    if args.output_dir is None:
        model_name = args.base_model.replace("/", "-")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = f"checkpoints/training_sts_{model_name}-{timestamp}"

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
