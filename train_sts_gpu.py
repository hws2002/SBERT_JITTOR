"""
Train SBERT on STS-style regression tasks (GPU-optimized pipeline).

This script mirrors the structure of ``train_nli_gpu.py`` but swaps the
classification loss for the cosine-similarity regression setup used on
STS Benchmark and related datasets.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable

import numpy as np
import jittor as jt
from jittor import nn
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from model.sbert_model import SBERTJittor
from losses.regression_loss import RegressionLoss
from utils.data_loader import prepare_sts_dataset, collate_sts


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

HF_DIR = "./hf"
MODEL_DIR_MAP = {
    "bert-large-uncased": "hf_bert_large",
    "bert-base-uncased": "hf_bert_base",
}


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


def setup_device(use_cuda: bool):
    if use_cuda and jt.has_cuda:
        jt.flags.use_cuda = 1
        logger.info("Using CUDA")
    else:
        jt.flags.use_cuda = 0
        logger.info("Using CPU")


def setup_wandb(args):
    if not args.wandb:
        return None

    try:
        import wandb

        run_name = args.run_name if args.run_name else f"sts-{args.base_model.split('/')[-1]}-{args.pooling}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": args.base_model,
                "pooling": args.pooling,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "max_length": args.max_length,
                "epochs": args.epochs,
                "train_dataset": args.train_dataset,
                "train_split": args.train_split,
                "eval_dataset": args.eval_dataset,
                "eval_split": args.eval_split,
                "normalize_scores": args.normalize_scores,
            },
        )
        logger.info(f"W&B initialized: {args.wandb_project}/{run_name}")
        return wandb
    except ImportError:
        logger.warning("wandb not installed. Skipping W&B logging.")
        return None


def evaluate_sts(model, dataloader, normalize_scores: bool = False, score_scale: float = 1.0):
    from scipy.stats import pearsonr, spearmanr

    model.eval()
    all_predictions = []
    all_scores = []

    with jt.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating STS", leave=False):
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
            scores_np = np.asarray(batch["scores"])
            if normalize_scores and score_scale != 0:
                scores_np = scores_np / score_scale
            all_scores.extend(scores_np.tolist())

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
    else:
        checkpoint_path = output_dir / f"{name}.pkl"

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


def load_training_checkpoint(model, optimizer, checkpoint_path: str):
    logger.info(f"Loading training checkpoint: {checkpoint_path}")
    checkpoint = jt.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        except Exception as exc:
            logger.warning(f"Failed to load optimizer state: {exc}")

    global_step = int(checkpoint.get("iteration", 0))
    start_epoch = max(int(checkpoint.get("epoch", 1)) - 1, 0)
    logger.info(f"Resuming from step {global_step}, epoch {start_epoch + 1}")
    return global_step, start_epoch


def save_eval_results(results: Dict[str, float], args, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.base_model.replace("/", "_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_sts_{safe_model}_{timestamp}.json"

    payload = {
        "model_info": {
            "type": "sbert_jittor",
            "model_name": safe_model,
            "max_length": args.max_length,
        },
        "evaluation": {
            "dataset": args.test_dataset,
            "split": args.test_split,
            "batch_size": args.eval_batch_size,
            "device": "cuda" if jt.flags.use_cuda else "cpu",
        },
        "results": results,
        "timestamp": timestamp,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    logger.info(f"Saved evaluation results to {output_path}")
    return output_path


def _resolve_tokenizer_source(args) -> str:
    tokenizer_source = args.base_model
    if not os.path.isdir(tokenizer_source):
        mapped = MODEL_DIR_MAP.get(args.base_model, args.base_model)
        candidate = os.path.join(HF_DIR, mapped)
        if os.path.isdir(candidate):
            tokenizer_source = candidate
    if not os.path.isdir(tokenizer_source):
        raise ValueError("Expected local model directory for tokenizer (base_model).")
    return tokenizer_source


def train(args):
    logger.info("=" * 70)
    logger.info("SBERT STS Regression Training (GPU-optimized)")
    logger.info("=" * 70)
    logger.info(f"Model: {args.base_model}")
    logger.info(f"Pooling: {args.pooling}")
    logger.info(f"Train dataset: {args.train_dataset}/{args.train_split}")
    logger.info(f"Eval dataset: {args.eval_dataset}/{args.eval_split}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Eval steps: {args.eval_steps}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info("=" * 70)

    setup_device(args.use_cuda)
    wandb = setup_wandb(args)

    tokenizer_source = _resolve_tokenizer_source(args)
    logger.info(f"Loading tokenizer from: {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, local_files_only=True)

    cache_dir = args.cache_dir or os.path.join(args.data_dir, "_cache")

    logger.info("Preparing STS training data (cached tokenization)...")
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
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_sts,
    )

    logger.info("Preparing STS evaluation data (cached tokenization)...")
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
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sts,
    )

    total_steps = args.epochs * len(train_dataloader)
    if total_steps == 0:
        raise RuntimeError("No training data found. Check data_dir/datasets arguments.")
    logger.info(f"Total training steps: {total_steps}")

    logger.info("Initializing SBERT model...")
    model = SBERTJittor(
        encoder_name=args.base_model,
        pooling=args.pooling,
        head_type="none",
        checkpoint_path=args.encoder_checkpoint,
    )
    logger.info(f"Model embedding dimension: {model.output_dim}")

    optimizer = nn.Adam(model.parameters(), lr=args.lr)
    loss_fn = RegressionLoss()

    global_step = 0
    start_epoch = 0
    if args.start_from_checkpoints:
        global_step, start_epoch = load_training_checkpoint(
            model,
            optimizer,
            args.start_from_checkpoints,
        )

    logger.info("Evaluation before training:")
    eval_results_before = evaluate_sts(
        model=model,
        dataloader=eval_dataloader,
        normalize_scores=args.normalize_scores,
        score_scale=args.score_scale,
    )
    logger.info(
        f"Initial Eval - Pearson: {eval_results_before['pearson']:.2f}, "
        f"Spearman: {eval_results_before['spearman']:.2f}"
    )
    if wandb:
        wandb.log(
            {
                "eval/pearson": eval_results_before["pearson"],
                "eval/spearman": eval_results_before["spearman"],
                "step": global_step,
            }
        )

    total_loss = 0.0
    total_samples = 0
    best_spearman = eval_results_before["spearman"]

    logger.info("\nStarting training...")
    logger.info("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for step, batch in enumerate(epoch_iterator, 1):
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

            batch_size = targets.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            global_step += 1
            avg_loss = total_loss / max(total_samples, 1)
            epoch_iterator.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{avg_loss:.4f}",
                }
            )

            if global_step % args.log_steps == 0:
                logger.info(f"Step {global_step}/{total_steps} | Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f}")
                if wandb:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/avg_loss": avg_loss,
                            "step": global_step,
                        }
                    )

            if global_step % args.eval_steps == 0:
                eval_results = evaluate_sts(
                    model=model,
                    dataloader=eval_dataloader,
                    normalize_scores=args.normalize_scores,
                    score_scale=args.score_scale,
                )
                logger.info(
                    f"Eval - Pearson: {eval_results['pearson']:.2f}, "
                    f"Spearman: {eval_results['spearman']:.2f}"
                )
                if wandb:
                    wandb.log(
                        {
                            "eval/pearson": eval_results["pearson"],
                            "eval/spearman": eval_results["spearman"],
                            "step": global_step,
                        }
                    )
                if eval_results["spearman"] > best_spearman:
                    best_spearman = eval_results["spearman"]
                    save_checkpoint(model, optimizer, global_step, epoch + 1, args, name="best")

        avg_loss = total_loss / max(total_samples, 1)
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs} Summary:")
        logger.info(f"  Average Loss: {avg_loss:.4f}")
        logger.info(f"  Total Steps: {global_step}")

        if wandb:
            wandb.log(
                {
                    "epoch/loss": avg_loss,
                    "epoch": epoch + 1,
                }
            )

        total_loss = 0.0
        total_samples = 0

    logger.info("\n" + "=" * 70)
    logger.info("Final Evaluation on STS Benchmark Test Split")
    logger.info("=" * 70)

    test_dataset = prepare_sts_dataset(
        data_dir=args.data_dir,
        dataset_name=args.test_dataset,
        split=args.test_split,
        tokenizer=tokenizer,
        max_length=args.max_length,
        cache_dir=cache_dir,
        overwrite_cache=args.overwrite_cache,
        tokenize_batch_size=args.tokenize_batch_size,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sts,
    )

    best_path = Path(args.output_dir) / "best.pkl"
    if best_path.is_file():
        logger.info(f"Loading best checkpoint for final evaluation: {best_path}")
        best_state = jt.load(str(best_path))
        model.load_state_dict(best_state["model_state"])

    test_results = evaluate_sts(
        model=model,
        dataloader=test_dataloader,
        normalize_scores=args.normalize_scores,
        score_scale=args.score_scale,
    )
    logger.info(
        f"STS-B Test - Pearson: {test_results['pearson']:.2f}, "
        f"Spearman: {test_results['spearman']:.2f}"
    )
    if wandb:
        wandb.log(
            {
                "test/pearson": test_results["pearson"],
                "test/spearman": test_results["spearman"],
            }
        )

    save_eval_results(test_results, args, Path(args.output_dir) / "result")

    logger.info("\nSaving final model...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    final_path = final_dir / f"sbert_{args.base_model.replace('/', '_')}_{args.pooling}.pkl"

    stats_path = final_dir / "stats.json"
    previous_spearman = None
    if stats_path.exists():
        try:
            with stats_path.open("r", encoding="utf-8") as handle:
                previous_stats = json.load(handle)
                previous_spearman = previous_stats.get("spearman")
        except Exception as exc:
            logger.warning(f"Failed to read existing final stats: {exc}")

    should_save_final = previous_spearman is None or test_results["spearman"] > previous_spearman
    if should_save_final:
        model.save(str(final_path))
        new_stats = {
            "pearson": test_results["pearson"],
            "spearman": test_results["spearman"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with stats_path.open("w", encoding="utf-8") as handle:
            json.dump(new_stats, handle, ensure_ascii=False, indent=2)
        logger.info(f"Final model saved: {final_path}")
    else:
        logger.info(
            "Final model not saved because existing final checkpoint has equal or better Spearman score."
        )

    if wandb:
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SBERT on STS datasets with cached tokenization and DataLoader workers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("base_model", nargs="?", default="bert-base-uncased",
                        help="Base encoder model name (resolved under ./hf if not a path)")
    parser.add_argument("--pooling", default="mean",
                        choices=["mean", "cls", "max"],
                        help="Pooling strategy")
    parser.add_argument("--encoder_checkpoint", type=str, default=None,
                        help="Optional pretrained encoder checkpoint (.bin/.pt)")

    parser.add_argument("--data_dir", default="./data",
                        help="Directory containing datasets")
    parser.add_argument("--cache_dir", type=str, default="./data/tokenized",
                        help="Cache directory for tokenized datasets (default: data_dir/_cache)")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite existing cached datasets")
    parser.add_argument("--tokenize_batch_size", type=int, default=1024,
                        help="Batch size used during dataset tokenization")

    parser.add_argument("--train_dataset", type=str, default="STS-B",
                        help="STS dataset used for training")
    parser.add_argument("--train_split", type=str, default="train",
                        help="Train split (train/validation/test)")
    parser.add_argument("--eval_dataset", type=str, default="STS-B",
                        help="STS dataset used for evaluation during training")
    parser.add_argument("--eval_split", type=str, default="validation",
                        help="Evaluation split (validation/test)")
    parser.add_argument("--test_dataset", type=str, default="STS-B",
                        help="Dataset for final evaluation")
    parser.add_argument("--test_split", type=str, default="test",
                        help="Final evaluation split")

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
    parser.add_argument("--log_steps", type=int, default=20,
                        help="Log metrics every N steps")
    parser.add_argument("--eval_steps", type=int, default=180,
                        help="Evaluate on STS benchmark every N steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every N steps (0 to disable)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes")
    parser.add_argument("--use_cuda", action="store_true",
                        help="Use CUDA if available")
    parser.add_argument("--normalize_scores", action="store_true",
                        help="Normalize STS scores by score_scale before regression loss")
    parser.add_argument("--score_scale", type=float, default=5.0,
                        help="Score scale for normalization (default: 5.0)")

    parser.add_argument("--start_from_checkpoints", type=str, default=None,
                        help="Path to a saved checkpoint to resume training")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for checkpoints and results")

    parser.add_argument("--wandb", action="store_true",
                        help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="sbert-sts-training",
                        help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name (default: auto-generated)")

    args = parser.parse_args()

    if args.output_dir is None:
        model_name = args.base_model.replace("/", "-")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = f"checkpoints/training_sts_{model_name}-{timestamp}"

    return args


if __name__ == "__main__":
    cli_args = parse_args()
    train(cli_args)
