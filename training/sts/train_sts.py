"""
Train SBERT on STS-style regression tasks.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
import jittor as jt
from jittor import nn
from tqdm import tqdm
from transformers import AutoTokenizer
from jittor.dataset import DataLoader

from model.sbert_model import SBERTJittor
from losses.regression_loss import RegressionLoss
from utils.data_loader import prepare_sts_dataset, collate_sts
from utils.jittor_batch import _to_jittor_batch
from utils.jittor_utils import setup_device
from utils.checkpoint_utils import (
    load_training_checkpoint_with_optimizer as load_training_checkpoint,
    save_checkpoint_with_optimizer as save_checkpoint,
)
from utils.training_utils import (
    TrainConfig,
    checkpoint_path,
)
from evaluation.sts_eval_utils import evaluate_sts, save_eval_results_single


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)



def train(args):
    suffix = "sts-nli" if args.start_from_checkpoints else "sts-only"
    config = TrainConfig.from_args(args, "training_sts", output_suffix=suffix)
    args.output_dir = config.checkpoint_output_path
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
    tokenizer_source = config.tokenizer_path
    logger.info(f"Loading tokenizer from: {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    cache_dir = config.cache_dir

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
        checkpoint_path=config.pretrained_checkpoint_path,
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
                if not args.disable_checkpoint and eval_results["spearman"] > best_spearman:
                    best_spearman = eval_results["spearman"]
                    save_checkpoint(model, optimizer, global_step, epoch + 1, args, name="best")

        avg_loss = total_loss / max(total_samples, 1)
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs} Summary:")
        logger.info(f"  Average Loss: {avg_loss:.4f}")
        logger.info(f"  Total Steps: {global_step}")

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

    best_path = checkpoint_path(args.output_dir, args.base_model, "best")
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

    save_eval_results_single(test_results, args, Path(args.output_dir) / "result")

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
    parser.add_argument("--tokenizer_dir", type=str, default=None,
                        help="Tokenizer directory (overrides base_model lookup)")

    parser.add_argument("--data_dir", default="./data",
                        help="Directory containing datasets")
    parser.add_argument("--cache_dir", type=str, default=None,
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
    parser.add_argument("--disable_checkpoint", action="store_true",
                        help="Disable saving best checkpoints")
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

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    cli_args = parse_args()
    train(cli_args)
