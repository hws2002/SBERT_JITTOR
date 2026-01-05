"""
Train SBERT on NLI datasets (SNLI + MultiNLI) with evaluation on STS benchmark

Inspired by: https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/nli/training_nli.py
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import jittor as jt
from jittor import nn
from tqdm import tqdm
from transformers import AutoTokenizer
from jittor.dataset import DataLoader

from model.sbert_model import SBERTJittor
from losses.softmax_loss import SoftmaxLoss
from losses.complex_softmax_loss import ComplexSoftmaxLoss
from utils.data_loader import prepare_nli_dataset, prepare_sts_dataset, collate_nli, collate_sts
from utils.jittor_batch import _to_jittor_batch
from utils.jittor_utils import setup_device
from utils.checkpoint_utils import (
    load_training_checkpoint_encoder_only as load_training_checkpoint,
    save_checkpoint_encoder_only as save_checkpoint,
    save_encoder_only,
)
from utils.training_utils import (
    TrainConfig,
    checkpoint_path,
    safe_model_name,
)
from evaluation.sts_eval_utils import evaluate_sts, evaluate_sts_all, save_eval_results_multi

# Set the log level to INFO
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def _filtered_loss_state(train_loss):
    loss_state = train_loss.state_dict()
    return {k: v for k, v in loss_state.items() if not k.startswith("model.")}


def train(args):
    config = TrainConfig.from_args(args, "training_nli")
    args.output_dir = config.checkpoint_output_path
    logger.info("=" * 70)
    logger.info("SBERT NLI Training (GPU-optimized)")
    logger.info("=" * 70)
    logger.info(f"Model: {args.base_model}")
    logger.info(f"Pooling: {args.pooling}")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Eval steps: {args.eval_steps}")
    logger.info(f"Num workers: {args.num_workers}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info("=" * 70)

    setup_device(args.use_cuda)
    tokenizer_source = config.tokenizer_path
    logger.info(f"Loading tokenizer from: {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    cache_dir = config.cache_dir

    logger.info("Preparing NLI training data (cached tokenization)...")
    train_dataset = prepare_nli_dataset(
        data_dir=args.data_dir,
        datasets=args.datasets,
        split="train",
        tokenizer=tokenizer,
        max_length=args.max_length,
        cache_dir=cache_dir,
        overwrite_cache=args.overwrite_cache,
        tokenize_batch_size=args.tokenize_batch_size
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_nli
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

    if args.loss == "complex":
        train_loss = ComplexSoftmaxLoss(
            model=model,
            num_labels=args.num_labels,
            ablation=args.ablation,
        )
    else:
        train_loss = SoftmaxLoss(
            model=model,
            num_labels=args.num_labels,
            ablation=args.ablation,
        )
    optimizer = jt.optim.AdamW(
        train_loss.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    warmup_steps = max(int(total_steps * args.warmup_ratio), 1)
    logger.info(f"Warmup steps: {warmup_steps}")

    logger.info("\nPreparing STS-B data (cached tokenization)...")
    sts_dataset = prepare_sts_dataset(
        data_dir=args.data_dir,
        dataset_name="STS-B",
        split="validation",
        tokenizer=tokenizer,
        max_length=args.max_length,
        cache_dir=cache_dir,
        overwrite_cache=args.overwrite_cache,
        tokenize_batch_size=args.tokenize_batch_size
    )
    sts_dataloader = DataLoader(
        sts_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sts
    )

    logger.info("\nEvaluation before training:")
    eval_results_before = evaluate_sts(model=model, dataloader=sts_dataloader)
    logger.info("\nStarting training...")
    logger.info("=" * 70)

    global_step = 0
    start_epoch = 0
    if args.start_from_checkpoints:
        global_step, start_epoch = load_training_checkpoint(
            model,
            train_loss,
            optimizer,
            args.start_from_checkpoints,
        )
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    best_spearman = eval_results_before["spearman"]

    model.train()
    train_loss.train()

    for epoch in range(start_epoch, args.epochs):
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(epoch_iterator, 1):
            jt_batch = _to_jittor_batch(batch, for_sts=False)
            labels = jt_batch["labels"]

            loss, logits = train_loss(jt_batch, labels)

            optimizer.step(loss)

            global_step += 1
            if global_step <= warmup_steps:
                lr_scale = global_step / warmup_steps
                current_lr = args.lr * lr_scale
                optimizer.lr = current_lr
            else:
                current_lr = args.lr

            predictions = jt.argmax(logits, dim=1)[0]
            correct = jt.sum(predictions == labels).item()
            batch_size = labels.shape[0]

            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += batch_size

            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples * 100
            epoch_iterator.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "acc": f"{accuracy:.2f}%",
                "lr": f"{current_lr:.2e}"
            })

            if global_step % args.log_steps == 0:
                logger.info(
                    f"Step {global_step}/{total_steps} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Acc: {accuracy:.2f}% | "
                    f"LR: {current_lr:.2e}"
                )

            if global_step % args.eval_steps == 0:
                eval_results = evaluate_sts(model=model, dataloader=sts_dataloader)

                if eval_results["spearman"] > best_spearman:
                    best_spearman = eval_results["spearman"]
                    save_checkpoint(model, train_loss, optimizer, global_step, epoch + 1, args, name="best")

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_checkpoint(model, train_loss, optimizer, global_step, epoch + 1, args, name="checkpoint")

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100

        logger.info(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        logger.info(f"  Average Loss: {avg_loss:.4f}")
        logger.info(f"  Accuracy: {accuracy:.2f}%")
        logger.info(f"  Total Steps: {global_step}")

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

    logger.info("\n" + "=" * 70)
    logger.info("Final Evaluation on STS Benchmark Test Set")
    logger.info("=" * 70)

    sts_test_dataset = prepare_sts_dataset(
        data_dir=args.data_dir,
        dataset_name="STS-B",
        split="test",
        tokenizer=tokenizer,
        max_length=args.max_length,
        cache_dir=cache_dir,
        overwrite_cache=args.overwrite_cache,
        tokenize_batch_size=args.tokenize_batch_size
    )
    sts_test_dataloader = DataLoader(
        sts_test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sts
    )

    safe_model = safe_model_name(args.base_model)
    best_path = checkpoint_path(args.output_dir, args.base_model, "best")
    if best_path.is_file():
        logger.info(f"Loading best checkpoint for final test: {best_path}")
        best_state = jt.load(str(best_path))
        model.load_state_dict(best_state["model_state"])

    test_results = evaluate_sts(model=model, dataloader=sts_test_dataloader)

    logger.info("\n" + "=" * 70)
    logger.info("Training completed!")
    logger.info(f"Best Spearman score (dev): {best_spearman:.2f}")
    logger.info(f"Final Spearman score (test): {test_results['spearman']:.2f}")
    logger.info("=" * 70)

    logger.info("\n" + "=" * 70)
    logger.info("Final Evaluation on STS/SICKR Test Sets (Best Checkpoint)")
    logger.info("=" * 70)

    best_path = checkpoint_path(args.output_dir, args.base_model, "best")
    if best_path.is_file():
        logger.info(f"Loading best checkpoint for evaluation: {best_path}")
        best_state = jt.load(str(best_path))
        model.load_state_dict(best_state["model_state"])

    all_results = evaluate_sts_all(
        model=model,
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        cache_dir=cache_dir,
        max_length=args.max_length,
        overwrite_cache=args.overwrite_cache,
        tokenize_batch_size=args.tokenize_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    payload, output_path = save_eval_results_multi(all_results, args, Path(args.output_dir) / "result")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SBERT on NLI datasets with cached tokenization and DataLoader workers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("base_model", nargs="?", default="bert-base-uncased",
                        help="Base encoder model name (resolved under ./hf if not a path)")
    parser.add_argument("--pooling", default="mean",
                        choices=["mean", "cls", "max"],
                        help="Pooling strategy")
    parser.add_argument("--loss", default="softmax",
                        choices=["softmax", "complex"],
                        help="Loss function for NLI training")
    parser.add_argument("--ablation", type=int, default=0,
                        choices=[0, 1, 2, 3, 4, 5, 6],
                        help="Ablation feature set: 0=[u;v;|u-v|], 1=[u;v], 2=[|u-v|], 3=[u*v], 4=[|u-v|;u*v], 5=[u;v;u*v], 6=[u;v;|u-v|;u*v]")
    parser.add_argument("--encoder_checkpoint", type=str, default=None,
                        help="Optional pretrained encoder checkpoint (.bin/.pt)")
    parser.add_argument("--tokenizer_dir", type=str, default=None,
                        help="Tokenizer directory (overrides base_model lookup)")
    parser.add_argument("--num_labels", type=int, default=3,
                        help="Number of NLI classification labels")

    parser.add_argument("--data_dir", default="./data",
                        help="Directory containing datasets")
    parser.add_argument("--datasets", nargs="+", default=["SNLI", "MultiNLI"],
                        help="NLI datasets to use")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")

    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Evaluation batch size for STS")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio (fraction of total steps)")

    parser.add_argument("--use_cuda", action="store_true",
                        help="Use CUDA if available")

    parser.add_argument("--log_steps", type=int, default=100,
                        help="Log metrics every N steps")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Evaluate on STS benchmark every N steps")
    parser.add_argument("--save_steps", type=int, default=5000,
                        help="Save checkpoint every N steps (0 to disable)")
    parser.add_argument("--start_from_checkpoints", type=str, default=None,
                        help="Path to a saved checkpoint to resume training")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for checkpoints")

    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for tokenized datasets (default: data_dir/_cache)")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite existing cached datasets")
    parser.add_argument("--tokenize_batch_size", type=int, default=1024,
                        help="Batch size used during dataset tokenization")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
