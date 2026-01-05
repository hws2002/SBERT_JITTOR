"""
Train SBERTJittor on SST-2 classification.

Mirrors train_mr.py style (warmup, eval, no checkpoint saving).
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import jittor as jt
from jittor import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.sbert_model import SBERTJittor
from utils.jt_utils import _to_jittor_batch_single
from utils.training_utils import TrainConfig
from utils.jt_utils import setup_device

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)



def _cache_path(cache_dir: str, split: str, model_name: str, max_length: int) -> str:
    model_id = Path(model_name).name.replace("/", "_")
    name = f"SST-2_{split}_{model_id}_len{max_length}"
    return os.path.join(cache_dir, name)


def prepare_sst_dataset(
    data_dir: str,
    split: str,
    tokenizer,
    max_length: int,
    cache_dir: str,
    overwrite_cache: bool,
    tokenize_batch_size: int,
):
    from datasets import load_from_disk

    data_path = os.path.join(data_dir, "SST-2")
    raw_ds = load_from_disk(data_path)[split]

    cache_path = _cache_path(cache_dir, split, tokenizer.name_or_path, max_length)
    if os.path.isdir(cache_path) and not overwrite_cache:
        logger.info(f"Loading cached SST-2 dataset: {cache_path}")
        return load_from_disk(cache_path)

    def tokenize_fn(batch):
        tok = tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        tok["labels"] = batch["label"]
        return tok

    tokenized = raw_ds.map(
        tokenize_fn,
        batched=True,
        batch_size=tokenize_batch_size,
        remove_columns=raw_ds.column_names,
        desc=f"Tokenizing SST-2/{split}",
    )
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(cache_path)
    return tokenized


def collate_sst(batch: List[Dict]) -> Dict[str, np.ndarray]:
    out = {
        "input_ids": np.asarray([b["input_ids"] for b in batch], dtype=np.int32),
        "attention_mask": np.asarray([b["attention_mask"] for b in batch], dtype=np.float32),
        "labels": np.asarray([b["labels"] for b in batch], dtype=np.int32),
    }
    if "token_type_ids" in batch[0]:
        out["token_type_ids"] = np.asarray([b["token_type_ids"] for b in batch], dtype=np.int32)
    return out


def evaluate(model, classifier, dataloader) -> Dict[str, float]:
    model.eval()
    classifier.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    loss_fct = nn.CrossEntropyLoss()

    with jt.no_grad():
        for batch in dataloader:
            jt_batch = _to_jittor_batch_single(batch)
            reps = model.encode(
                jt_batch["input_ids"],
                jt_batch["attention_mask"],
                jt_batch.get("token_type_ids", None),
            )
            logits = classifier(reps)
            loss = loss_fct(logits, jt_batch["labels"])
            preds = jt.argmax(logits, dim=1)[0]
            total_correct += jt.sum(preds == jt_batch["labels"]).item()
            total_samples += jt_batch["labels"].shape[0]
            total_loss += loss.item() * jt_batch["labels"].shape[0]

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1) * 100
    model.train()
    classifier.train()
    return {"loss": avg_loss, "accuracy": acc}


def train(args):
    config = TrainConfig.from_args(args, "training_sst")
    logger.info("=" * 70)
    logger.info("SBERT SST-2 Training")
    logger.info("=" * 70)
    logger.info(f"Model: {args.base_model}")
    logger.info(f"Pooling: {args.pooling}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info("Checkpoint saving disabled (train-only run).")
    logger.info("=" * 70)

    setup_device(args.use_cuda)
    tokenizer_source = config.tokenizer_path
    logger.info(f"Loading tokenizer from: {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    cache_dir = config.cache_dir

    logger.info("Preparing SST-2 training data (cached tokenization)...")
    train_dataset = prepare_sst_dataset(
        data_dir=args.data_dir,
        split="train",
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
        collate_fn=collate_sst,
    )

    logger.info("Preparing SST-2 validation data...")
    dev_dataset = prepare_sst_dataset(
        data_dir=args.data_dir,
        split="validation",
        tokenizer=tokenizer,
        max_length=args.max_length,
        cache_dir=cache_dir,
        overwrite_cache=args.overwrite_cache,
        tokenize_batch_size=args.tokenize_batch_size,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sst,
    )

    total_steps = args.epochs * len(train_loader)
    if total_steps == 0:
        raise RuntimeError("No training data found. Check data_dir arguments.")
    logger.info(f"Total training steps: {total_steps}")

    model = SBERTJittor(
        encoder_name=args.base_model,
        pooling=args.pooling,
        head_type="none",
        checkpoint_path=config.pretrained_checkpoint_path,
    )
    if args.jittor_checkpoint:
        payload = jt.load(args.jittor_checkpoint)
        if isinstance(payload, dict) and "model_state" in payload:
            model.load_state_dict(payload["model_state"])
        else:
            model.load_state_dict(payload)

    classifier = nn.Linear(model.output_dim, args.num_labels)
    if args.train_encoder:
        optimizer_params = list(model.parameters()) + list(classifier.parameters())
    else:
        optimizer_params = list(classifier.parameters())
    optimizer = nn.Adam(optimizer_params, lr=args.lr)
    loss_fct = nn.CrossEntropyLoss()
    warmup_steps = max(int(total_steps * args.warmup_ratio), 1)
    logger.info(f"Warmup steps: {warmup_steps}")

    global_step = 0
    start_epoch = 0

    logger.info("Evaluation before training:")
    eval_before = evaluate(model, classifier, dev_loader)
    logger.info(
        f"Initial Eval - Loss: {eval_before['loss']:.4f}, Acc: {eval_before['accuracy']:.2f}"
    )

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    best_acc = eval_before["accuracy"]

    logger.info("\nStarting training...")
    logger.info("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        for step, batch in enumerate(train_loader, 1):
            jt_batch = _to_jittor_batch_single(batch)
            if args.train_encoder:
                reps = model.encode(
                    jt_batch["input_ids"],
                    jt_batch["attention_mask"],
                    jt_batch.get("token_type_ids", None),
                )
            else:
                with jt.no_grad():
                    reps = model.encode(
                        jt_batch["input_ids"],
                        jt_batch["attention_mask"],
                        jt_batch.get("token_type_ids", None),
                    )
            logits = classifier(reps)
            loss = loss_fct(logits, jt_batch["labels"])
            optimizer.step(loss)

            global_step += 1
            if global_step <= warmup_steps:
                lr_scale = global_step / warmup_steps
                optimizer.lr = args.lr * lr_scale

            preds = jt.argmax(logits, dim=1)[0]
            correct = jt.sum(preds == jt_batch["labels"]).item()
            batch_size = jt_batch["labels"].shape[0]

            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += batch_size

            avg_loss = total_loss / total_samples
            acc = total_correct / total_samples * 100

            if global_step % args.log_steps == 0:
                logger.info(
                    f"Step {global_step}/{total_steps} | Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | "
                    f"LR: {optimizer.lr:.2e}"
                )

            if global_step % args.eval_steps == 0:
                eval_scores = evaluate(model, classifier, dev_loader)
                logger.info(
                    f"Eval - Loss: {eval_scores['loss']:.4f}, "
                    f"Acc: {eval_scores['accuracy']:.2f}"
                )
                if eval_scores["accuracy"] > best_acc:
                    best_acc = eval_scores["accuracy"]

        logger.info(
            f"\nEpoch {epoch + 1}/{args.epochs} Summary: "
            f"Loss {avg_loss:.4f}, Acc {acc:.2f}%"
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

    logger.info("\nFinal Evaluation on SST-2 Validation Split (as test)")
    test_dataset = prepare_sst_dataset(
        data_dir=args.data_dir,
        split="validation",
        tokenizer=tokenizer,
        max_length=args.max_length,
        cache_dir=cache_dir,
        overwrite_cache=args.overwrite_cache,
        tokenize_batch_size=args.tokenize_batch_size,
    )
    if "labels" in test_dataset.column_names:
        before_count = len(test_dataset)
        test_dataset = test_dataset.filter(
            lambda ex: ex["labels"] >= 0,
            desc="Filtering SST-2 test labels",
        )
        after_count = len(test_dataset)
        removed = before_count - after_count
        logger.info(f"SST-2 test labels filtered: kept {after_count}, removed {removed}.")
    if "labels" not in test_dataset.column_names or len(test_dataset) == 0:
        logger.warning("SST-2 validation split has no labeled samples; skipping final eval.")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sst,
    )

    test_scores = evaluate(model, classifier, test_loader)
    if np.isnan(test_scores["loss"]) or np.isnan(test_scores["accuracy"]):
        logger.warning("Test metrics are NaN; skipping W&B test logging.")
        return

    logger.info(
        f"SST-2 Test - Loss: {test_scores['loss']:.4f}, Acc: {test_scores['accuracy']:.2f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SBERTJittor on SST-2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("base_model", nargs="?", default="bert-base-uncased",
                        help="Base encoder model name (resolved under ./hf if not a path)")
    parser.add_argument("--pooling", default="mean",
                        choices=["mean", "cls", "max"],
                        help="Pooling strategy")
    parser.add_argument("--encoder_checkpoint", type=str, default=None,
                        help="Optional pretrained encoder checkpoint (.bin/.pt)")
    parser.add_argument("--jittor_checkpoint", type=str, default=None,
                        help="Optional Jittor checkpoint (.pkl) with model_state")
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

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Evaluation batch size")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio (fraction of total steps)")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes")
    parser.add_argument("--use_cuda", action="store_true",
                        help="Use CUDA if available")
    parser.add_argument("--num_labels", type=int, default=2,
                        help="Number of SST labels")
    parser.add_argument("--train_encoder", action="store_true",
                        help="Fine-tune encoder weights (default: classifier-only)")

    parser.add_argument("--log_steps", type=int, default=100,
                        help="Log metrics every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate on validation set every N steps")
    # Checkpoint saving disabled by default for SST-2 training

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    cli_args = parse_args()
    train(cli_args)
