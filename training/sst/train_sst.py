"""
Train SBERTJittor on SST-2 classification.

Mirrors train_nli_gpu.py style with warmup, eval, and best checkpoint.
Default trains classifier-only; use --train_encoder for full fine-tuning.
"""

import argparse
import logging
import os
import sys
import time
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
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
}


def _jt_array(data, dtype: str):
    return jt.array(np.asarray(data, dtype=dtype))


def _to_jittor_batch(batch: Dict[str, Iterable]) -> Dict[str, jt.Var]:
    out: Dict[str, jt.Var] = {
        "input_ids": _jt_array(batch["input_ids"], "int32"),
        "attention_mask": _jt_array(batch["attention_mask"], "float32"),
        "labels": _jt_array(batch["labels"], "int32"),
    }
    if "token_type_ids" in batch:
        out["token_type_ids"] = _jt_array(batch["token_type_ids"], "int32")
    return out


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

        run_name = args.run_name if args.run_name else f"sst-{args.base_model}-{args.pooling}"
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
                "warmup_ratio": args.warmup_ratio,
            },
        )
        logger.info(f"W&B initialized: {args.wandb_project}/{run_name}")
        return wandb
    except ImportError:
        logger.warning("wandb not installed. Skipping W&B logging.")
        return None


def _resolve_tokenizer_source(base_model: str) -> str:
    tokenizer_source = base_model
    if not os.path.isdir(tokenizer_source):
        mapped = MODEL_DIR_MAP.get(base_model, base_model)
        candidate = os.path.join(HF_DIR, mapped)
        if os.path.isdir(candidate):
            tokenizer_source = candidate
    if not os.path.isdir(tokenizer_source):
        raise ValueError("Expected local model directory for tokenizer (base_model).")
    return tokenizer_source


def evaluate(model, classifier, dataloader) -> Dict[str, float]:
    model.eval()
    classifier.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    loss_fct = nn.CrossEntropyLoss()

    with jt.no_grad():
        for batch in dataloader:
            jt_batch = _to_jittor_batch(batch)
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


def save_checkpoint(model, classifier, optimizer, iteration, epoch, args, name: str):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.base_model.replace("/", "_")
    if name == "best":
        checkpoint_path = output_dir / f"{safe_model}_best.pkl"
    else:
        checkpoint_path = output_dir / f"{safe_model}_{name}.pkl"

    payload = {
        "iteration": iteration,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "classifier_state": classifier.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "base_model": args.base_model,
        "pooling": args.pooling,
        "num_labels": args.num_labels,
    }
    jt.save(payload, str(checkpoint_path))
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_training_checkpoint(model, classifier, optimizer, checkpoint_path: str):
    logger.info(f"Loading training checkpoint: {checkpoint_path}")
    checkpoint = jt.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    if "classifier_state" in checkpoint:
        classifier.load_state_dict(checkpoint["classifier_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        except Exception as exc:
            logger.warning(f"Failed to load optimizer state: {exc}")

    global_step = int(checkpoint.get("iteration", 0))
    start_epoch = max(int(checkpoint.get("epoch", 1)) - 1, 0)
    logger.info(f"Resuming from step {global_step}, epoch {start_epoch + 1}")
    return global_step, start_epoch


def train(args):
    logger.info("=" * 70)
    logger.info("SBERT SST-2 Training")
    logger.info("=" * 70)
    logger.info(f"Model: {args.base_model}")
    logger.info(f"Pooling: {args.pooling}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info("=" * 70)

    setup_device(args.use_cuda)
    wandb = setup_wandb(args)

    tokenizer_source = _resolve_tokenizer_source(args.base_model)
    logger.info(f"Loading tokenizer from: {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, local_files_only=True)

    cache_dir = args.cache_dir or os.path.join(args.data_dir, "_cache")

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
        checkpoint_path=args.encoder_checkpoint,
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
    if args.start_from_checkpoints:
        global_step, start_epoch = load_training_checkpoint(
            model, classifier, optimizer, args.start_from_checkpoints
        )

    logger.info("Evaluation before training:")
    eval_before = evaluate(model, classifier, dev_loader)
    logger.info(
        f"Initial Eval - Loss: {eval_before['loss']:.4f}, Acc: {eval_before['accuracy']:.2f}"
    )
    if wandb:
        wandb.log(
            {
                "eval/loss": eval_before["loss"],
                "eval/accuracy": eval_before["accuracy"],
                "step": global_step,
            }
        )

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    best_acc = eval_before["accuracy"]

    logger.info("\nStarting training...")
    logger.info("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        for step, batch in enumerate(train_loader, 1):
            jt_batch = _to_jittor_batch(batch)
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
                if wandb:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/avg_loss": avg_loss,
                            "train/accuracy": acc,
                            "train/lr": optimizer.lr,
                            "step": global_step,
                        }
                    )

            if global_step % args.eval_steps == 0:
                eval_scores = evaluate(model, classifier, dev_loader)
                logger.info(
                    f"Eval - Loss: {eval_scores['loss']:.4f}, "
                    f"Acc: {eval_scores['accuracy']:.2f}"
                )
                if wandb:
                    wandb.log(
                        {
                            "eval/loss": eval_scores["loss"],
                            "eval/accuracy": eval_scores["accuracy"],
                            "step": global_step,
                        }
                    )
                if eval_scores["accuracy"] > best_acc:
                    best_acc = eval_scores["accuracy"]
                    save_checkpoint(model, classifier, optimizer, global_step, epoch + 1, args, name="best")

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_checkpoint(model, classifier, optimizer, global_step, epoch + 1, args, name="checkpoint")

        logger.info(
            f"\nEpoch {epoch + 1}/{args.epochs} Summary: "
            f"Loss {avg_loss:.4f}, Acc {acc:.2f}%"
        )
        if wandb:
            wandb.log(
                {
                    "epoch/loss": avg_loss,
                    "epoch/accuracy": acc,
                    "epoch": epoch + 1,
                }
            )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

    logger.info("\nFinal Evaluation on SST-2 Test Split")
    test_dataset = prepare_sst_dataset(
        data_dir=args.data_dir,
        split="test",
        tokenizer=tokenizer,
        max_length=args.max_length,
        cache_dir=cache_dir,
        overwrite_cache=args.overwrite_cache,
        tokenize_batch_size=args.tokenize_batch_size,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sst,
    )

    test_scores = evaluate(model, classifier, test_loader)
    logger.info(
        f"SST-2 Test - Loss: {test_scores['loss']:.4f}, Acc: {test_scores['accuracy']:.2f}"
    )
    if wandb:
        wandb.log(
            {
                "test/loss": test_scores["loss"],
                "test/accuracy": test_scores["accuracy"],
            }
        )
        wandb.finish()


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

    parser.add_argument("--data_dir", default="./data",
                        help="Directory containing datasets")
    parser.add_argument("--cache_dir", type=str, default="./data/tokenized",
                        help="Cache directory for tokenized datasets")
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
    parser.add_argument("--save_steps", type=int, default=0,
                        help="Save checkpoint every N steps (0 to disable)")
    parser.add_argument("--start_from_checkpoints", type=str, default=None,
                        help="Path to a saved checkpoint to resume training")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for checkpoints")

    parser.add_argument("--wandb", action="store_true",
                        help="Log training metrics to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="sbert-sst",
                        help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name (default: auto-generated)")

    args = parser.parse_args()

    if args.output_dir is None:
        model_name = args.base_model.replace("/", "-")
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = f"checkpoints/training_sst_{model_name}-{timestamp}"

    return args


if __name__ == "__main__":
    cli_args = parse_args()
    train(cli_args)
