"""
Train SBERTJittor on SST-2 classification.
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


def save_checkpoint(model, classifier, args, name: str):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.base_model.replace("/", "_")
    checkpoint_path = output_dir / f"{safe_model}_{name}.pkl"
    payload = {
        "model_state": model.state_dict(),
        "classifier_state": classifier.state_dict(),
        "base_model": args.base_model,
        "pooling": args.pooling,
    }
    jt.save(payload, str(checkpoint_path))
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


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

    tokenizer_source = _resolve_tokenizer_source(args.base_model)
    logger.info(f"Loading tokenizer from: {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, local_files_only=True)

    cache_dir = args.cache_dir or os.path.join(args.data_dir, "_cache")

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

    model = SBERTJittor(
        encoder_name=args.base_model,
        pooling=args.pooling,
        head_type="none",
        checkpoint_path=args.encoder_checkpoint,
    )
    classifier = nn.Linear(model.output_dim, args.num_labels)
    optimizer = nn.Adam(list(model.parameters()) + list(classifier.parameters()), lr=args.lr)
    loss_fct = nn.CrossEntropyLoss()

    best_acc = -1.0
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_samples = 0
        for batch in train_loader:
            jt_batch = _to_jittor_batch(batch)
            reps = model.encode(
                jt_batch["input_ids"],
                jt_batch["attention_mask"],
                jt_batch.get("token_type_ids", None),
            )
            logits = classifier(reps)
            loss = loss_fct(logits, jt_batch["labels"])
            optimizer.step(loss)
            batch_size = jt_batch["labels"].shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        logger.info(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {avg_loss:.4f}")

        dev_scores = evaluate(model, classifier, dev_loader)
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} - Dev Loss: {dev_scores['loss']:.4f} "
            f"Acc: {dev_scores['accuracy']:.2f}"
        )
        if dev_scores["accuracy"] > best_acc:
            best_acc = dev_scores["accuracy"]
            save_checkpoint(model, classifier, args, "best")

    logger.info("Training completed.")
    logger.info(f"Best Dev Accuracy: {best_acc:.2f}")


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
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes")
    parser.add_argument("--use_cuda", action="store_true",
                        help="Use CUDA if available")
    parser.add_argument("--num_labels", type=int, default=2,
                        help="Number of SST labels")

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for checkpoints")

    args = parser.parse_args()

    if args.output_dir is None:
        model_name = args.base_model.replace("/", "-")
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = f"checkpoints/training_sst_{model_name}-{timestamp}"

    return args


if __name__ == "__main__":
    cli_args = parse_args()
    train(cli_args)
