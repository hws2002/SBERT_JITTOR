"""
Train SBERT on NLI datasets (SNLI + MultiNLI) with evaluation on STS benchmark

The system trains BERT on the SNLI + MultiNLI (AllNLI) dataset
with cross-entropy loss. At every N training steps, the model is evaluated on the
STS benchmark dataset.

Inspired by: https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/nli/training_nli.py
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import numpy as np
import jittor as jt
from jittor import nn
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_from_disk, concatenate_datasets
from torch.utils.data import DataLoader

from model.sbert_model import SBERTJittor
from losses.softmax_loss import SoftmaxLoss
from losses.complex_softmax_loss import ComplexSoftmaxLoss

# Set the log level to INFO
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

STS_DATASETS = ["STS-12", "STS-13", "STS-14", "STS-15", "STS-16", "STS-B", "SICKR"]
HF_DIR = "./hf"
MODEL_DIR_MAP = {
    "bert-large-uncased": "hf_bert_large",
    "bert-base-uncased": "hf_bert_base",
}


def _jt_array(data, dtype: str):
    return jt.array(np.asarray(data, dtype=dtype))


def _to_jittor_batch(batch: Dict[str, Iterable], for_sts: bool = False) -> Dict[str, jt.Var]:
    """Convert a numpy batch to Jittor arrays."""
    out: Dict[str, jt.Var] = {
        "input_ids_a": _jt_array(batch["input_ids_a"], "int32"),
        "attention_mask_a": _jt_array(batch["attention_mask_a"], "float32"),
        "input_ids_b": _jt_array(batch["input_ids_b"], "int32"),
        "attention_mask_b": _jt_array(batch["attention_mask_b"], "float32"),
    }

    if "token_type_ids_a" in batch:
        out["token_type_ids_a"] = _jt_array(batch["token_type_ids_a"], "int32")
    if "token_type_ids_b" in batch:
        out["token_type_ids_b"] = _jt_array(batch["token_type_ids_b"], "int32")

    if for_sts:
        out["scores"] = _jt_array(batch["scores"], "float32")
    else:
        out["labels"] = _jt_array(batch["labels"], "int32")

    return out


def _safe_model_id(model_name: str) -> str:
    base_name = Path(model_name).name
    return base_name.replace("/", "_")


def _cache_path(cache_dir: str, dataset_name: str, split: str, model_name: str, max_length: int) -> str:
    model_id = _safe_model_id(model_name)
    name = f"{dataset_name}_{split}_{model_id}_len{max_length}"
    return os.path.join(cache_dir, name)


def _tokenize_pair(
    tokenizer,
    sentences_a: List[str],
    sentences_b: List[str],
    max_length: int
) -> Dict[str, List[List[int]]]:
    enc_a = tokenizer(
        sentences_a,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    enc_b = tokenizer(
        sentences_b,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    out = {
        "input_ids_a": enc_a["input_ids"],
        "attention_mask_a": enc_a["attention_mask"],
        "input_ids_b": enc_b["input_ids"],
        "attention_mask_b": enc_b["attention_mask"],
    }

    if "token_type_ids" in enc_a:
        out["token_type_ids_a"] = enc_a["token_type_ids"]
    if "token_type_ids" in enc_b:
        out["token_type_ids_b"] = enc_b["token_type_ids"]

    return out


def prepare_nli_dataset(
    data_dir: str,
    datasets: List[str],
    split: str,
    tokenizer,
    max_length: int,
    cache_dir: str,
    overwrite_cache: bool,
    tokenize_batch_size: int
):
    def _map_split(dataset_name: str, requested_split: str) -> str:
        if dataset_name.lower() in {"multinli", "multi_nli", "multi-nli"}:
            if requested_split in {"train", "validation_matched", "validation_mismatched"}:
                return requested_split
            if requested_split in {"validation", "dev"}:
                return "validation_matched"
            if requested_split == "test":
                return "validation_mismatched"
            raise ValueError(
                f"Unsupported split '{requested_split}' for MultiNLI. "
                "Use train / validation_matched / validation_mismatched."
            )

        if dataset_name.lower() == "snli":
            if requested_split in {"train", "validation", "test"}:
                return requested_split
            if requested_split in {"validation_matched", "dev"}:
                return "validation"
            if requested_split == "validation_mismatched":
                return "test"
            raise ValueError(
                f"Unsupported split '{requested_split}' for SNLI. Use train / validation / test."
            )

        return requested_split

    tokenized_datasets = []

    for dataset_name in datasets:
        data_path = os.path.join(data_dir, dataset_name)
        actual_split = _map_split(dataset_name, split)
        raw_ds = load_from_disk(data_path)[actual_split]
        raw_ds = raw_ds.filter(lambda x: x["label"] != -1)

        cache_path = _cache_path(cache_dir, dataset_name, actual_split, tokenizer.name_or_path, max_length)
        if os.path.isdir(cache_path) and not overwrite_cache:
            logger.info(f"Loading cached NLI dataset: {cache_path}")
            tokenized = load_from_disk(cache_path)
        else:
            logger.info(f"Tokenizing NLI dataset: {dataset_name}/{actual_split}")

            def tokenize_fn(batch):
                tok = _tokenize_pair(
                    tokenizer,
                    batch["premise"],
                    batch["hypothesis"],
                    max_length
                )
                tok["labels"] = batch["label"]
                return tok

            tokenized = raw_ds.map(
                tokenize_fn,
                batched=True,
                batch_size=tokenize_batch_size,
                remove_columns=raw_ds.column_names,
                desc=f"Tokenizing {dataset_name}/{actual_split}"
            )
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            tokenized.save_to_disk(cache_path)

        tokenized_datasets.append(tokenized)

    if len(tokenized_datasets) == 1:
        return tokenized_datasets[0]
    return concatenate_datasets(tokenized_datasets)


def prepare_sts_dataset(
    data_dir: str,
    dataset_name: str,
    split: str,
    tokenizer,
    max_length: int,
    cache_dir: str,
    overwrite_cache: bool,
    tokenize_batch_size: int
):
    data_path = os.path.join(data_dir, dataset_name)
    raw_ds = load_from_disk(data_path)[split]

    cache_path = _cache_path(cache_dir, dataset_name, split, tokenizer.name_or_path, max_length)
    if os.path.isdir(cache_path) and not overwrite_cache:
        logger.info(f"Loading cached STS dataset: {cache_path}")
        return load_from_disk(cache_path)

    logger.info(f"Tokenizing STS dataset: {dataset_name}/{split}")

    def tokenize_fn(batch):
        tok = _tokenize_pair(
            tokenizer,
            batch["sentence1"],
            batch["sentence2"],
            max_length
        )
        tok["scores"] = batch["score"]
        return tok

    tokenized = raw_ds.map(
        tokenize_fn,
        batched=True,
        batch_size=tokenize_batch_size,
        remove_columns=raw_ds.column_names,
        desc=f"Tokenizing {dataset_name}/{split}"
    )
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(cache_path)
    return tokenized


def collate_nli(batch: List[Dict]) -> Dict[str, np.ndarray]:
    out = {
        "input_ids_a": np.asarray([b["input_ids_a"] for b in batch], dtype=np.int32),
        "attention_mask_a": np.asarray([b["attention_mask_a"] for b in batch], dtype=np.float32),
        "input_ids_b": np.asarray([b["input_ids_b"] for b in batch], dtype=np.int32),
        "attention_mask_b": np.asarray([b["attention_mask_b"] for b in batch], dtype=np.float32),
        "labels": np.asarray([b["labels"] for b in batch], dtype=np.int32),
    }

    if "token_type_ids_a" in batch[0]:
        out["token_type_ids_a"] = np.asarray([b["token_type_ids_a"] for b in batch], dtype=np.int32)
    if "token_type_ids_b" in batch[0]:
        out["token_type_ids_b"] = np.asarray([b["token_type_ids_b"] for b in batch], dtype=np.int32)

    return out


def collate_sts(batch: List[Dict]) -> Dict[str, np.ndarray]:
    out = {
        "input_ids_a": np.asarray([b["input_ids_a"] for b in batch], dtype=np.int32),
        "attention_mask_a": np.asarray([b["attention_mask_a"] for b in batch], dtype=np.float32),
        "input_ids_b": np.asarray([b["input_ids_b"] for b in batch], dtype=np.int32),
        "attention_mask_b": np.asarray([b["attention_mask_b"] for b in batch], dtype=np.float32),
        "scores": np.asarray([b["scores"] for b in batch], dtype=np.float32),
    }

    if "token_type_ids_a" in batch[0]:
        out["token_type_ids_a"] = np.asarray([b["token_type_ids_a"] for b in batch], dtype=np.int32)
    if "token_type_ids_b" in batch[0]:
        out["token_type_ids_b"] = np.asarray([b["token_type_ids_b"] for b in batch], dtype=np.int32)

    return out


def evaluate_sts(model, dataloader):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    logger.info("Evaluating on STS-B...")
    model.eval()

    all_predictions = []
    all_scores = []

    with jt.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating STS-B", leave=False):
            jt_batch = _to_jittor_batch(batch, for_sts=True)

            emb_a = model.encode(
                jt_batch["input_ids_a"],
                jt_batch["attention_mask_a"],
                jt_batch.get("token_type_ids_a", None)
            )
            emb_b = model.encode(
                jt_batch["input_ids_b"],
                jt_batch["attention_mask_b"],
                jt_batch.get("token_type_ids_b", None)
            )

            emb_a_np = emb_a.numpy()
            emb_b_np = emb_b.numpy()
            denom = np.linalg.norm(emb_a_np, axis=1) * np.linalg.norm(emb_b_np, axis=1) + 1e-9
            sim = np.sum(emb_a_np * emb_b_np, axis=1) / denom

            all_predictions.extend(sim.tolist())
            all_scores.extend(np.asarray(batch["scores"]).tolist())

    pearson_corr, _ = pearsonr(all_predictions, all_scores)
    spearman_corr, _ = spearmanr(all_predictions, all_scores)

    results = {
        "pearson": pearson_corr * 100,
        "spearman": spearman_corr * 100,
    }

    logger.info(f"STS-B - Pearson: {results['pearson']:.2f}, Spearman: {results['spearman']:.2f}")
    model.train()
    return results


def evaluate_sts_all(
    model,
    tokenizer,
    data_dir: str,
    cache_dir: str,
    max_length: int,
    overwrite_cache: bool,
    tokenize_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
):
    all_results = {}
    for dataset_name in STS_DATASETS:
        split = "test"
        sts_dataset = prepare_sts_dataset(
            data_dir=data_dir,
            dataset_name=dataset_name,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
            tokenize_batch_size=tokenize_batch_size,
        )
        sts_dataloader = DataLoader(
            sts_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_sts,
        )
        logger.info(f"Evaluating {dataset_name} ({split})...")
        scores = evaluate_sts(model=model, dataloader=sts_dataloader)
        all_results[dataset_name] = {
            "split": split,
            "pearson": scores["pearson"],
            "spearman": scores["spearman"],
            "n_samples": len(sts_dataset),
        }
    return all_results


def save_eval_results(
    results: Dict[str, Dict[str, float]],
    args,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.base_model.replace("/", "_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_{safe_model}_{timestamp}.json"

    avg_pearson = sum(v["pearson"] for v in results.values()) / max(len(results), 1)
    avg_spearman = sum(v["spearman"] for v in results.values()) / max(len(results), 1)

    payload = {
        "model_info": {
            "type": "sbert_jittor",
            "model_name": safe_model,
            "max_length": args.max_length,
        },
        "evaluation": {
            "split": "test",
            "batch_size": args.eval_batch_size,
            "device": "cuda" if jt.flags.use_cuda else "cpu",
        },
        "results": [
            {
                "dataset": name.lower().replace("sts-", "sts").replace("-b", "b"),
                "split": info["split"],
                "pearson": info["pearson"],
                "spearman": info["spearman"],
                "n_samples": info["n_samples"],
            }
            for name, info in results.items()
        ],
        "average": {
            "pearson": avg_pearson,
            "spearman": avg_spearman,
            "n_datasets": len(results),
        },
        "timestamp": timestamp,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)

    logger.info(f"Saved evaluation results to {output_path}")
    return payload, output_path

def setup_device(use_cuda):
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

        if args.run_name:
            run_name = args.run_name
        else:
            ablation_suffix = f"-abl{args.ablation}" if args.ablation else ""
            run_name = f"nli-{args.base_model.split('/')[-1]}-{args.pooling}{ablation_suffix}"

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": args.base_model,
                "pooling": args.pooling,
                "loss": args.loss,
                "ablation": args.ablation,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "max_length": args.max_length,
                "warmup_ratio": args.warmup_ratio,
                "epochs": args.epochs,
                "datasets": args.datasets,
                "eval_steps": args.eval_steps,
                "num_workers": args.num_workers,
            }
        )
        logger.info(f"W&B initialized: {args.wandb_project}/{run_name}")
        return wandb
    except ImportError:
        logger.warning("wandb not installed. Skipping W&B logging.")
        return None


def _filtered_loss_state(train_loss):
    loss_state = train_loss.state_dict()
    return {k: v for k, v in loss_state.items() if not k.startswith("model.")}


def save_checkpoint(model, train_loss, optimizer, iteration, epoch, args, name="checkpoint"):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model = args.base_model.replace("/", "_")
    if name == "best":
        checkpoint_path = output_dir / f"{safe_model}_best.pkl"
    else:
        logger.info("Skipping non-best checkpoint save (encoder-only mode).")
        return None

    checkpoint = {
        "model_state": model.state_dict(),
        "base_model": args.base_model,
        "pooling": args.pooling,
    }

    jt.save(checkpoint, str(checkpoint_path))
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def save_encoder_only(model, args, output_dir: Path, name: str):
    payload = {
        "model_state": model.state_dict(),
        "base_model": args.base_model,
        "pooling": args.pooling,
    }
    checkpoint_path = output_dir / f"{name}.pkl"
    jt.save(payload, str(checkpoint_path))
    logger.info(f"Encoder-only checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_training_checkpoint(model, train_loss, optimizer, checkpoint_path: str) -> tuple[int, int]:
    logger.info(f"Loading training checkpoint: {checkpoint_path}")
    checkpoint = jt.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    if "loss_state" in checkpoint:
        try:
            train_loss.load_state_dict(checkpoint["loss_state"])
        except Exception as exc:
            logger.warning(f"Partial loss_state load (expected): {exc}")
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
    wandb = setup_wandb(args)

    tokenizer_source = args.tokenizer_dir or args.base_model
    if not os.path.isdir(tokenizer_source):
        mapped = MODEL_DIR_MAP.get(args.base_model, args.base_model)
        candidate = os.path.join(HF_DIR, mapped)
        if os.path.isdir(candidate):
            tokenizer_source = candidate
        elif args.encoder_checkpoint and os.path.isfile(args.encoder_checkpoint):
            checkpoint_dir = os.path.dirname(args.encoder_checkpoint)
            if os.path.isdir(checkpoint_dir):
                tokenizer_source = checkpoint_dir
    if not os.path.isdir(tokenizer_source):
        raise ValueError(
            "Expected local model directory for tokenizer (base_model or encoder_checkpoint dir)."
        )
    logger.info(f"Loading tokenizer from: {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, local_files_only=True)

    cache_dir = args.cache_dir or os.path.join(args.data_dir, "_cache")

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
        checkpoint_path=args.encoder_checkpoint,
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
    optimizer = nn.Adam(train_loss.parameters(), lr=args.lr)
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
    if wandb:
        wandb.log({
            "eval/pearson": eval_results_before["pearson"],
            "eval/spearman": eval_results_before["spearman"],
            "step": 0,
        })

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

                if wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/avg_loss": avg_loss,
                        "train/accuracy": accuracy,
                        "train/lr": current_lr,
                        "step": global_step,
                    })

            if global_step % args.eval_steps == 0:
                eval_results = evaluate_sts(model=model, dataloader=sts_dataloader)

                if wandb:
                    wandb.log({
                        "eval/pearson": eval_results["pearson"],
                        "eval/spearman": eval_results["spearman"],
                        "step": global_step,
                    })

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

        if wandb:
            wandb.log({
                "epoch/loss": avg_loss,
                "epoch/accuracy": accuracy,
                "epoch": epoch + 1,
            })

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

    safe_model = args.base_model.replace("/", "_")
    best_path = Path(args.output_dir) / f"{safe_model}_best.pkl"
    if best_path.is_file():
        logger.info(f"Loading best checkpoint for final test: {best_path}")
        best_state = jt.load(str(best_path))
        model.load_state_dict(best_state["model_state"])

    test_results = evaluate_sts(model=model, dataloader=sts_test_dataloader)

    if wandb:
        wandb.log({
            "test/pearson": test_results["pearson"],
            "test/spearman": test_results["spearman"],
        })

    logger.info("\n" + "=" * 70)
    logger.info("Training completed!")
    logger.info(f"Best Spearman score (dev): {best_spearman:.2f}")
    logger.info(f"Final Spearman score (test): {test_results['spearman']:.2f}")
    logger.info("=" * 70)

    logger.info("\n" + "=" * 70)
    logger.info("Final Evaluation on STS/SICKR Test Sets (Best Checkpoint)")
    logger.info("=" * 70)

    best_path = Path(args.output_dir) / f"{safe_model}_best.pkl"
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
    payload, output_path = save_eval_results(all_results, args, Path(args.output_dir) / "result")

    if wandb:
        dataset_metrics = {}
        for name, info in all_results.items():
            key = name.lower().replace("sts-", "sts").replace("-b", "b")
            dataset_metrics[f"{key}/pearson"] = info["pearson"]
            dataset_metrics[f"{key}/spearman"] = info["spearman"]
            dataset_metrics[f"{key}/n_samples"] = info["n_samples"]
        dataset_metrics["avg/pearson"] = payload["average"]["pearson"]
        dataset_metrics["avg/spearman"] = payload["average"]["spearman"]
        dataset_metrics["avg/n_datasets"] = payload["average"]["n_datasets"]
        wandb.log(dataset_metrics)

    if wandb:
        wandb.finish()


def train_torch(args):
    import torch
    from torch.utils.data import DataLoader as TorchDataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    model_source = args.base_model
    if not os.path.isdir(model_source):
        mapped = MODEL_DIR_MAP.get(args.base_model, args.base_model)
        candidate = os.path.join(HF_DIR, mapped)
        if os.path.isdir(candidate):
            model_source = candidate
    if not os.path.isdir(model_source):
        raise ValueError("Torch framework requires a local model directory for base_model.")

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using torch device: {device}")

    logger.info(f"Loading tokenizer from: {model_source}")
    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True, local_files_only=True)

    cache_dir = args.cache_dir or os.path.join(args.data_dir, "_cache")

    logger.info("Preparing NLI training data (cached tokenization)...")
    train_dataset = prepare_nli_dataset(
        data_dir=args.data_dir,
        datasets=args.datasets,
        split="train",
        tokenizer=tokenizer,
        max_length=args.max_length,
        cache_dir=cache_dir,
        overwrite_cache=args.overwrite_cache,
        tokenize_batch_size=args.tokenize_batch_size,
    )

    train_dataloader = TorchDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_nli,
    )

    total_steps = args.epochs * len(train_dataloader)
    if total_steps == 0:
        raise RuntimeError("No training data found. Check data_dir/datasets arguments.")
    logger.info(f"Total training steps: {total_steps}")

    logger.info(f"Initializing Hugging Face classification model from: {model_source}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        num_labels=args.num_labels,
        local_files_only=True,
    )
    if args.encoder_checkpoint:
        logger.info(f"Loading encoder checkpoint: {args.encoder_checkpoint}")
        state = torch.load(args.encoder_checkpoint, map_location="cpu")
        missing, unexpected = model.base_model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.info(f"Encoder load missing={len(missing)} unexpected={len(unexpected)}")

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    warmup_steps = max(int(total_steps * args.warmup_ratio), 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    logger.info("\nStarting torch training...")
    logger.info("=" * 70)

    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for step, batch in enumerate(epoch_iterator, 1):
            input_ids = torch.tensor(batch["input_ids_a"], device=device)
            attention_mask = torch.tensor(batch["attention_mask_a"], device=device)
            token_type_ids = batch.get("token_type_ids_a")
            if token_type_ids is not None:
                token_type_ids = torch.tensor(token_type_ids, device=device)
            labels = torch.tensor(batch["labels"], device=device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs.loss
            logits = outputs.logits

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1

            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                correct = (preds == labels).sum().item()
                batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += batch_size

            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples * 100
            epoch_iterator.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "acc": f"{accuracy:.2f}%",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            if global_step % args.log_steps == 0:
                logger.info(
                    f"Step {global_step}/{total_steps} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Acc: {accuracy:.2f}% | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

    logger.info("\nSaving torch model...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)

    encoder_dir = final_dir / "hf_encoder"
    encoder_dir.mkdir(exist_ok=True)

    model.base_model.save_pretrained(encoder_dir)
    tokenizer.save_pretrained(encoder_dir)

    logger.info(f"Saved HF encoder to {encoder_dir}")


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
    parser.add_argument("--framework", default="jittor",
                        choices=["jittor", "torch"],
                        help="Training framework")
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
    parser.add_argument("--cache_dir", type=str, default='./data/tokenized',
                        help="Cache directory for tokenized datasets (default: data_dir/_cache)")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite existing cached datasets")
    parser.add_argument("--tokenize_batch_size", type=int, default=1024,
                        help="Batch size used during dataset tokenization")

    parser.add_argument("--wandb", action="store_true",
                        help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="sbert-nli-training",
                        help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name (default: auto-generated)")

    args = parser.parse_args()

    if args.output_dir is None:
        model_name = args.base_model.replace("/", "-")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = f"checkpoints/training_nli_{model_name}-{timestamp}"

    return args


if __name__ == "__main__":
    args = parse_args()
    if args.framework == "torch":
        train_torch(args)
    else:
        train(args)
