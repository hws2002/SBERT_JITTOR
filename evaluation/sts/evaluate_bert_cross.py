"""
Evaluate BERT cross-encoder on STS datasets.

This expects a HuggingFace sequence-classification model directory that includes
the regression head. Optionally, a separate head checkpoint can be loaded.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DATASET_MAP = {
    "sts12": "STS-12",
    "sts13": "STS-13",
    "sts14": "STS-14",
    "sts15": "STS-15",
    "sts16": "STS-16",
    "stsb": "STS-B",
    "sick-r": "SICKR",
}


def _cache_path(cache_dir: str, dataset_name: str, split: str, model_name: str, max_length: int) -> str:
    model_id = Path(model_name).name.replace("/", "_")
    name = f"{dataset_name}_{split}_pair_{model_id}_len{max_length}"
    return os.path.join(cache_dir, name)


def prepare_pair_dataset(
    data_dir: str,
    dataset_name: str,
    split: str,
    tokenizer,
    max_length: int,
    cache_dir: str,
    overwrite_cache: bool,
    tokenize_batch_size: int,
):
    from datasets import load_from_disk

    data_path = os.path.join(data_dir, dataset_name)
    raw_ds = load_from_disk(data_path)[split]

    cache_path = _cache_path(cache_dir, dataset_name, split, tokenizer.name_or_path, max_length)
    if os.path.isdir(cache_path) and not overwrite_cache:
        return load_from_disk(cache_path)

    def tokenize_fn(batch):
        tok = tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        tok["scores"] = batch["score"]
        return tok

    tokenized = raw_ds.map(
        tokenize_fn,
        batched=True,
        batch_size=tokenize_batch_size,
        remove_columns=raw_ds.column_names,
        desc=f"Tokenizing pair {dataset_name}/{split}",
    )
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(cache_path)
    return tokenized


def collate_pair(batch: List[Dict]) -> Dict[str, np.ndarray]:
    out = {
        "input_ids": np.asarray([b["input_ids"] for b in batch], dtype=np.int32),
        "attention_mask": np.asarray([b["attention_mask"] for b in batch], dtype=np.float32),
        "scores": np.asarray([b["scores"] for b in batch], dtype=np.float32),
    }
    if "token_type_ids" in batch[0]:
        out["token_type_ids"] = np.asarray([b["token_type_ids"] for b in batch], dtype=np.int32)
    return out


def _load_head_weights(model, head_checkpoint: str) -> None:
    payload = torch.load(head_checkpoint, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]

    # Try direct loading into classifier.
    try:
        model.classifier.load_state_dict(payload)
        return
    except Exception:
        pass

    # Try prefix matching.
    if isinstance(payload, dict):
        classifier_state = {}
        for key, value in payload.items():
            if key.startswith("classifier."):
                classifier_state[key.replace("classifier.", "")] = value
        if classifier_state:
            model.classifier.load_state_dict(classifier_state)
            return

    raise ValueError("Unable to load classifier head from checkpoint.")


def evaluate_cross_encoder(
    model,
    tokenizer,
    data_dir: str,
    dataset_name: str,
    split: str,
    max_length: int,
    cache_dir: str,
    overwrite_cache: bool,
    tokenize_batch_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
):
    from scipy.stats import pearsonr, spearmanr

    start_time = time.time()
    sts_dataset = prepare_pair_dataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        split=split,
        tokenizer=tokenizer,
        max_length=max_length,
        cache_dir=cache_dir,
        overwrite_cache=overwrite_cache,
        tokenize_batch_size=tokenize_batch_size,
    )
    dataloader = DataLoader(
        sts_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_pair,
    )

    all_predictions = []
    all_scores = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = torch.tensor(batch["input_ids"], device=device)
            attention_mask = torch.tensor(batch["attention_mask"], device=device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = torch.tensor(token_type_ids, device=device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits
            if logits.dim() == 2 and logits.size(-1) > 1:
                preds = logits[:, 0]
                logger.warning("Cross-encoder logits have more than 1 column; using logits[:, 0].")
            else:
                preds = logits.squeeze(-1)

            all_predictions.extend(preds.detach().cpu().numpy().tolist())
            all_scores.extend(batch["scores"].tolist())

    pearson_corr, _ = pearsonr(all_predictions, all_scores)
    spearman_corr, _ = spearmanr(all_predictions, all_scores)

    elapsed = time.time() - start_time
    return {
        "pearson": pearson_corr * 100,
        "spearman": spearman_corr * 100,
        "n_samples": len(sts_dataset),
        "eval_time": elapsed,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate BERT cross-encoder on STS datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace model directory for sequence classification")
    parser.add_argument("--head_checkpoint", type=str, default=None,
                        help="Optional classifier head checkpoint")

    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing datasets")
    parser.add_argument("--cache_dir", type=str, default="./data/tokenized",
                        help="Cache directory for tokenized datasets")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite existing cached datasets")
    parser.add_argument("--tokenize_batch_size", type=int, default=1024,
                        help="Batch size used during dataset tokenization")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Evaluation batch size")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes")
    parser.add_argument("--use_cuda", action="store_true",
                        help="Use CUDA if available")

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["stsb"],
        choices=["all", "sts12", "sts13", "sts14", "sts15", "sts16", "stsb", "sick-r"],
        help="Datasets to evaluate (default: stsb)",
    )
    parser.add_argument("--split", type=str, default="test",
                        help="Split to evaluate (STS-B supports validation/test)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to write JSON results (default: ./result/eval_bert_sts_<timestamp>.json)")
    parser.add_argument("--wandb", action="store_true",
                        help="Log evaluation results to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="sbert_evaluation",
                        help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name (default: auto-generated)")

    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, local_files_only=True)
    if args.head_checkpoint:
        logger.info(f"Loading classifier head from: {args.head_checkpoint}")
        _load_head_weights(model, args.head_checkpoint)
    model.to(device)

    wandb = None
    if args.wandb:
        try:
            import wandb as _wandb

            run_name = args.run_name if args.run_name else f"eval-bert-cross-{Path(args.model_path).name}"
            _wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "mode": "cross",
                    "model_path": args.model_path,
                    "batch_size": args.batch_size,
                    "max_length": args.max_length,
                    "split": args.split,
                    "datasets": args.datasets,
                },
            )
            wandb = _wandb
            logger.info(f"W&B initialized: {args.wandb_project}/{run_name}")
        except ImportError:
            logger.warning("wandb not installed. Skipping W&B logging.")

    dataset_keys = args.datasets
    if "all" in dataset_keys:
        dataset_keys = list(DATASET_MAP.keys())

    results = {}
    for key in dataset_keys:
        dataset_name = DATASET_MAP[key]
        split = args.split
        if dataset_name != "STS-B" and split != "test":
            split = "test"

        logger.info(f"Evaluating {dataset_name} ({split})...")
        scores = evaluate_cross_encoder(
            model=model,
            tokenizer=tokenizer,
            data_dir=args.data_dir,
            dataset_name=dataset_name,
            split=split,
            max_length=args.max_length,
            cache_dir=args.cache_dir,
            overwrite_cache=args.overwrite_cache,
            tokenize_batch_size=args.tokenize_batch_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        results[dataset_name] = scores
        logger.info(
            f"{dataset_name} - Pearson: {scores['pearson']:.2f}, Spearman: {scores['spearman']:.2f}"
        )

    avg_pearson = sum(v["pearson"] for v in results.values()) / max(len(results), 1)
    avg_spearman = sum(v["spearman"] for v in results.values()) / max(len(results), 1)

    payload = {
        "model_info": {
            "mode": "cross",
            "model_path": args.model_path,
            "max_length": args.max_length,
        },
        "evaluation": {
            "split": args.split,
            "batch_size": args.batch_size,
            "device": device,
        },
        "results": [
            {
                "dataset": name.lower().replace("sts-", "sts").replace("-b", "b"),
                "split": args.split if name == "STS-B" else "test",
                "pearson": scores["pearson"],
                "spearman": scores["spearman"],
                "n_samples": scores["n_samples"],
                "eval_time": scores["eval_time"],
            }
            for name, scores in results.items()
        ],
        "average": {
            "pearson": avg_pearson,
            "spearman": avg_spearman,
            "n_datasets": len(results),
        },
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }

    output_path = args.output_json
    if output_path is None:
        output_dir = Path("./result")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"eval_bert_sts_{payload['timestamp']}.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)

    logger.info(f"Evaluation complete. Results saved to {output_path}")
    if wandb:
        dataset_metrics = {}
        for name, scores in results.items():
            key = name.lower().replace("sts-", "sts").replace("-b", "b")
            dataset_metrics[f"{key}/pearson"] = scores["pearson"]
            dataset_metrics[f"{key}/spearman"] = scores["spearman"]
            dataset_metrics[f"{key}/n_samples"] = scores["n_samples"]
            dataset_metrics[f"{key}/eval_time"] = scores["eval_time"]
        dataset_metrics["avg/pearson"] = avg_pearson
        dataset_metrics["avg/spearman"] = avg_spearman
        dataset_metrics["avg/n_datasets"] = len(results)
        wandb.log(dataset_metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
