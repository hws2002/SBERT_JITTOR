"""
Evaluate a trained SBERT checkpoint on STS-style datasets.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import jittor as jt
from jittor.dataset import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.sbert_model import SBERTJittor
from utils.data_loader import collate_sts, prepare_sts_dataset
from utils.jt_utils import _to_jittor_batch

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

MODEL_DIR_MAP = {
    "bert-large-uncased": "hf_bert_large",
    "bert-base-uncased": "hf_bert_base",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
}


def _safe_model_id(model_name: str) -> str:
    return model_name.replace("/", "_")


def _cache_path(cache_dir: str, dataset_name: str, split: str, model_name: str, max_length: int) -> str:
    model_id = _safe_model_id(model_name)
    name = f"{dataset_name}_{split}_{model_id}_len{max_length}"
    return os.path.join(cache_dir, name)


def _resolve_sts_file(data_dir: str, dataset_name: str, split: str) -> str:
    root = os.path.join(data_dir, dataset_name)
    if dataset_name.upper() == "STS-B":
        if split == "validation":
            return os.path.join(root, "dev.tsv")
        if split == "test":
            return os.path.join(root, "test.tsv")
        return os.path.join(root, "train.tsv")
    if dataset_name.upper().startswith("STS-"):
        return os.path.join(root, f"{split}.tsv")
    if dataset_name.upper() in {"SICKR", "SICK-R"}:
        return os.path.join(root, f"{split}.tsv")
    return os.path.join(root, f"{split}.tsv")


def _load_sts_pairs_local(data_dir: str, dataset_name: str, split: str):
    path = _resolve_sts_file(data_dir, dataset_name, split)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing STS file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle if line.strip()]
    if not lines:
        return [], [], np.asarray([], dtype=np.float32)
    header = lines[0].split("\t")
    has_header = any("sentence" in col.lower() or "score" in col.lower() for col in header)
    rows = lines[1:] if has_header else lines
    s1 = []
    s2 = []
    scores = []
    if has_header:
        lower = [h.lower() for h in header]
        s1_idx = lower.index("sentence1") if "sentence1" in lower else 0
        s2_idx = lower.index("sentence2") if "sentence2" in lower else 1
        score_idx = lower.index("score") if "score" in lower else 2
        for row in rows:
            parts = row.split("\t")
            if len(parts) <= max(s1_idx, s2_idx, score_idx):
                continue
            s1.append(parts[s1_idx])
            s2.append(parts[s2_idx])
            scores.append(float(parts[score_idx]))
    else:
        for row in rows:
            parts = row.split("\t")
            if len(parts) < 3:
                continue
            s1.append(parts[0])
            s2.append(parts[1])
            scores.append(float(parts[2]))
    return s1, s2, np.asarray(scores, dtype=np.float32)

def evaluate_dataset(
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
    require_cache: bool,
):
    from scipy.stats import pearsonr, spearmanr

    start_time = time.time()
    cache_path = _cache_path(cache_dir, dataset_name, split, tokenizer.name_or_path, max_length)
    if require_cache and not os.path.isdir(cache_path):
        raise FileNotFoundError(
            f"Missing cached dataset at {cache_path}. "
            "Generate caches first or disable --require_cache."
        )
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
    dataloader = DataLoader(
        sts_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_sts,
    )

    all_predictions = []
    all_scores = []

    model.eval()
    with jt.no_grad():
        for batch in dataloader:
            jt_batch = _to_jittor_batch(batch, for_sts=True)
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

    elapsed = time.time() - start_time
    return {
        "pearson": pearson_corr * 100,
        "spearman": spearman_corr * 100,
        "n_samples": len(sts_dataset),
        "eval_time": elapsed,
    }


def evaluate_dataset_torch(
    model,
    data_dir: str,
    dataset_name: str,
    split: str,
    batch_size: int,
):
    from scipy.stats import pearsonr, spearmanr

    start_time = time.time()
    sentences1, sentences2, scores = _load_sts_pairs_local(data_dir, dataset_name, split)

    emb_a = model.encode(sentences1, batch_size=batch_size, show_progress_bar=False)
    emb_b = model.encode(sentences2, batch_size=batch_size, show_progress_bar=False)

    denom = np.linalg.norm(emb_a, axis=1) * np.linalg.norm(emb_b, axis=1) + 1e-9
    sim = np.sum(emb_a * emb_b, axis=1) / denom

    pearson_corr, _ = pearsonr(sim, scores)
    spearman_corr, _ = spearmanr(sim, scores)

    elapsed = time.time() - start_time
    return {
        "pearson": pearson_corr * 100,
        "spearman": spearman_corr * 100,
        "n_samples": len(scores),
        "eval_time": elapsed,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SBERT checkpoint on STS datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to a saved Jittor checkpoint (.pkl)")
    parser.add_argument("--base_model", type=str, default="bert-base-uncased",
                        help="Base encoder model name")
    parser.add_argument("--framework", type=str, default="jittor",
                        choices=["jittor", "torch"],
                        help="Evaluation framework")
    parser.add_argument("--encoder_checkpoint_path", type=str, default="./hf",
                        help="Base directory containing local models/tokenizers")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "cls", "max"],
                        help="Pooling strategy")
    parser.add_argument("--num_labels", type=int, default=3,
                        help="Number of NLI classification labels")
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
    parser.add_argument("--require_cache", action="store_true",
                        help="Require pre-tokenized cache (no tokenization during eval)")

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "sts12", "sts13", "sts14", "sts15", "sts16", "stsb", "sick-r"],
        help="Datasets to evaluate (default: all)",
    )
    parser.add_argument("--split", type=str, default="test",
                        help="Split to evaluate (STS-B supports validation/test)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to write JSON results (default: ./result/eval_<timestamp>.json)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.framework == "jittor":
        if args.use_cuda and jt.has_cuda:
            jt.flags.use_cuda = 1
            logger.info("Using CUDA")
        else:
            jt.flags.use_cuda = 0
            logger.info("Using CPU")

    tokenizer = None
    model = None
    if args.framework == "jittor":
        tokenizer_source = args.base_model
        if not os.path.isdir(tokenizer_source):
            mapped = MODEL_DIR_MAP.get(args.base_model, args.base_model)
            candidate = os.path.join(args.encoder_checkpoint_path, mapped)
            if os.path.isdir(candidate):
                tokenizer_source = candidate
        if not os.path.isdir(tokenizer_source):
            raise ValueError(
                "Tokenizer not found locally. Provide a local base_model directory "
                "or place it under --encoder_checkpoint_path."
            )
        logger.info(f"Loading tokenizer from: {tokenizer_source}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

        logger.info("Loading checkpoint...")
        checkpoint = jt.load(args.checkpoint_path)
        model = SBERTJittor(
            encoder_name=args.base_model,
            pooling=args.pooling,
            head_type="none",
            checkpoint_path=None,
        )
        model.load_state_dict(checkpoint["model_state"])
    else:
        from sentence_transformers import SentenceTransformer

        model_path = args.checkpoint_path
        logger.info(f"Loading sentence-transformers model from: {model_path}")
        model = SentenceTransformer(
            model_path,
            device="cuda" if args.use_cuda else "cpu",
        )

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
        if args.framework == "jittor":
            scores = evaluate_dataset(
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
                require_cache=args.require_cache,
            )
        else:
            scores = evaluate_dataset_torch(
                model=model,
                data_dir=args.data_dir,
                dataset_name=dataset_name,
                split=split,
                batch_size=args.batch_size,
            )
        results[dataset_name] = scores
        logger.info(
            f"{dataset_name} - Pearson: {scores['pearson']:.2f}, Spearman: {scores['spearman']:.2f}"
        )
    avg_pearson = sum(v["pearson"] for v in results.values()) / max(len(results), 1)
    avg_spearman = sum(v["spearman"] for v in results.values()) / max(len(results), 1)

    payload = {
        "model_info": {
            "model_name": args.base_model,
            "max_length": args.max_length,
        },
        "evaluation": {
            "split": args.split,
            "batch_size": args.batch_size,
            "device": "cuda" if jt.flags.use_cuda else "cpu",
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
        safe_model = args.base_model.replace("/", "_")
        output_path = output_dir / f"eval_{safe_model}_{payload['timestamp']}.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)

    logger.info(f"Evaluation complete. Results saved to {output_path}")


if __name__ == "__main__":
    main()
