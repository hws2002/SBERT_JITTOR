"""
Evaluate BERT bi-encoder on STS datasets (cosine similarity).
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
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
}


def _resolve_model_dir(base_model: str, model_root: str) -> str:
    if os.path.isdir(base_model):
        return base_model
    mapped = MODEL_DIR_MAP.get(base_model, base_model)
    candidate = os.path.join(model_root, mapped)
    if os.path.isdir(candidate):
        return candidate
    return base_model


def evaluate_bi_encoder(
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
):
    from scipy.stats import pearsonr, spearmanr

    start_time = time.time()
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate BERT bi-encoder on STS datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base_model", type=str, default="bert-base-uncased",
                        choices=["bert-base-uncased", "bert-large-uncased"],
                        help="Base encoder model name")
    parser.add_argument("--model_root", type=str, default="./hf",
                        help="Root directory containing local models/tokenizers")
    parser.add_argument("--encoder_checkpoint", type=str, default=None,
                        help="Optional pretrained encoder checkpoint (.bin/.pt)")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "cls"],
                        help="Pooling strategy")

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
                        help="Path to write JSON results (default: ./result/eval_bert_bi_<timestamp>.json)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.use_cuda and jt.has_cuda:
        jt.flags.use_cuda = 1
        logger.info("Using CUDA")
    else:
        jt.flags.use_cuda = 0
        logger.info("Using CPU")

    model_dir = _resolve_model_dir(args.base_model, args.model_root)
    logger.info(f"Loading tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    model = SBERTJittor(
        encoder_name=args.base_model,
        pooling=args.pooling,
        head_type="none",
        checkpoint_path=args.encoder_checkpoint,
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
        scores = evaluate_bi_encoder(
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
        )
        results[dataset_name] = scores
        logger.info(
            f"{dataset_name} - Pearson: {scores['pearson']:.2f}, Spearman: {scores['spearman']:.2f}"
        )

    avg_pearson = sum(v["pearson"] for v in results.values()) / max(len(results), 1)
    avg_spearman = sum(v["spearman"] for v in results.values()) / max(len(results), 1)

    payload = {
        "model_info": {
            "mode": "bi",
            "model_name": args.base_model,
            "pooling": args.pooling,
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
        output_path = output_dir / f"eval_bert_bi_{payload['timestamp']}.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)

    logger.info(f"Evaluation complete. Results saved to {output_path}")


if __name__ == "__main__":
    main()
