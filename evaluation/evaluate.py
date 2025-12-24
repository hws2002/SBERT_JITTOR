"""
Evaluate a trained SBERT checkpoint on STS-style datasets.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import jittor as jt
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.sbert_model import SBERTJittor
from utils.data_loader import collate_sts, prepare_sts_dataset

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
):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

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
            all_scores.extend(np.asarray(batch["scores"]).tolist())

    pearson_corr, _ = pearsonr(all_predictions, all_scores)
    spearman_corr, _ = spearmanr(all_predictions, all_scores)

    return {
        "pearson": pearson_corr * 100,
        "spearman": spearman_corr * 100,
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

    # Evaluation dataset selection
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

    return parser.parse_args()


def main():
    args = parse_args()

    if args.use_cuda and jt.has_cuda:
        jt.flags.use_cuda = 1
        logger.info("Using CUDA")
    else:
        jt.flags.use_cuda = 0
        logger.info("Using CPU")

    logger.info("Loading tokenizer (online)...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    logger.info("Loading checkpoint...")
    checkpoint = jt.load(args.checkpoint_path)
    model = SBERTJittor(
        encoder_name=args.base_model,
        pooling=args.pooling,
        head_type="none",
        checkpoint_path=None,
    )
    model.load_state_dict(checkpoint["model_state"])

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
        )
        results[dataset_name] = scores
        logger.info(
            f"{dataset_name} - Pearson: {scores['pearson']:.2f}, Spearman: {scores['spearman']:.2f}"
        )

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
