"""
STS evaluation helpers shared by training/evaluation entry points.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import jittor as jt
from jittor.dataset import DataLoader
from tqdm import tqdm

from utils.data_loader import prepare_sts_dataset, collate_sts
from utils.jt_utils import _to_jittor_batch
from utils.training_utils import safe_model_name

logger = logging.getLogger(__name__)

STS_DATASETS = ["STS-12", "STS-13", "STS-14", "STS-15", "STS-16", "STS-B", "SICKR"]


def evaluate_sts(
    model,
    dataloader,
    normalize_scores: bool = False,
    score_scale: float = 1.0,
    desc: str = "Evaluating STS",
    total_batches: Optional[int] = None,
) -> Dict[str, float]:
    from scipy.stats import pearsonr, spearmanr

    model.eval()
    all_predictions: List[float] = []
    all_scores: List[float] = []

    if total_batches is None:
        try:
            dataset_len = len(getattr(dataloader, "dataset", []))
            batch_size = getattr(dataloader, "batch_size", None)
            if batch_size:
                total_batches = int(np.ceil(dataset_len / batch_size))
        except Exception:
            total_batches = None

    with jt.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False, total=total_batches):
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
            if emb_a_np.ndim == 1:
                emb_a_np = emb_a_np.reshape(1, -1)
            if emb_b_np.ndim == 1:
                emb_b_np = emb_b_np.reshape(1, -1)
            denom = np.linalg.norm(emb_a_np, axis=1) * np.linalg.norm(emb_b_np, axis=1) + 1e-9
            sim = np.sum(emb_a_np * emb_b_np, axis=1) / denom

            all_predictions.extend(sim.tolist())
            scores_np = jt_batch["scores"].numpy().reshape(-1)
            if normalize_scores and score_scale != 0:
                scores_np = scores_np / score_scale
            all_scores.extend(scores_np.tolist())

            sim_np = np.asarray(sim).reshape(-1)
            if sim_np.shape[0] != scores_np.shape[0]:
                logger.warning(
                    f"STS eval length mismatch (pred {sim_np.shape[0]} vs scores {scores_np.shape[0]}). "
                    "Recomputing per-sample."
                )
                preds = []
                for idx in range(scores_np.shape[0]):
                    emb_a_i = model.encode(
                        jt_batch["input_ids_a"][idx:idx + 1],
                        jt_batch["attention_mask_a"][idx:idx + 1],
                        jt_batch.get("token_type_ids_a", None)[idx:idx + 1]
                        if "token_type_ids_a" in jt_batch else None,
                    )
                    emb_b_i = model.encode(
                        jt_batch["input_ids_b"][idx:idx + 1],
                        jt_batch["attention_mask_b"][idx:idx + 1],
                        jt_batch.get("token_type_ids_b", None)[idx:idx + 1]
                        if "token_type_ids_b" in jt_batch else None,
                    )
                    emb_a_i = emb_a_i.numpy().reshape(1, -1)
                    emb_b_i = emb_b_i.numpy().reshape(1, -1)
                    denom_i = np.linalg.norm(emb_a_i, axis=1) * np.linalg.norm(emb_b_i, axis=1) + 1e-9
                    sim_i = np.sum(emb_a_i * emb_b_i, axis=1) / denom_i
                    preds.append(float(sim_i[0]))
                sim_np = np.asarray(preds)
                all_predictions[-scores_np.shape[0]:] = sim_np.tolist()

    pearson_corr, _ = pearsonr(all_predictions, all_scores)
    spearman_corr, _ = spearmanr(all_predictions, all_scores)

    model.train()
    return {
        "pearson": pearson_corr * 100,
        "spearman": spearman_corr * 100,
    }


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
    dataset_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    dataset_names = dataset_names or STS_DATASETS
    for dataset_name in dataset_names:
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
            collate_batch=collate_sts,
        )
        logger.info(f"Evaluating {dataset_name} ({split})...")
        scores = evaluate_sts(model=model, dataloader=sts_dataloader, desc=f"Evaluating {dataset_name}")
        results[dataset_name] = {
            "split": split,
            "pearson": scores["pearson"],
            "spearman": scores["spearman"],
            "n_samples": len(sts_dataset),
        }
    return results


def save_eval_results_multi(
    results: Dict[str, Dict[str, float]],
    args,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = safe_model_name(args.base_model)
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


def save_eval_results_single(results: Dict[str, float], args, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = safe_model_name(args.base_model)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_sts_{safe_model}_{timestamp}.json"

    payload = {
        "model_info": {
            "type": "sbert_jittor",
            "model_name": safe_model,
            "max_length": args.max_length,
        },
        "evaluation": {
            "dataset": args.test_dataset,
            "split": args.test_split,
            "batch_size": args.eval_batch_size,
            "device": "cuda" if jt.flags.use_cuda else "cpu",
        },
        "results": results,
        "timestamp": timestamp,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    logger.info(f"Saved evaluation results to {output_path}")
    return output_path
