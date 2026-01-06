"""
Explicit STS evaluation script
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import jittor as jt
from jittor.dataset import DataLoader
from transformers import AutoTokenizer
from scipy.stats import pearsonr, spearmanr

from model.sbert_model import SBERTJittor
from utils.data_loader import prepare_sts_dataset, collate_sts
from utils.jt_utils import _to_jittor_batch, setup_device


def parse_args():
    parser = argparse.ArgumentParser(description="Explicit STS evaluation")
    parser.add_argument("--data_dir", default="./data", help="Dataset root")
    parser.add_argument("--dataset", default="STS-B", help="Dataset name")
    parser.add_argument("--split", default="test", help="Split name")
    parser.add_argument("--repo_id", default="Kyle-han/roberta-base-nli-mean-tokens",
                        help="HF repo id containing tokenizer + Jittor checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--cache_dir", default=None, help="Cache dir for tokenized data")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite cached tokenization")
    parser.add_argument("--tokenize_batch_size", type=int, default=1024, help="Tokenize batch size")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")
    return parser.parse_args()


def main():
    args = parse_args()

    setup_device(args.use_cuda)

    model, tokenizer, _ = SBERTJittor.from_pretrained(
        args.repo_id,
        return_tokenizer=True,
    )

    sts_dataset = prepare_sts_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        split=args.split,
        tokenizer=tokenizer,
        max_length=args.max_length,
        cache_dir=args.cache_dir,
        overwrite_cache=args.overwrite_cache,
        tokenize_batch_size=args.tokenize_batch_size,
    )

    sts_loader = DataLoader(
        sts_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_batch=collate_sts,
    )

    all_preds = []
    all_scores = []
    model.eval()

    with jt.no_grad():
        for batch in sts_loader:
            jt_batch = _to_jittor_batch(batch, for_sts=True)
            emb_a = model.encode(
                jt_batch["input_ids_a"],
                jt_batch["attention_mask_a"],
                jt_batch.get("token_type_ids_a"),
            )
            emb_b = model.encode(
                jt_batch["input_ids_b"],
                jt_batch["attention_mask_b"],
                jt_batch.get("token_type_ids_b"),
            )

            emb_a_np = emb_a.numpy()
            emb_b_np = emb_b.numpy()
            denom = np.linalg.norm(emb_a_np, axis=1) * np.linalg.norm(emb_b_np, axis=1) + 1e-9
            sim = np.sum(emb_a_np * emb_b_np, axis=1) / denom

            all_preds.extend(sim.tolist())
            all_scores.extend(jt_batch["scores"].numpy().reshape(-1).tolist())

    pearson, _ = pearsonr(all_preds, all_scores)
    spearman, _ = spearmanr(all_preds, all_scores)

    print({"pearson": pearson * 100, "spearman": spearman * 100})
    print("scores nan:", np.isnan(sts_dataset.arrays["scores"]).any())
    print("preds nan:", np.isnan(all_preds).any())


if __name__ == "__main__":
    main()
