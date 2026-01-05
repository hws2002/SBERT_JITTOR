"""
Dataset download script (via Hugging Face datasets) with local TSV/JSONL export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


def _write_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_tsv(path: Path, header: list[str], rows: Iterable[list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(str(x) for x in row) + "\n")


def download_snli(data_dir: str):
    ds = load_dataset("stanfordnlp/snli")
    out_root = Path(data_dir) / "SNLI"

    split_map = {
        "train": "snli_1.0_train.jsonl",
        "validation": "snli_1.0_dev.jsonl",
        "test": "snli_1.0_test.jsonl",
    }

    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    for split, name in split_map.items():
        rows = []
        for item in ds[split]:
            label = item.get("label")
            if label is None or label == -1:
                continue
            rows.append({
                "sentence1": item.get("premise"),
                "sentence2": item.get("hypothesis"),
                "gold_label": label_map.get(int(label), str(label)),
            })
        _write_jsonl(out_root / name, rows)
    print(f"SNLI saved to {out_root}")


def download_multinli(data_dir: str):
    ds = load_dataset("nyu-mll/multi_nli")
    out_root = Path(data_dir) / "MultiNLI"

    split_map = {
        "train": "multinli_1.0_train.jsonl",
        "validation_matched": "multinli_1.0_dev_matched.jsonl",
        "validation_mismatched": "multinli_1.0_dev_mismatched.jsonl",
    }

    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    for split, name in split_map.items():
        rows = []
        for item in ds[split]:
            label = item.get("label")
            if label is None or label == -1:
                continue
            rows.append({
                "sentence1": item.get("premise"),
                "sentence2": item.get("hypothesis"),
                "gold_label": label_map.get(int(label), str(label)),
            })
        _write_jsonl(out_root / name, rows)
    print(f"MultiNLI saved to {out_root}")


def download_glue(data_dir: str):
    out_root = Path(data_dir)

    stsb = load_dataset("glue", "stsb")
    for split in ["train", "validation", "test"]:
        rows = []
        for item in stsb[split]:
            score = item.get("label")
            if score is None:
                continue
            rows.append([item["sentence1"], item["sentence2"], float(score)])
        name = "dev.tsv" if split == "validation" else f"{split}.tsv"
        _write_tsv(out_root / "STS-B" / name, ["sentence1", "sentence2", "score"], rows)

    sst = load_dataset("glue", "sst2")
    for split in ["train", "validation", "test"]:
        rows = []
        for item in sst[split]:
            label = item.get("label", -1)
            rows.append([item["sentence"], int(label) if label is not None else -1])
        name = "dev.tsv" if split == "validation" else f"{split}.tsv"
        _write_tsv(out_root / "SST-2" / name, ["sentence", "label"], rows)

    print(f"GLUE STS-B/SST-2 saved to {out_root}")


def _download_sts_dataset(repo_id: str, out_dir: Path):
    ds = load_dataset(repo_id)
    split = "test" if "test" in ds else list(ds.keys())[0]
    rows = []
    for item in ds[split]:
        rows.append([item["sentence1"], item["sentence2"], float(item["score"])])
    _write_tsv(out_dir / "test.tsv", ["sentence1", "sentence2", "score"], rows)


def download_sts12_16(data_dir: str):
    base = Path(data_dir)
    mapping = {
        "STS-12": "mteb/sts12-sts",
        "STS-13": "mteb/sts13-sts",
        "STS-14": "mteb/sts14-sts",
        "STS-15": "mteb/sts15-sts",
        "STS-16": "mteb/sts16-sts",
    }
    for name, repo in mapping.items():
        _download_sts_dataset(repo, base / name)
    print(f"STS12-16 saved under {base}")


def download_sickr(data_dir: str):
    base = Path(data_dir) / "SICKR"
    _download_sts_dataset("mteb/sickr-sts", base)
    print(f"SICKR saved to {base}")


def download_mr(data_dir: str):
    ds = load_dataset("rotten_tomatoes")
    base = Path(data_dir) / "MR"
    for split in ["train", "validation", "test"]:
        rows = []
        for item in ds[split]:
            rows.append([item["text"], int(item["label"])])
        _write_tsv(base / f"{split}.tsv", ["text", "label"], rows)
    print(f"MR saved to {base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for SBERT (HF datasets -> local files)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory to save datasets")
    parser.add_argument("--nli_only", action="store_true",
                        help="Download only NLI datasets")
    parser.add_argument("--sts_only", action="store_true",
                        help="Download only STS datasets")
    parser.add_argument("--sst_only", action="store_true",
                        help="Download only SST-2 dataset")
    parser.add_argument("--mr_only", action="store_true",
                        help="Download only MR dataset")
    args = parser.parse_args()

    only_flags = [args.nli_only, args.sts_only, args.sst_only, args.mr_only]
    if any(only_flags):
        if args.nli_only:
            download_snli(args.data_dir)
            download_multinli(args.data_dir)
        if args.sts_only:
            download_glue(args.data_dir)
            download_sts12_16(args.data_dir)
            download_sickr(args.data_dir)
        if args.sst_only:
            download_glue(args.data_dir)
        if args.mr_only:
            download_mr(args.data_dir)
    else:
        download_snli(args.data_dir)
        download_multinli(args.data_dir)
        download_glue(args.data_dir)
        download_sts12_16(args.data_dir)
        download_sickr(args.data_dir)
        download_mr(args.data_dir)

    print("Done!")
