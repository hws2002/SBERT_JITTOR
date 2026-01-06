"""
Local dataset helpers without Hugging Face datasets.

Expected layout under data_dir:
- SNLI/snli_1.0_train.jsonl, snli_1.0_dev.jsonl, snli_1.0_test.jsonl
- MultiNLI/multinli_1.0_train.jsonl, multinli_1.0_dev_matched.jsonl, multinli_1.0_dev_mismatched.jsonl
- STS-B/train.tsv, STS-B/dev.tsv, STS-B/test.tsv
- STS-12..16/test.tsv
- SICKR/test.tsv
- MR/train.tsv, MR/validation.tsv, MR/test.tsv
- SST-2/train.tsv, SST-2/dev.tsv, SST-2/test.tsv
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from jittor.dataset import Dataset as JtDataset


def _safe_model_id(model_name: str) -> str:
    base_name = Path(model_name).name
    return base_name.replace("/", "_")


def _cache_dir_or_default(data_dir: str, cache_dir: str | None) -> str:
    return cache_dir if cache_dir else os.path.join(data_dir, "_cache")


def _cache_path(cache_dir: str, dataset_name: str, split: str, model_name: str, max_length: int) -> str:
    model_id = _safe_model_id(model_name)
    name = f"{dataset_name}_{split}_{model_id}_len{max_length}.npz"
    return os.path.join(cache_dir, name)


class NumpyDictDataset(JtDataset):
    def __init__(self, arrays: Dict[str, np.ndarray]):
        super().__init__()
        self.arrays = arrays
        first = next(iter(arrays.values()))
        self.set_attrs(total_len=len(first))
        self.collate_batch = None

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.arrays.items()}

    def __len__(self):
        return len(next(iter(self.arrays.values())))


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _save_npz(path: str, arrays: Dict[str, np.ndarray]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _read_tsv(path: str) -> Tuple[List[str], List[List[str]]]:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="	")
        rows = list(reader)
    if not rows:
        return [], []
    header = rows[0]
    has_header = any("sentence" in col.lower() or "label" in col.lower() or "score" in col.lower() for col in header)
    if has_header:
        return header, rows[1:]
    return [], rows


def _read_csv(path: str) -> Tuple[List[str], List[List[str]]]:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=",")
        rows = list(reader)
    if not rows:
        return [], []
    header = rows[0]
    has_header = any("sentence" in col.lower() or "label" in col.lower() or "score" in col.lower() for col in header)
    if has_header:
        return header, rows[1:]
    return [], rows


def _resolve_split_file(root: str, split: str, candidates: List[str]) -> str:
    for name in candidates:
        path = os.path.join(root, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Missing split '{split}' under {root}. Tried: {candidates}")


def _tokenize_pairs(
    tokenizer,
    sentences_a: List[str],
    sentences_b: List[str],
    max_length: int,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    input_ids_a = []
    attention_mask_a = []
    input_ids_b = []
    attention_mask_b = []
    token_type_ids_a = []
    token_type_ids_b = []
    has_token_type = False

    for start in range(0, len(sentences_a), batch_size):
        end = start + batch_size
        batch_a = sentences_a[start:end]
        batch_b = sentences_b[start:end]

        enc_a = tokenizer(
            batch_a,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        enc_b = tokenizer(
            batch_b,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        input_ids_a.append(np.asarray(enc_a["input_ids"], dtype=np.int32))
        attention_mask_a.append(np.asarray(enc_a["attention_mask"], dtype=np.float32))
        input_ids_b.append(np.asarray(enc_b["input_ids"], dtype=np.int32))
        attention_mask_b.append(np.asarray(enc_b["attention_mask"], dtype=np.float32))

        if "token_type_ids" in enc_a:
            has_token_type = True
            token_type_ids_a.append(np.asarray(enc_a["token_type_ids"], dtype=np.int32))
        if "token_type_ids" in enc_b:
            has_token_type = True
            token_type_ids_b.append(np.asarray(enc_b["token_type_ids"], dtype=np.int32))

    arrays = {
        "input_ids_a": np.concatenate(input_ids_a, axis=0),
        "attention_mask_a": np.concatenate(attention_mask_a, axis=0),
        "input_ids_b": np.concatenate(input_ids_b, axis=0),
        "attention_mask_b": np.concatenate(attention_mask_b, axis=0),
    }
    if has_token_type:
        arrays["token_type_ids_a"] = np.concatenate(token_type_ids_a, axis=0) if token_type_ids_a else None
        arrays["token_type_ids_b"] = np.concatenate(token_type_ids_b, axis=0) if token_type_ids_b else None
        if arrays["token_type_ids_a"] is None or arrays["token_type_ids_b"] is None:
            arrays.pop("token_type_ids_a", None)
            arrays.pop("token_type_ids_b", None)
    return arrays


def _tokenize_single(
    tokenizer,
    sentences: List[str],
    max_length: int,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    input_ids = []
    attention_mask = []
    token_type_ids = []
    has_token_type = False

    for start in range(0, len(sentences), batch_size):
        end = start + batch_size
        batch = sentences[start:end]
        enc = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        input_ids.append(np.asarray(enc["input_ids"], dtype=np.int32))
        attention_mask.append(np.asarray(enc["attention_mask"], dtype=np.float32))
        if "token_type_ids" in enc:
            has_token_type = True
            token_type_ids.append(np.asarray(enc["token_type_ids"], dtype=np.int32))

    arrays = {
        "input_ids": np.concatenate(input_ids, axis=0),
        "attention_mask": np.concatenate(attention_mask, axis=0),
    }
    if has_token_type:
        arrays["token_type_ids"] = np.concatenate(token_type_ids, axis=0)
    return arrays


def _read_nli_jsonl(path: str) -> Tuple[List[str], List[str], List[int]]:
    sentences_a = []
    sentences_b = []
    labels = []
    label_map = {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2,
    }
    for row in _read_jsonl(path):
        label = row.get("gold_label", row.get("label"))
        if label in (None, "-", "-1"):
            continue
        if isinstance(label, str):
            if label not in label_map:
                continue
            label = label_map[label]
        try:
            label = int(label)
        except (TypeError, ValueError):
            continue
        if label < 0:
            continue
        sentences_a.append(row.get("sentence1") or row.get("premise"))
        sentences_b.append(row.get("sentence2") or row.get("hypothesis"))
        labels.append(int(label))
    return sentences_a, sentences_b, labels


def prepare_nli_dataset(
    data_dir: str,
    datasets: List[str],
    split: str,
    tokenizer,
    max_length: int,
    cache_dir: str | None,
    overwrite_cache: bool,
    tokenize_batch_size: int,
):
    cache_dir = _cache_dir_or_default(data_dir, cache_dir)
    all_arrays = []

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

    for dataset_name in datasets:
        actual_split = _map_split(dataset_name, split)
        cache_path = _cache_path(cache_dir, dataset_name, actual_split, tokenizer.name_or_path, max_length)
        if os.path.isfile(cache_path) and not overwrite_cache:
            arrays = _load_npz(cache_path)
            all_arrays.append(arrays)
            continue

        root = os.path.join(data_dir, dataset_name)
        if dataset_name.lower() in {"snli"}:
            split_file = {
                "train": "snli_1.0_train.jsonl",
                "validation": "snli_1.0_dev.jsonl",
                "test": "snli_1.0_test.jsonl",
            }[actual_split]
        elif dataset_name.lower() in {"multinli", "multi_nli", "multi-nli"}:
            split_file = {
                "train": "multinli_1.0_train.jsonl",
                "validation_matched": "multinli_1.0_dev_matched.jsonl",
                "validation_mismatched": "multinli_1.0_dev_mismatched.jsonl",
            }[actual_split]
        else:
            raise ValueError(f"Unsupported NLI dataset: {dataset_name}")

        path = os.path.join(root, split_file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing NLI file: {path}")

        sentences_a, sentences_b, labels = _read_nli_jsonl(path)
        arrays = _tokenize_pairs(tokenizer, sentences_a, sentences_b, max_length, tokenize_batch_size)
        arrays["labels"] = np.asarray(labels, dtype=np.int32)
        _save_npz(cache_path, arrays)
        all_arrays.append(arrays)

    if len(all_arrays) == 1:
        return NumpyDictDataset(all_arrays[0])

    merged = {}
    keys = all_arrays[0].keys()
    for key in keys:
        merged[key] = np.concatenate([arr[key] for arr in all_arrays], axis=0)
    return NumpyDictDataset(merged)


def _resolve_sts_file(data_dir: str, dataset_name: str, split: str) -> str:
    root = os.path.join(data_dir, dataset_name)
    if dataset_name.upper() == "STS-B":
        if split == "validation":
            return _resolve_split_file(root, split, ["dev.tsv", "validation.tsv"])
        if split == "test":
            return _resolve_split_file(root, split, ["test.tsv"])
        return _resolve_split_file(root, split, ["train.tsv"])

    if dataset_name.upper().startswith("STS-"):
        return _resolve_split_file(root, split, [f"{split}.tsv", "test.tsv"])

    if dataset_name.upper() in {"SICKR", "SICK-R"}:
        return _resolve_split_file(root, split, [f"{split}.tsv", "test.tsv"])

    return _resolve_split_file(root, split, [f"{split}.tsv"])


def _parse_sts_rows(header: List[str], rows: List[List[str]]) -> Tuple[List[str], List[str], List[float]]:
    if header:
        lower = [h.lower() for h in header]
        s1_idx = lower.index("sentence1") if "sentence1" in lower else None
        s2_idx = lower.index("sentence2") if "sentence2" in lower else None
        score_idx = None
        if "score" in lower:
            score_idx = lower.index("score")
        elif "label" in lower:
            score_idx = lower.index("label")
        if s1_idx is not None and s2_idx is not None and score_idx is not None:
            s1 = []
            s2 = []
            scores = []
            for row in rows:
                if len(row) <= max(s1_idx, s2_idx, score_idx):
                    continue
                try:
                    score = float(row[score_idx])
                except ValueError:
                    score = None
                    for value in reversed(row):
                        try:
                            score = float(value)
                            break
                        except ValueError:
                            continue
                if score is None:
                    continue
                s1.append(row[s1_idx])
                s2.append(row[s2_idx])
                scores.append(score)
            return s1, s2, scores

    s1 = []
    s2 = []
    scores = []
    for row in rows:
        if len(row) < 3:
            continue
        score = None
        try:
            score = float(row[2])
        except ValueError:
            for value in reversed(row):
                try:
                    score = float(value)
                    break
                except ValueError:
                    continue
        if score is None:
            continue
        s1.append(row[0])
        s2.append(row[1])
        scores.append(score)
    return s1, s2, scores


def prepare_sts_dataset(
    data_dir: str,
    dataset_name: str,
    split: str,
    tokenizer,
    max_length: int,
    cache_dir: str | None,
    overwrite_cache: bool,
    tokenize_batch_size: int,
):
    cache_dir = _cache_dir_or_default(data_dir, cache_dir)
    cache_path = _cache_path(cache_dir, dataset_name, split, tokenizer.name_or_path, max_length)
    if os.path.isfile(cache_path) and not overwrite_cache:
        arrays = _load_npz(cache_path)
        return NumpyDictDataset(arrays)

    path = _resolve_sts_file(data_dir, dataset_name, split)
    if path.endswith(".csv"):
        header, rows = _read_csv(path)
    else:
        header, rows = _read_tsv(path)

    sentences_a, sentences_b, scores = _parse_sts_rows(header, rows)
    arrays = _tokenize_pairs(tokenizer, sentences_a, sentences_b, max_length, tokenize_batch_size)
    scores_arr = np.asarray(scores, dtype=np.float32)
    # Drop NaN/inf and negative placeholder scores (e.g., GLUE test = -1).
    if scores_arr.size:
        mask = np.isfinite(scores_arr) & (scores_arr >= 0)
        if not np.all(mask):
            scores_arr = scores_arr[mask]
            arrays = {k: v[mask] for k, v in arrays.items()}
    if scores_arr.size == 0:
        raise ValueError(
            f"No valid STS scores found for {dataset_name}/{split}. "
            "If using GLUE STS-B, the test split has -1 labels; use validation."
        )
    arrays["scores"] = scores_arr
    _save_npz(cache_path, arrays)
    return NumpyDictDataset(arrays)


def prepare_text_classification_dataset(
    data_dir: str,
    dataset_name: str,
    split: str,
    tokenizer,
    max_length: int,
    cache_dir: str | None,
    overwrite_cache: bool,
    tokenize_batch_size: int,
):
    cache_dir = _cache_dir_or_default(data_dir, cache_dir)
    cache_path = _cache_path(cache_dir, dataset_name, split, tokenizer.name_or_path, max_length)
    if os.path.isfile(cache_path) and not overwrite_cache:
        arrays = _load_npz(cache_path)
        return NumpyDictDataset(arrays)

    root = os.path.join(data_dir, dataset_name)
    if split == "validation":
        path = _resolve_split_file(root, split, ["validation.tsv", "dev.tsv"])
    else:
        path = _resolve_split_file(root, split, [f"{split}.tsv"])

    header, rows = _read_tsv(path)
    text_idx = None
    label_idx = None
    if header:
        lower = [h.lower() for h in header]
        for candidate in ("sentence", "text"):
            if candidate in lower:
                text_idx = lower.index(candidate)
                break
        if "label" in lower:
            label_idx = lower.index("label")
    if text_idx is None:
        text_idx = 0
    if label_idx is None:
        label_idx = 1 if len(rows[0]) > 1 else None

    texts = [row[text_idx] for row in rows]
    labels = [int(row[label_idx]) if label_idx is not None and row[label_idx] != "" else -1 for row in rows]

    arrays = _tokenize_single(tokenizer, texts, max_length, tokenize_batch_size)
    arrays["labels"] = np.asarray(labels, dtype=np.int32)
    _save_npz(cache_path, arrays)
    return NumpyDictDataset(arrays)


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
