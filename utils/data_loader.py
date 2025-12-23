"""
Cached NLI/STS dataset helpers with tokenizer-based preprocessing.
"""

import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from datasets import concatenate_datasets, load_from_disk


def _safe_model_id(model_name: str) -> str:
    return model_name.replace("/", "_")


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
            tokenized = load_from_disk(cache_path)
        else:
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
        return load_from_disk(cache_path)

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
