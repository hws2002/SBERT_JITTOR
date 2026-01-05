"""
Shared training helpers for tokenizer, cache/output paths, and checkpoints.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

HF_DIR = "./hf"
MODEL_DIR_MAP: Dict[str, str] = {
    "bert-large-uncased": "hf_bert_large",
    "bert-base-uncased": "hf_bert_base",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
}


def safe_model_name(base_model: str) -> str:
    return base_model.replace("/", "_")


def resolve_cache_dir(data_dir: str, cache_dir: Optional[str]) -> str:
    return cache_dir if cache_dir else os.path.join(data_dir, "_cache")


def resolve_output_dir(
    output_dir: Optional[str],
    task: str,
    base_model: str,
    suffix: Optional[str] = None,
) -> str:
    if output_dir:
        return os.path.join(output_dir, suffix) if suffix else output_dir
    model_name = base_model.replace("/", "-")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"checkpoints/{task}_{model_name}-{timestamp}"


def resolve_tokenizer_source(
    base_model: str,
    tokenizer_dir: Optional[str] = None,
    encoder_checkpoint: Optional[str] = None,
    hf_dir: str = HF_DIR,
    model_dir_map: Optional[Dict[str, str]] = None,
) -> str:
    model_dir_map = model_dir_map or MODEL_DIR_MAP
    candidates = []
    if tokenizer_dir:
        candidates.append(tokenizer_dir)
    candidates.append(base_model)
    mapped = model_dir_map.get(base_model, base_model)
    candidates.append(os.path.join(hf_dir, mapped))
    if encoder_checkpoint and os.path.isfile(encoder_checkpoint):
        candidates.append(os.path.dirname(encoder_checkpoint))
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    raise ValueError(
        "Expected local model directory for tokenizer (base_model/tokenizer_dir/encoder_checkpoint dir)."
    )


def checkpoint_path(output_dir: str | Path, base_model: str, tag: str = "best") -> Path:
    safe_model = safe_model_name(base_model)
    if tag == "best":
        name = f"{safe_model}_best.pkl"
    else:
        name = f"{safe_model}_{tag}.pkl"
    return Path(output_dir) / name
