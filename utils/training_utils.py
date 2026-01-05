"""
Shared training helpers for tokenizer, cache/output paths, and checkpoints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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
    return base_model


def checkpoint_path(output_dir: str | Path, base_model: str, tag: str = "best") -> Path:
    safe_model = safe_model_name(base_model)
    if tag == "best":
        name = f"{safe_model}_best.pkl"
    else:
        name = f"{safe_model}_{tag}.pkl"
    return Path(output_dir) / name


@dataclass(frozen=True)
class TrainConfig:
    base_model: str
    data_dir: str
    cache_dir: str
    tokenizer_path: str
    pretrained_checkpoint_path: Optional[str]
    checkpoint_output_path: str

    @classmethod
    def from_args(
        cls,
        args,
        task: str,
        output_suffix: Optional[str] = None,
    ) -> "TrainConfig":
        output_dir = getattr(args, "output_dir", None)
        cache_dir = resolve_cache_dir(args.data_dir, args.cache_dir)
        tokenizer_path = resolve_tokenizer_source(
            args.base_model,
            tokenizer_dir=getattr(args, "tokenizer_dir", None),
            encoder_checkpoint=getattr(args, "encoder_checkpoint", None),
        )
        checkpoint_output_path = resolve_output_dir(
            output_dir,
            task,
            args.base_model,
            suffix=output_suffix if output_dir else None,
        )
        pretrained_checkpoint_path = getattr(args, "encoder_checkpoint", None)
        return cls(
            base_model=args.base_model,
            data_dir=args.data_dir,
            cache_dir=cache_dir,
            tokenizer_path=tokenizer_path,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
            checkpoint_output_path=checkpoint_output_path,
        )
