"""
Jittor runtime + batch helpers.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable

import numpy as np
import jittor as jt

logger = logging.getLogger(__name__)


def setup_device(use_cuda: bool) -> None:
    if use_cuda and jt.has_cuda:
        jt.flags.use_cuda = 1
        logger.info("Using CUDA")
    else:
        jt.flags.use_cuda = 0
        logger.info("Using CPU")


def _jt_array(data, dtype: str) -> jt.Var:
    if isinstance(data, jt.Var):
        return data if str(data.dtype) == dtype else data.astype(dtype)
    return jt.array(np.asarray(data, dtype=dtype))


def _to_jittor_batch(batch: Dict[str, Iterable], for_sts: bool = False) -> Dict[str, jt.Var]:
    """Convert a numpy batch to Jittor arrays."""
    out: Dict[str, jt.Var] = {
        "input_ids_a": _jt_array(batch["input_ids_a"], "int32"),
        "attention_mask_a": _jt_array(batch["attention_mask_a"], "float32"),
        "input_ids_b": _jt_array(batch["input_ids_b"], "int32"),
        "attention_mask_b": _jt_array(batch["attention_mask_b"], "float32"),
    }

    if "token_type_ids_a" in batch:
        out["token_type_ids_a"] = _jt_array(batch["token_type_ids_a"], "int32")
    if "token_type_ids_b" in batch:
        out["token_type_ids_b"] = _jt_array(batch["token_type_ids_b"], "int32")

    if for_sts:
        out["scores"] = _jt_array(batch["scores"], "float32")
    else:
        out["labels"] = _jt_array(batch["labels"], "int32")

    return out


def _to_jittor_batch_single(batch: Dict[str, Iterable]) -> Dict[str, jt.Var]:
    """Convert a single-sentence batch to Jittor arrays."""
    out: Dict[str, jt.Var] = {
        "input_ids": _jt_array(batch["input_ids"], "int32"),
        "attention_mask": _jt_array(batch["attention_mask"], "float32"),
        "labels": _jt_array(batch["labels"], "int32"),
    }
    if "token_type_ids" in batch:
        out["token_type_ids"] = _jt_array(batch["token_type_ids"], "int32")
    return out
