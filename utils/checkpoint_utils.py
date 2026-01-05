"""
Checkpoint helpers for training scripts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import jittor as jt

from utils.training_utils import checkpoint_path

logger = logging.getLogger(__name__)


def save_checkpoint_encoder_only(model, train_loss, optimizer, iteration, epoch, args, name="checkpoint"):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if name == "best":
        path = checkpoint_path(output_dir, args.base_model, "best")
    else:
        logger.info("Skipping non-best checkpoint save (encoder-only mode).")
        return None

    checkpoint = {
        "model_state": model.state_dict(),
        "base_model": args.base_model,
        "pooling": args.pooling,
    }

    jt.save(checkpoint, str(path))
    logger.info(f"Checkpoint saved: {path}")
    return path


def save_checkpoint_with_optimizer(model, optimizer, iteration, epoch, args, name="checkpoint"):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = checkpoint_path(output_dir, args.base_model, "best" if name == "best" else name)

    checkpoint = {
        "iteration": iteration,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "base_model": args.base_model,
        "pooling": args.pooling,
    }

    jt.save(checkpoint, str(path))
    logger.info(f"Checkpoint saved: {path}")
    return path


def save_encoder_only(model, args, output_dir: Path, name: str):
    payload = {
        "model_state": model.state_dict(),
        "base_model": args.base_model,
        "pooling": args.pooling,
    }
    path = output_dir / f"{name}.pkl"
    jt.save(payload, str(path))
    logger.info(f"Encoder-only checkpoint saved: {path}")
    return path


def load_training_checkpoint_encoder_only(model, train_loss, optimizer, checkpoint_path: str) -> tuple[int, int]:
    logger.info(f"Loading training checkpoint: {checkpoint_path}")
    checkpoint = jt.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    if "loss_state" in checkpoint:
        try:
            train_loss.load_state_dict(checkpoint["loss_state"])
        except Exception as exc:
            logger.warning(f"Partial loss_state load (expected): {exc}")
    if optimizer is not None and "optimizer_state" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        except Exception as exc:
            logger.warning(f"Failed to load optimizer state: {exc}")

    global_step = int(checkpoint.get("iteration", 0))
    start_epoch = max(int(checkpoint.get("epoch", 1)) - 1, 0)
    logger.info(f"Resuming from step {global_step}, epoch {start_epoch + 1}")
    return global_step, start_epoch


def load_training_checkpoint_with_optimizer(model, optimizer, checkpoint_path: str) -> tuple[int, int]:
    logger.info(f"Loading training checkpoint: {checkpoint_path}")
    checkpoint = jt.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        except Exception as exc:
            logger.warning(f"Failed to load optimizer state: {exc}")

    global_step = int(checkpoint.get("iteration", 0))
    start_epoch = max(int(checkpoint.get("epoch", 1)) - 1, 0)
    logger.info(f"Resuming from step {global_step}, epoch {start_epoch + 1}")
    return global_step, start_epoch
