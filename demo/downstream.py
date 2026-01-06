import math
import os
import warnings
from pathlib import Path

import jittor as jt
import numpy as np
from jittor import nn
from jittor.dataset import DataLoader
from tqdm import tqdm

from model.sbert_model import SBERTJittor
from utils.data_loader import prepare_text_classification_dataset, collate_text_classification
from utils.jt_utils import _to_jittor_batch_single
from utils.training_utils import setup_device


def main():
    # Optional HF cache + warning control
    os.environ.setdefault("HF_HOME", "./.hf_cache")
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings(
        "ignore",
        message="Using `TRANSFORMERS_CACHE` is deprecated",
        category=FutureWarning,
    )

    # 1) Load config and device
    data_dir = "./data"
    batch_size = 32
    max_length = 128
    repo_id = "Kyle-han/roberta-base-nli-mean-tokens"
    num_workers = 4

    setup_device(True)

    # 2) Load pretrained SBERT + tokenizer from HF
    model, tokenizer, _ = SBERTJittor.from_pretrained(
        repo_id,
        return_tokenizer=True,
    )

    # 3) Load datasets
    train_ds = prepare_text_classification_dataset(
        data_dir=data_dir,
        dataset_name="MR",
        split="train",
        tokenizer=tokenizer,
        max_length=max_length,
        cache_dir=None,
        overwrite_cache=False,
        tokenize_batch_size=1024,
    )
    val_ds = prepare_text_classification_dataset(
        data_dir=data_dir,
        dataset_name="MR",
        split="validation",
        tokenizer=tokenizer,
        max_length=max_length,
        cache_dir=None,
        overwrite_cache=False,
        tokenize_batch_size=1024,
    )
    test_ds = prepare_text_classification_dataset(
        data_dir=data_dir,
        dataset_name="MR",
        split="test",
        tokenizer=tokenizer,
        max_length=max_length,
        cache_dir=None,
        overwrite_cache=False,
        tokenize_batch_size=1024,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        collate_batch=collate_text_classification,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_batch=collate_text_classification,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_batch=collate_text_classification,
    )

    # 4) Train classifier head (SBERT frozen)
    for param in model.parameters():
        param.stop_grad()

    clf = nn.Linear(model.output_dim, 2)
    optimizer = nn.Adam(clf.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    num_epochs = 4
    steps_per_epoch = math.ceil(len(train_ds) / batch_size)
    for epoch in range(1, num_epochs + 1):
        model.train()
        clf.train()
        for step, batch in enumerate(
            tqdm(train_loader, total=steps_per_epoch, desc=f"MR train (epoch {epoch})"),
            1,
        ):
            jt_batch = _to_jittor_batch_single(batch)
            reps = model.encode(
                jt_batch["input_ids"],
                jt_batch["attention_mask"],
                jt_batch.get("token_type_ids"),
            )
            logits = clf(reps)
            loss = loss_fn(logits, jt_batch["labels"])
            optimizer.step(loss)
            if step >= steps_per_epoch:
                break

    # 5) Evaluate on validation + test set
    def eval_loop(loader, name, dataset_len):
        model.eval()
        clf.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        total_batches = math.ceil(dataset_len / batch_size)
        with jt.no_grad():
            for batch in tqdm(loader, total=total_batches, desc=f"{name} eval"):
                jt_batch = _to_jittor_batch_single(batch)
                reps = model.encode(
                    jt_batch["input_ids"],
                    jt_batch["attention_mask"],
                    jt_batch.get("token_type_ids"),
                )
                logits = clf(reps)
                loss = loss_fn(logits, jt_batch["labels"])
                preds = jt.argmax(logits, dim=1)[0]
                total_correct += jt.sum(preds == jt_batch["labels"]).item()
                total_samples += jt_batch["labels"].shape[0]
                total_loss += loss.item() * jt_batch["labels"].shape[0]
        avg_loss = total_loss / max(total_samples, 1)
        acc = total_correct / max(total_samples, 1) * 100
        print({name: {"loss": avg_loss, "accuracy": acc}})

    eval_loop(val_loader, "MR validation", len(val_ds))
    eval_loop(test_loader, "MR test", len(test_ds))


if __name__ == "__main__":
    main()
