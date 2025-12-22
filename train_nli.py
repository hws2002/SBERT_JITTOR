import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import jittor as jt
from jittor import nn
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.data_loader import load_nli_data
from model.sbert_model import SBERTWithClassification


def _convert_tensor(tensor, dtype: str = "float32"):
    np_val = tensor.detach().cpu().numpy()
    if dtype == "int32":
        np_val = np_val.astype("int32")
    return jt.array(np_val)


def tokenize_sentences(tokenizer, sentences: List[str], max_length: int):
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": jt.array(encoded["input_ids"].detach().cpu().numpy().astype("int32")),
        "attention_mask": jt.array(encoded["attention_mask"].detach().cpu().numpy().astype("float32")),
        "token_type_ids": jt.array(encoded["token_type_ids"].detach().cpu().numpy().astype("int32"))
        if "token_type_ids" in encoded
        else None,
    }


def make_batch(tokenizer, batch: Dict[str, Iterable[str]], max_length: int):
    enc_a = tokenize_sentences(tokenizer, batch["sentence1"], max_length)
    enc_b = tokenize_sentences(tokenizer, batch["sentence2"], max_length)
    labels = _convert_tensor(batch["labels"], dtype="int32")

    jt_batch = {
        "input_ids_a": enc_a["input_ids"],
        "attention_mask_a": enc_a["attention_mask"],
        "token_type_ids_a": enc_a["token_type_ids"],
        "input_ids_b": enc_b["input_ids"],
        "attention_mask_b": enc_b["attention_mask"],
        "token_type_ids_b": enc_b["token_type_ids"],
        "labels": labels,
    }
    return jt_batch


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    dataloader = load_nli_data(
        data_dir=args.data_dir,
        datasets=args.datasets,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=True,
        tokenizer=None,
        max_length=args.max_length,
    )

    total_steps = args.epochs * len(dataloader)
    if total_steps == 0:
        raise RuntimeError("No training data found. Check data_dir/datasets arguments.")

    model = SBERTWithClassification(
        encoder_name=args.base_model,
        pooling=args.pooling,
        num_labels=args.num_labels,
        checkpoint_path=args.encoder_checkpoint,
    )
    optimizer = nn.optim.Adam(model.parameters(), lr=args.lr)

    warmup_steps = max(int(total_steps * args.warmup_ratio), 1)
    global_step = 0

    model.train()
    for epoch in range(args.epochs):
        epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(epoch_iterator, 1):
            jt_batch = make_batch(tokenizer, batch, args.max_length)

            logits = model(jt_batch)
            loss = nn.cross_entropy_loss(logits, jt_batch["labels"])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step <= warmup_steps:
                lr_scale = global_step / warmup_steps
                optimizer.lr = args.lr * lr_scale

            if global_step % args.log_every == 0:
                epoch_iterator.set_postfix(loss=float(loss))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "sbert_jittor_nli.pkl"
    checkpoint = {
        "state_dict": model.state_dict(),
        "base_model": args.base_model,
        "pooling": args.pooling,
        "num_labels": args.num_labels,
        "encoder_checkpoint": args.encoder_checkpoint,
    }
    jt.save(checkpoint, str(ckpt_path))
    print(f"Checkpoint saved to {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SBERT Jittor on SNLI / MultiNLI")
    parser.add_argument("--base_model", default="bert-base-uncased", choices=["bert-base-uncased", "bert-large-uncased"])
    parser.add_argument("--pooling", default="mean", choices=["mean", "cls", "max"])
    parser.add_argument("--encoder_checkpoint", type=str, default=None, help="Optional HuggingFace checkpoint (.bin/.pt)")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--datasets", nargs="+", default=["SNLI", "MultiNLI"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--output_dir", default="./checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
