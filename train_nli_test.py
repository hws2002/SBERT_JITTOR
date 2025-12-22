import argparse
from pathlib import Path

import jittor as jt
from jittor import nn
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.data_loader import load_nli_data
from train_nli import SBERTWithClassification, build_config, make_batch


def train_subset(args):
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

    if len(dataloader) == 0:
        raise RuntimeError("No data available. Check --data_dir and dataset names.")
    print(f"loading {datasets} successful")


    config = build_config(args.base_model)
    model = SBERTWithClassification(config, num_labels=args.num_labels)
    optimizer = nn.optim.Adam(model.parameters(), lr=args.lr)

    warmup_steps = max(int(args.max_batches * args.warmup_ratio), 1)
    global_step = 0

    model.train()
    iterator = tqdm(dataloader, total=min(args.max_batches, len(dataloader)), desc="Sanity Train")

    for step, batch in enumerate(iterator):
        if step >= args.max_batches:
            break

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

        iterator.set_postfix(loss=float(loss))

    model.eval()
    with jt.no_grad():
        eval_batch = make_batch(tokenizer, batch, args.max_length)
        logits = model(eval_batch)
        preds = nn.softmax(logits, dim=-1)
        print("\nSample probabilities:", preds.numpy()[:2])


def parse_args():
    parser = argparse.ArgumentParser(description="Quick SBERT Jittor sanity training")
    parser.add_argument("--base_model", default="bert-base-uncased")
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--data_dir", default=str("./data"))
    parser.add_argument("--datasets", nargs="+", default=["SNLI"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_labels", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_subset(args)
