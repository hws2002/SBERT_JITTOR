"""
Train SBERT on NLI datasets (SNLI + MultiNLI) with evaluation on STS benchmark

The system trains BERT (or any transformer model) on the SNLI + MultiNLI (AllNLI) dataset
with cross-entropy loss. At every N training steps, the model is evaluated on the
STS benchmark dataset.

Usage:
    python train_nli.py

OR
    python train_nli.py bert-base-uncased --pooling mean --batch_size 16 --wandb

Inspired by: https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/nli/training_nli.py
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, List

import jittor as jt
from jittor import nn
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.data_loader import load_nli_data, load_sts_data
from model.sbert_model import SBERTWithClassification

# Set the log level to INFO
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def _convert_tensor(tensor, dtype: str = "float32"):
    """Convert PyTorch tensor to Jittor array"""
    np_val = tensor.detach().cpu().numpy()
    if dtype == "int32":
        np_val = np_val.astype("int32")
    return jt.array(np_val)


def tokenize_sentences(tokenizer, sentences: List[str], max_length: int):
    """Tokenize sentences and convert to Jittor arrays"""
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
    """Prepare batch for training"""
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


def evaluate_sts(model, tokenizer, data_dir, dataset_name='STS-B', split='validation', max_length=128):
    """
    Evaluate model on STS benchmark dataset

    Returns:
        dict with 'pearson' and 'spearman' correlation scores
    """
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    logger.info(f"Evaluating on {dataset_name} ({split})...")

    model.eval()

    # Load STS data
    dataloader = load_sts_data(
        data_dir=data_dir,
        dataset_name=dataset_name,
        split=split,
        batch_size=32,
        tokenizer=None,
        max_length=max_length
    )

    all_predictions = []
    all_scores = []

    with jt.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}", leave=False):
            sentences1 = batch['sentence1']
            sentences2 = batch['sentence2']
            scores = batch['scores']

            # Encode sentences
            enc_a = tokenize_sentences(tokenizer, sentences1, max_length)
            enc_b = tokenize_sentences(tokenizer, sentences2, max_length)

            # Get embeddings
            emb_a = model.encode(
                enc_a['input_ids'],
                enc_a['attention_mask'],
                enc_a.get('token_type_ids', None)
            )
            emb_b = model.encode(
                enc_b['input_ids'],
                enc_b['attention_mask'],
                enc_b.get('token_type_ids', None)
            )

            # Compute cosine similarity
            emb_a_np = emb_a.numpy()
            emb_b_np = emb_b.numpy()

            # Cosine similarity
            denom = np.linalg.norm(emb_a_np, axis=1) * np.linalg.norm(emb_b_np, axis=1) + 1e-9
            sim = np.sum(emb_a_np * emb_b_np, axis=1) / denom

            all_predictions.extend(sim.tolist())
            all_scores.extend(scores.cpu().numpy().tolist())

    # Compute correlations
    pearson_corr, _ = pearsonr(all_predictions, all_scores)
    spearman_corr, _ = spearmanr(all_predictions, all_scores)

    results = {
        'pearson': pearson_corr * 100,
        'spearman': spearman_corr * 100,
    }

    logger.info(f"{dataset_name} ({split}) - Pearson: {results['pearson']:.2f}, Spearman: {results['spearman']:.2f}")

    model.train()
    return results


def setup_device(use_cuda):
    """Setup Jittor device"""
    if use_cuda and jt.has_cuda:
        jt.flags.use_cuda = 1
        logger.info("Using CUDA")
    else:
        jt.flags.use_cuda = 0
        logger.info("Using CPU")


def setup_wandb(args):
    """Initialize Weights & Biases"""
    if not args.wandb:
        return None

    try:
        import wandb

        run_name = args.run_name if args.run_name else f"nli-{args.base_model.split('/')[-1]}-{args.pooling}"

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                'model': args.base_model,
                'pooling': args.pooling,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'max_length': args.max_length,
                'warmup_ratio': args.warmup_ratio,
                'epochs': args.epochs,
                'datasets': args.datasets,
                'eval_steps': args.eval_steps,
            }
        )
        logger.info(f"W&B initialized: {args.wandb_project}/{run_name}")
        return wandb
    except ImportError:
        logger.warning("wandb not installed. Skipping W&B logging.")
        return None


def save_checkpoint(model, optimizer, iteration, epoch, args, name='checkpoint'):
    """Save model checkpoint"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / f'{name}_step{iteration}.pkl'

    checkpoint = {
        'iteration': iteration,
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'base_model': args.base_model,
        'pooling': args.pooling,
        'num_labels': args.num_labels,
    }

    jt.save(checkpoint, str(checkpoint_path))
    logger.info(f"Checkpoint saved: {checkpoint_path}")

    return checkpoint_path


def train(args):
    """Main training function"""

    logger.info("="*70)
    logger.info("SBERT NLI Training")
    logger.info("="*70)
    logger.info(f"Model: {args.base_model}")
    logger.info(f"Pooling: {args.pooling}")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Eval steps: {args.eval_steps}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info("="*70)

    # Setup device
    setup_device(args.use_cuda)

    # Setup wandb
    wandb = setup_wandb(args)

    # 1. Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    # 2. Load NLI training data
    logger.info("Loading NLI training data...")
    train_dataloader = load_nli_data(
        data_dir=args.data_dir,
        datasets=args.datasets,
        split='train',
        batch_size=args.batch_size,
        shuffle=True,
        tokenizer=None,
        max_length=args.max_length,
    )

    total_steps = args.epochs * len(train_dataloader)
    if total_steps == 0:
        raise RuntimeError("No training data found. Check data_dir/datasets arguments.")

    # Override total_steps if max_steps is set
    if args.max_steps > 0:
        total_steps = args.max_steps
        logger.info(f"Max training steps (limited): {total_steps}")
    else:
        logger.info(f"Total training steps: {total_steps}")

    # 3. Create model with SoftmaxLoss (Cross-entropy)
    logger.info("Initializing SBERT model...")
    model = SBERTWithClassification(
        encoder_name=args.base_model,
        pooling=args.pooling,
        num_labels=args.num_labels,
        checkpoint_path=args.encoder_checkpoint,
    )
    logger.info(f"Model embedding dimension: {model.sbert.output_dim}")

    # 4. Setup optimizer
    optimizer = nn.Adam(model.parameters(), lr=args.lr)
    warmup_steps = max(int(total_steps * args.warmup_ratio), 1)
    logger.info(f"Warmup steps: {warmup_steps}")

    # 5. Evaluation before training
    if args.skip_initial_eval:
        logger.info("\nSkipping initial evaluation (--skip_initial_eval)")
        best_spearman = 0.0
    else:
        logger.info("\nEvaluation before training:")
        eval_results_before = evaluate_sts(
            model=model.sbert,
            tokenizer=tokenizer,
            data_dir=args.data_dir,
            dataset_name='STS-B',
            split='validation',
            max_length=args.max_length
        )
        best_spearman = eval_results_before['spearman']

    # 6. Training loop
    logger.info("\nStarting training...")
    logger.info("="*70)

    global_step = 0
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    model.train()

    for epoch in range(args.epochs):
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(epoch_iterator, 1):
            # Prepare batch
            jt_batch = make_batch(tokenizer, batch, args.max_length)
            labels = jt_batch["labels"]

            # Forward pass
            logits = model(jt_batch)
            loss = nn.cross_entropy_loss(logits, labels)

            # Backward pass
            optimizer.step(loss)

            # Update learning rate (warmup)
            global_step += 1
            if global_step <= warmup_steps:
                lr_scale = global_step / warmup_steps
                current_lr = args.lr * lr_scale
                optimizer.lr = current_lr
            else:
                current_lr = args.lr

            # Compute accuracy
            predictions = jt.argmax(logits, dim=1)[0]
            correct = jt.sum(predictions == labels).item()
            batch_size = labels.shape[0]

            # Update statistics
            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += batch_size

            # Update progress bar
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples * 100
            epoch_iterator.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2f}%',
                'lr': f'{current_lr:.2e}'
            })

            # Log to console and wandb
            if global_step % args.log_steps == 0:
                logger.info(
                    f"Step {global_step}/{total_steps} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Acc: {accuracy:.2f}% | "
                    f"LR: {current_lr:.2e}"
                )

                if wandb:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/avg_loss': avg_loss,
                        'train/accuracy': accuracy,
                        'train/lr': current_lr,
                        'step': global_step,
                    })

            # Evaluate on STS benchmark
            if global_step % args.eval_steps == 0:
                eval_results = evaluate_sts(
                    model=model.sbert,
                    tokenizer=tokenizer,
                    data_dir=args.data_dir,
                    dataset_name='STS-B',
                    split='validation',
                    max_length=args.max_length
                )

                if wandb:
                    wandb.log({
                        'eval/pearson': eval_results['pearson'],
                        'eval/spearman': eval_results['spearman'],
                        'step': global_step,
                    })

                # Save best model
                if eval_results['spearman'] > best_spearman:
                    best_spearman = eval_results['spearman']
                    save_checkpoint(model, optimizer, global_step, epoch+1, args, name='best')

            # Save checkpoint periodically
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_checkpoint(model, optimizer, global_step, epoch+1, args, name='checkpoint')

            # Check if max_steps reached
            if args.max_steps > 0 and global_step >= args.max_steps:
                logger.info(f"\nReached max_steps ({args.max_steps}). Stopping training.")
                break

        # Check if max_steps reached (break outer loop)
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

        # Epoch summary
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100

        logger.info(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        logger.info(f"  Average Loss: {avg_loss:.4f}")
        logger.info(f"  Accuracy: {accuracy:.2f}%")
        logger.info(f"  Total Steps: {global_step}")

        if wandb:
            wandb.log({
                'epoch/loss': avg_loss,
                'epoch/accuracy': accuracy,
                'epoch': epoch + 1,
            })

        # Reset statistics
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

    # 7. Final evaluation on test set
    logger.info("\n" + "="*70)
    logger.info("Final Evaluation on STS Benchmark Test Set")
    logger.info("="*70)

    test_results = evaluate_sts(
        model=model.sbert,
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        dataset_name='STS-B',
        split='test',
        max_length=args.max_length
    )

    if wandb:
        wandb.log({
            'test/pearson': test_results['pearson'],
            'test/spearman': test_results['spearman'],
        })

    # 8. Save final model
    logger.info("\nSaving final model...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_path = output_dir / "final" / f"sbert_{args.base_model.replace('/', '_')}_{args.pooling}.pkl"
    final_path.parent.mkdir(exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "base_model": args.base_model,
        "pooling": args.pooling,
        "num_labels": args.num_labels,
        "test_results": test_results,
    }
    jt.save(checkpoint, str(final_path))

    logger.info("\n" + "="*70)
    logger.info("Training completed!")
    logger.info(f"Best Spearman score (dev): {best_spearman:.2f}")
    logger.info(f"Final Spearman score (test): {test_results['spearman']:.2f}")
    logger.info(f"Final model saved: {final_path}")
    logger.info("="*70)

    if wandb:
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SBERT on NLI datasets with STS evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument("base_model", nargs="?", default="bert-base-uncased",
                        help="Base encoder model (bert-base-uncased, bert-large-uncased, roberta-base, roberta-large)")
    parser.add_argument("--pooling", default="mean",
                        choices=["mean", "cls", "max"],
                        help="Pooling strategy")
    parser.add_argument("--encoder_checkpoint", type=str, default=None,
                        help="Optional pretrained encoder checkpoint (.bin/.pt)")
    parser.add_argument("--num_labels", type=int, default=3,
                        help="Number of NLI classification labels")

    # Data arguments
    parser.add_argument("--data_dir", default="./data",
                        help="Directory containing datasets")
    parser.add_argument("--datasets", nargs="+", default=["SNLI", "MultiNLI"],
                        help="NLI datasets to use")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum training steps (overrides epochs, -1 for no limit)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio (fraction of total steps)")

    # Device
    parser.add_argument("--use_cuda", action="store_true",
                        help="Use CUDA if available")

    # Logging and evaluation
    parser.add_argument("--log_steps", type=int, default=100,
                        help="Log metrics every N steps")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Evaluate on STS benchmark every N steps")
    parser.add_argument("--skip_initial_eval", action="store_true",
                        help="Skip evaluation before training (faster startup)")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every N steps (0 to disable)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for checkpoints")

    # WandB
    parser.add_argument("--wandb", action="store_true",
                        help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="sbert-nli-training",
                        help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name (default: auto-generated)")

    args = parser.parse_args()

    # Set default output_dir if not specified
    if args.output_dir is None:
        model_name = args.base_model.replace("/", "-")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = f"output/training_nli_{model_name}-{timestamp}"

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
