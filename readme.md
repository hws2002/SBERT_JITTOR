# SBERT-Jittor

Sentence-BERT implemented in Jittor with training, evaluation, and downstream demos.

## What this repo does

- Jittor-native BERT encoder + SBERT pooling (mean/cls/max).
- NLI training (SNLI + MultiNLI) and STS regression fine-tuning.
- Evaluation on STS datasets with Pearson/Spearman.
- Simple downstream classification demos (MR/SST).

## Project structure

- `model/`: Jittor implementations of BERT and SBERT.
- `losses/`: The 3 SBERT paper losses plus a custom complex softmax loss, with ablation study support.

## Datasets

Place raw datasets under `./data`:

```
data/
  SNLI/snli_1.0_{train,dev,test}.jsonl
  MultiNLI/multinli_1.0_{train,dev_matched,dev_mismatched}.jsonl
  STS-B/{train,dev,test}.tsv
  STS-12/test.tsv
  STS-13/test.tsv
  STS-14/test.tsv
  STS-15/test.tsv
  STS-16/test.tsv
  SICKR/test.tsv
  MR/{train,validation,test}.tsv
  SST-2/{train,dev,test}.tsv
```

or download via provided script:
```bash
python utils/download_data.py --data_dir ./data
```

## Installation

If you see import errors like `No module named 'model'`, install the repo as a package:

```bash
pip install -e .
```

This uses `pyproject.toml` in the repo root.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Tokenizer note:
- Recommended: pre-download a local tokenizer to reduce AutoTokenizer download time.
- Optional: if missing, AutoTokenizer will download from Hugging Face.

## Train commands

All training hyperparameters used here follow the original SBERT paper.

NLI (SNLI + MultiNLI):

```bash
python training/nli/train_nli.py bert-base-uncased \
  --data_dir ./data \
  --datasets SNLI MultiNLI \
  --pooling mean \
  --batch_size 16 \
  --eval_batch_size 32 \
  --epochs 1 \
  --max_length 128 \
  --use_cuda
```
The first argument (bert-base-uncased) is the encoder model name.  
You can change it to any other encoder model name. (In the same way as the STS training command)

STS :

```bash
python training/sts/train_sts.py bert-base-uncased \
  --data_dir ./data \
  --train_dataset STS-B \
  --train_split train \
  --eval_dataset STS-B \
  --eval_split validation \
  --test_dataset STS-B \
  --test_split test \
  --pooling mean \
  --batch_size 32 \
  --eval_batch_size 32 \
  --epochs 1 \
  --max_length 128 \
  --use_cuda
```

STS after NLI checkpoint:

```bash
python training/sts/train_sts.py bert-base-uncased \
  --data_dir ./data \
  --train_dataset STS-B \
  --train_split train \
  --eval_dataset STS-B \
  --eval_split validation \
  --test_dataset STS-B \
  --test_split test \
  --pooling mean \
  --batch_size 32 \
  --eval_batch_size 32 \
  --epochs 1 \
  --max_length 128 \
  --use_cuda \
  --start_from_checkpoints path/to/your/checkpoints/best.pkl
```

Provide a pretrained NLI checkpoint path via `--start_from_checkpoints`.

SentEval (MR / SST):

```bash
python training/mr/train_mr.py bert-base-uncased --data_dir ./data --pooling mean --use_cuda
python training/sst/train_sst.py bert-base-uncased --data_dir ./data --pooling mean --use_cuda
```

## Evaluation command

Evaluate a Jittor SBERT checkpoint on STS benchmarks:

```bash
python evaluation/sts/evaluate_sbert.py \
  --checkpoint_path ./checkpoints/best.pkl \
  --base_model bert-base-uncased \
  --datasets all
```

## SBERTJittor usage

### Basic usage patterns:


```python
from model.sbert_model import SBERTJittor

# 1) Basic SBERT (mean pooling)
model1 = SBERTJittor("bert-base-uncased", pooling="mean", head_type="none")
print(model1.output_dim)

# 2) RoBERTa SBERT
model2 = SBERTJittor("roberta-base", pooling="mean", head_type="none")
print(model2.output_dim)
print(model2.config.vocab_size, model2.config.max_position_embeddings)

# 3) Linear projection head
model3 = SBERTJittor("bert-base-uncased", pooling="mean", head_type="linear", output_dim=256)
print(model3.output_dim)

# 4) MLP projection head
model4 = SBERTJittor("bert-base-uncased", pooling="mean", head_type="mlp", output_dim=128, num_layers=2)
print(model4.output_dim)
```

Notes:
- `pooling` controls the sentence embedding strategy (e.g., mean/cls/max).
- `head_type` controls the projection head; use `none` for pure SBERT embeddings.

### Load from Hugging Face checkpoint:

```python
from model.sbert_model import SBERTJittor

model, tokenizer, repo_dir = SBERTJittor.from_pretrained(
    "Kyle-han/roberta-base-nli-mean-tokens",
    return_tokenizer=True,
)
```

This loads the HF repo, initializes the encoder, and returns a ready-to-use tokenizer.

Encoding:

```python
import jittor as jt

batch = tokenizer("hello world", return_tensors="np")
input_ids = jt.array(batch["input_ids"])
attention_mask = jt.array(batch["attention_mask"])
token_type_ids = jt.array(batch["token_type_ids"]) if "token_type_ids" in batch else None

emb = model.encode(input_ids, attention_mask, token_type_ids)
```

`emb` is a sentence embedding tensor with shape `[batch, dim]`.

## Demo notebooks

- `demo/general_use.ipynb`: basic SBERTJittor construction, HF loading, and encoding.
- `demo/evaluation.ipynb`: evaluate a pretrained SBERT on STS-B with Pearson/Spearman.
- `demo/downstream.ipynb`: attach a classifier head and test transfer on MR.
