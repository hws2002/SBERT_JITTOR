# SBERT-Jittor

Sentence-BERT implemented in Jittor with training, evaluation, and downstream demos.

## What this repo does

- Jittor-native BERT encoder + SBERT pooling (mean/cls/max).
- NLI training (SNLI + MultiNLI) and STS regression fine-tuning.
- Evaluation on STS datasets with Pearson/Spearman.
- Simple downstream classification demos (MR/SST).

## Train commands

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

STS (regression):

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

MR / SST:

```bash
python training/mr/train_mr.py bert-base-uncased --data_dir ./data --pooling mean --use_cuda
python training/sst/train_sst.py bert-base-uncased --data_dir ./data --pooling mean --use_cuda
```

### Local dataset layout

Place raw datasets under `./data` (no Hugging Face datasets dependency):

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

Use the downloader to fetch via Hugging Face datasets and export to local files:

```bash
python utils/download_data.py --data_dir ./data
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

Load from Hugging Face:

```python
from model.sbert_model import SBERTJittor

model, tokenizer, repo_dir = SBERTJittor.from_pretrained(
    "Kyle-han/roberta-base-nli-mean-tokens",
    return_tokenizer=True,
)
```

Load with a pretrained flag:

```python
from model.sbert_model import SBERTJittor

model = SBERTJittor(
    pretrained=True,
    model_id="Kyle-han/roberta-base-nli-mean-tokens",
    pooling="mean",
)
```

Encoding:

```python
import jittor as jt

batch = tokenizer("hello world", return_tensors="np")
input_ids = jt.array(batch["input_ids"])
attention_mask = jt.array(batch["attention_mask"])
token_type_ids = jt.array(batch["token_type_ids"]) if "token_type_ids" in batch else None

emb = model.encode(input_ids, attention_mask, token_type_ids)
```

## Demo notebooks

- `demo/fine-tune.ipynb`: NLI/STS fine-tuning walkthrough
- `demo/evaluation.ipynb`: STS evaluation workflow
- `demo/downstream.ipynb`: MR/SST downstream usage
