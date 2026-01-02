<div align="center">

# SBERT-Jittor

Sentence-BERT implemented in Jittor with a full NLI training pipeline.

</div>

## Overview

This project reimplements Sentence-BERT in pure Jittor. It vendors a Jittor-native
BERT encoder, adds SBERT pooling heads, and provides training/evaluation scripts
to reproduce the standard NLI -> STS workflow. The goal is a self-contained,
open-source SBERT stack that trains end-to-end in Jittor.

The Jittor BERT implementation is adapted from
https://github.com/LetianLee/BERT-Jittor and follows the Apache 2.0 license.

## What is implemented

- **Encoder**: Jittor BERT (BertConfig, BertModel) adapted from BERT-Jittor.
- **Pooling**: mean / cls / max pooling to produce sentence embeddings.
- **Heads**:
  - Identity (no projection)
  - Linear projection
  - MLP projection
  - Classification head for NLI training
- **Objectives**:
  - Regression (STS): cosine similarity between sentence embeddings u and v,
    optimized with mean squared error.
  - Triplet loss: for anchor a, positive p, negative n, minimize
    `max(||s_a - s_p|| - ||s_a - s_n|| + margin, 0)` with Euclidean distance
    and margin = 1.
- **Training**: NLI (SNLI + MultiNLI) with periodic STS evaluation.
- **Evaluation**: STS12-16, STS-B, SICK-R with Pearson/Spearman.

## Repository layout

```
SBERT_JITTOR/
  model/                 # BERT + SBERT encoder implementation
  heads/                 # Identity, Linear, MLP, Classification heads
  utils/                 # data/cache utils, download scripts
  training/              # train_nli.py, train_nli_test.py
  evaluation/            # evaluate.py
  train_nli_gpu.py        # cached NLI training for GPU servers
  hf/                    # local HF assets (tokenizer + checkpoints)
  data/                  # datasets (SNLI, MultiNLI, STS-*, SICKR)
```

## Environment

```bash
conda create -n sbert_jittor python=3.12
conda activate sbert_jittor
pip install -r requirements.txt
```

## Assets and data

### 1) Datasets

```bash
python utils/download_data.py
```

Datasets are stored under `./data/`:
`SNLI`, `MultiNLI`, `STS-12..16`, `STS-B`, `SICKR`, `SST-2`, `MR`.

## Data sources (Hugging Face)

- **SNLI**: https://huggingface.co/datasets/stanfordnlp/snli
- **MultiNLI**: https://huggingface.co/datasets/nyu-mll/multi_nli
- **STS12-16**: https://huggingface.co/datasets/mteb/sts12-sts, https://huggingface.co/datasets/mteb/sts13-sts, https://huggingface.co/datasets/mteb/sts14-sts, https://huggingface.co/datasets/mteb/sts15-sts, https://huggingface.co/datasets/mteb/sts16-sts
- **STS-B**: https://huggingface.co/datasets/mteb/stsbenchmark-sts
- **SICK-R**: https://huggingface.co/datasets/mteb/sickr-sts
- **SST-2**: https://huggingface.co/datasets/glue
- **MR (Rotten Tomatoes)**: https://huggingface.co/datasets/rotten_tomatoes

### 2) Tokenizer (local)

```bash
python utils/download_tokenizer.py --repo_id bert-large-uncased
```

Default output:
```
./hf/tokenizer/<repo_id>/
```

### 3) HF checkpoints (local)

```bash
python utils/download_checkpoints.py --model-name bert-large-uncased
```

Default output:
```
./hf/pretrained_bert_checkpoints/<model_name>/pytorch_model.bin
```

## Training

The training scripts automatically load:
- tokenizer from `./hf/tokenizer/<base_model>` if present
- encoder checkpoint from `./hf/pretrained_bert_checkpoints/<base_model>/pytorch_model.bin` if present

### Training details (from SBERT paper)

- **NLI fine-tuning**: 1 epoch on SNLI + MultiNLI
- **Optimizer**: Adam, lr `2e-5`, linear warmup `10%`
- **Pooling**: MEAN (default)
- **STS-B supervised**: train/dev/test = 5,749 / 1,500 / 1,379
- **Regression**: cosine similarity + MSE on STS targets (0-5)
- **Two-stage**: NLI pretraining, then STS-B fine-tuning

### Local training (cached tokenization)

```bash
python training/train_nli.py bert-large-uncased --pooling mean --use_cuda
```

### SST-2 training

```bash
python training/sst/train_sst.py bert-base-uncased \
  --encoder_checkpoint ./hf/hf_bert_base/pytorch_model.bin \
  --data_dir ./data --cache_dir ./data/tokenized \
  --pooling mean --use_cuda
```

### MR training

```bash
python training/mr/train_mr.py bert-base-uncased \
  --encoder_checkpoint ./hf/hf_bert_base/pytorch_model.bin \
  --data_dir ./data --cache_dir ./data/tokenized \
  --pooling mean --use_cuda
```

### GPU server training (cached + resume + checkpoints)

```bash
python train_nli_gpu.py bert-large-uncased \
  --encoder_checkpoint ./hf/pretrained_bert_checkpoints/bert-large-uncased/pytorch_model.bin \
  --data_dir ./data \
  --cache_dir ./data/tokenized \
  --pooling mean \
  --use_cuda \
  --output_dir ./output \
  --save_steps 1000
```

Checkpoint behavior:
- `{model_name}_best.pkl` keeps the best validation score.

Resume training:
```bash
python train_nli_gpu.py bert-large-uncased \
  --start_from_checkpoints ./output/bert-large-uncased_best.pkl \
  --use_cuda
```

## Evaluation

Evaluate a trained checkpoint on STS datasets:

```bash
python evaluation/evaluate.py \
  --checkpoint_path ./output/best.pkl \
  --base_model bert-large-uncased \
  --datasets all
```

Supported datasets:
`sts12`, `sts13`, `sts14`, `sts15`, `sts16`, `stsb`, `sick-r`, `all`.

### BERT STS baselines

Bi-encoder (mean/cls pooling + cosine similarity):

```bash
python evaluation/sts/evaluate_bert_bi.py \
  --base_model bert-base-uncased \
  --pooling cls \
  --model_root ./hf \
  --encoder_checkpoint ./hf/hf_bert_base/pytorch_model.bin \
  --datasets stsb
```

Cross-encoder (pair input + regression head):

```bash
python evaluation/sts/evaluate_bert_cross.py \
  --model_path ./hf/hf_bert_base_stsb \
  --datasets stsb
```

## Typical usage (code)

Encoding example :

```python
from model.sbert_model import SBERTJittor

model = SBERTJittor("bert-base-uncased", pooling="mean", head_type="none")

u = model.encode(input_ids_a, attention_mask_a, token_type_ids_a)
v = model.encode(input_ids_b, attention_mask_b, token_type_ids_b)
```

Common training flows:

```bash
# 1) NLI fine-tuning (classification objective)
python training/train_nli.py bert-base-uncased --pooling mean --use_cuda

# 2) STS regression fine-tuning
python training/train_sts.py bert-base-uncased --pooling mean --use_cuda
```

Evaluation on STS benchmarks:

```bash
python evaluation/evaluate.py --checkpoint_path ./output/best.pkl --base_model bert-base-uncased --datasets stsb
```

## Licensing

- The BERT implementation is derived from Google BERT and BERT-Jittor
  and remains under Apache 2.0.
- Any redistributed checkpoints must comply with the original model licenses.

```
SPDX-License-Identifier: Apache-2.0
Copyright 2018 Google
Modifications Copyright 2025 Wooseok Han
```

## Citation

```
@article{devlin2019bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={NAACL},
  year={2019}
}

@inproceedings{reimers2019sbert,
  title={Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  author={Reimers, Nils and Gurevych, Iryna},
  booktitle={EMNLP},
  year={2019}
}
```
