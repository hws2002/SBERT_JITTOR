<div align="center">

# SBERT-Jittor

Pure Jittor implementation of Sentence-BERT (SBERT) distributed as a standalone open-source library.

</div>

## Overview

SBERT-Jittor vendors the original Google BERT architecture (Apache-2.0) into the [Jittor](https://jittor.org) deep-learning framework and layers SBERT pooling / siamese training utilities on top. The core BERT implementation is adapted from [LetianLee/BERT-Jittor](https://github.com/LetianLee/BERT-Jittor) and redistributed here under the same Apache-2.0 terms. This repository is intended to be the **library / toolkit** that other projects can import.

Goals:

- Deliver a **pure-Jittor Transformer stack** (no PyTorch dependency for the training/eval core).
- Provide **Sentence-BERT style pooling + encode APIs** that mirror HuggingFace/SentenceTransformers.
- Ship **reference examples & playgrounds** demonstrating how to plug the modules into real training scripts.
- Capture **licensing & environment guidance** so downstream consumers can safely redistribute.

## Repository Structure

```
SBERT_JITTOR/
├── model/                      # Vendored core BERT (BertConfig, BertModel, heads, etc.)
├── sbert/                      # Sentence-level pooling & siamese wrappers (CLS/mean/max, etc.)
├── examples/                   # Reference scripts (train_nli.py, train_sts.py, evaluate_sts.py)
├── playground/                 # Small demos: transformer block, config inspectors, etc.
├── scripts/                    # Utility scripts (weight conversion, dataset prep)
├── requirements.txt            # Library dependencies
├── LICENSE                     # Apache-2.0 (inherits from Google BERT)
└── README.md                   # This file
```


## Environment Setup

```bash
git clone https://github.com/<you>/SBERT_JITTOR.git
cd SBERT_JITTOR
conda create -n sbert_jittor python=3.10
conda activate sbert_jittor
pip install -r requirements.txt

# Ensure a compiler is available (example for Ubuntu / WSL)
sudo apt update && sudo apt install -y build-essential
```

If Jittor cannot find `g++`, set `export cc_path=/usr/bin/g++` (or the correct path on your system). On Windows install the latest MSVC build tools.

## Quick Start

### 1. Inspect the Jittor BERT backbone

```bash
python playground/bert_base_jittor_demo.py
```

Prints the canonical `bert-base-uncased` config, model summary, total parameters, and verifies a dummy forward pass.

### 2. Prepare datasets

Use the helper in `scripts/prepare_data.py` (or integrate HuggingFace `datasets`) to download SNLI/MNLI + STS-B. Set `HF_HOME` if you need a custom cache path.

### 3. Train SBERT (Jittor)

```bash
 # 4. 전체 옵션
  python train_nli.py bert-base-uncased \
      --pooling mean \
      --batch_size 16 \
      --epochs 1 \
      --eval_steps 1000 \
      --save_steps 1000 \
      --wandb \
      --run_name "nli-v1" \
      --use_cuda

# 5. 사전학습 가중치 로드
python train_nli.py bert-large-uncased --encoder_checkpoint ./checkpoints/hf_bert_large/pytorch_model.bin --pooling mean --wandb
```

The script uses Jittor’s AdamW optimizer and autograd, plus the vendored `BertModel`.

huggingface sentence transformer의 훈련과정을 모방함.

### 4. Evaluate on STS benchmarks

```bash
python examples/evaluate_sts.py \
  --checkpoint checkpoints/sbert_jittor_base/best_model.pkl \
  --eval_split sts-dev \
  --batch_size 32
```

The evaluator dispatches per model type (PyTorch baseline, Jittor SBERT, or HuggingFace SentenceTransformer) but shares the same encode → cosine → Pearson/Spearman pipeline.

### 5. Compare against HuggingFace models

Use the playground summaries:

```bash
python playground/bert_summary.py          # HF BERT base/large reference
python playground/model_summary.py         # SentenceTransformer configs
```

## Weight Handling

- **Conversion from HuggingFace**: 현재 map PyTorch weights into Jittor by downloading it from HuggingFace.

- **Baseline**: Optional PyTorch baselines can live in downstream repos; SBERT-Jittor focuses on the Jittor-native path.

## Licensing

- The BERT implementation is derived from Google’s [BERT](https://github.com/google-research/bert) and the Jittor port [LetianLee/BERT-Jittor](https://github.com/LetianLee/BERT-Jittor), and remains under **Apache License 2.0**.
- Any redistributed checkpoints must comply with the original pretraining data licenses.
- When publishing to HuggingFace or other hubs, include attribution in the model card and reference this repository.

```
SPDX-License-Identifier: Apache-2.0
Copyright 2018 Google
Modifications Copyright 2025 Wooseok Han
```

## Citation

If you use this project, please cite the original BERT paper and SBERT paper:

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
