"""
load NLI and STS dataset
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_from_disk
from typing import List, Tuple, Dict
import random


class NLIDataset(Dataset):
    """
    Dataset made of SNLI or MultiNLI
    """

    def __init__(self, data_path: str, split: str = 'train', max_length: int = 128):
        """
        Args:
            data_path: SNLI or MultiNLI dataset path
            split: 
                'train', 'validation', 'test' for SNLI
                'train', 'validation_matched', 'validation_mismatched' for MultiNLI

            max_length: max sequence length
        """
        self.dataset = load_from_disk(data_path)[split]
        self.max_length = max_length

        # label이 -1인 데이터 제거 (SNLI의 경우 일부 예제에 레이블 없음)
        to_remove_len = len(self.dataset.filter(lambda x: x['label'] == -1))
        self.dataset = self.dataset.filter(lambda x: x['label'] != -1)

        print(f"Loaded {len(self.dataset)} examples from {data_path}/{split}")
        print(f"Removed {to_remove_len} examples with label -1")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # SNLI: premise, hypothesis, label
        # MultiNLI: premise, hypothesis, label
        return {
            'sentence1': item['premise'],
            'sentence2': item['hypothesis'],
            'label': item['label'] # 0 : entailment, 1 : neutral, 2 : contradiction
        }


class STSbDataset(Dataset):
    """
    STSb (Semantic Textual Similarity) dataset
    return : sentence pair and cosine similarity score (0-5 scale)
    """

    def __init__(self, data_path: str, split: str = 'test', max_length: int = 128):
        """
        Args:
            data_path: STSb dataset path
            split: 'train', 'validation', 'test'
            max_length: maximum sequence length
        """
        self.dataset = load_from_disk(data_path)[split]
        self.max_length = max_length

        print(f"Loaded {len(self.dataset)} examples from {data_path}/{split}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # STS-B: sentence1, sentence2, similarity_score (0-5)
        return {
            'sentence1': item['sentence1'],
            'sentence2': item['sentence2'],
            'score': float(item['score'])
        }

class STSDataset(Dataset):
    """
    STS (Semantic Textual Similarity) dataset
    return : sentence pair and cosine similarity score (0-5 scale)
    """

    def __init__(self, data_path: str, split: str = 'test', max_length: int = 128):
        """
        Args:
            data_path: STSb dataset path
            split: 
                'test' for STS13-16 
                'test','train' for STS-12
            max_length: maximum sequence length
        """
        self.dataset = load_from_disk(data_path)[split]
        self.max_length = max_length

        print(f"Loaded {len(self.dataset)} examples from {data_path}/{split}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        return {
            'sentence1': item['sentence1'],
            'sentence2': item['sentence2'],
            'score': float(item['score'])
        }

def load_nli_data(data_dir: str = './data',
                  datasets: List[str] = ['SNLI', 'MultiNLI'],
                  split: str = 'train',
                  batch_size: int = 16,
                  shuffle: bool = True,
                  tokenizer=None,
                  max_length: int = 128) -> DataLoader:
    """
    NLI dataset loader

    Args:
        data_dir: 
        datasets: ['SNLI', 'MultiNLI']
        split: 
            'train', 'validation', 'test' if SNLI, 
            'validation_matched' or 'validation_mismatched' for MultiNLI validation

    Returns:
        DataLoader
    """

    def _map_split(dataset_name: str, requested_split: str) -> str:
        if dataset_name.lower() in {'multinli', 'multi_nli', 'multi-nli'}:
            if requested_split in {'train', 'validation_matched', 'validation_mismatched'}:
                return requested_split
            if requested_split in {'validation', 'dev'}:
                return 'validation_matched'
            if requested_split == 'test':
                return 'validation_mismatched'
            raise ValueError(
                f"Unsupported split '{requested_split}' for MultiNLI. "
                "Use train / validation_matched / validation_mismatched (or pass validation/test for auto-mapping)."
            )

        if dataset_name.lower() == 'snli':
            if requested_split in {'train', 'validation', 'test'}:
                return requested_split
            if requested_split in {'validation_matched', 'dev'}:
                return 'validation'
            if requested_split == 'validation_mismatched':
                return 'test'
            raise ValueError(
                f"Unsupported split '{requested_split}' for SNLI. Use train / validation / test."
            )

        return requested_split

    dataset_objs: List[NLIDataset] = []
    total_len = 0
    for dataset_name in datasets:
        data_path = os.path.join(data_dir, dataset_name)
        actual_split = _map_split(dataset_name, split)
        ds = NLIDataset(data_path, split=actual_split)
        dataset_objs.append(ds)
        total_len += len(ds)

    combined = ConcatDataset(dataset_objs)
    print(f"\nTotal {split} examples: {total_len}")

    # Custom collate function
    def collate_fn(batch):
        sentence1 = [item['sentence1'] for item in batch]
        sentence2 = [item['sentence2'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch])

        if tokenizer is None:
            return {
                'sentence1': sentence1,
                'sentence2': sentence2,
                'labels': labels
            }

        enc = tokenizer(
            sentence1,
            sentence2,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        out = dict(enc)
        out['sentence1'] = sentence1
        out['sentence2'] = sentence2
        out['labels'] = labels
        return out

    return DataLoader(combined, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def load_sts_data(data_dir: str = './data',
                  dataset_name: str = 'STS-B',
                  split: str = 'test',
                  batch_size: int = 16,
                  tokenizer=None,
                  max_length: int = 128) -> DataLoader:
    """
    STS 데이터셋 로더

    Args:
        data_dir: 데이터 디렉토리
        dataset_name: 'STS-B', 'STS12-16'
        split:
            'train', 'validation', 'test' for STS-B
            only 'test' for STS14-16

        batch_size: 배치 크기

    Returns:
        DataLoader
    """
    data_path = os.path.join(data_dir, dataset_name)
    if dataset_name == "STS-B":
        dataset = STSbDataset(data_path, split=split)
    else:
        dataset = STSDataset(data_path, split=split)

    def collate_fn(batch):
        sentence1 = [item['sentence1'] for item in batch]
        sentence2 = [item['sentence2'] for item in batch]
        scores = torch.tensor([item['score'] for item in batch], dtype=torch.float)

        if tokenizer is None:
            return {
                'sentence1': sentence1,
                'sentence2': sentence2,
                'scores': scores
            }

        enc = tokenizer(
            sentence1,
            sentence2,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        out = dict(enc)
        out['sentence1'] = sentence1
        out['sentence2'] = sentence2
        out['scores'] = scores
        return out

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)



