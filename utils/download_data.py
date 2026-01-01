"""
Dataset download script
Download SNLI, MultiNLI, STS[12:16], STS Benchmark datasets
"""

import os
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm


def download_nli_datasets(data_dir='./data'):
    """
    Download SNLI and MultiNLI Dataset
    Args:
        data_dir:
    """
    print("=" * 60)
    print("Downloading NLI Datasets (SNLI + MultiNLI)")
    print("=" * 60)

    # SNLI Download
    print("\n[1/2] Downloading SNLI dataset...")
    snli = load_dataset("stanfordnlp/snli")
    snli.save_to_disk(os.path.join(data_dir, 'SNLI'))
    print(f"SNLI saved to {os.path.join(data_dir, 'SNLI')}")
    print(f"  - Train: {len(snli['train'])} examples")
    print(f"  - Validation: {len(snli['validation'])} examples")
    print(f"  - Test: {len(snli['test'])} examples")

    # MultiNLI Download
    print("\n[2/2] Downloading MultiNLI dataset...")
    mnli = load_dataset("nyu-mll/multi_nli")
    mnli.save_to_disk(os.path.join(data_dir, 'MultiNLI'))
    print(f"MultiNLI saved to {os.path.join(data_dir, 'MultiNLI')}")
    print(f"  - Train: {len(mnli['train'])} examples")
    print(f"  - Validation (matched): {len(mnli['validation_matched'])} examples")
    print(f"  - Validation (mismatched): {len(mnli['validation_mismatched'])} examples")

    total_train = len(snli['train']) + len(mnli['train'])
    print(f"\n Total training examples: {total_train:,}")


def download_sts_benchmark(data_dir='./data'):
    """
    Download STS Benchmark dataset

    Args:
        data_dir:
    """
    print("\n" + "=" * 60)
    print("Downloading STS Benchmark Datasets")
    print("=" * 60)

    sts_datasets = [
        'stsb_multi_mt',  # STS-B
        'sts/sts12-sts',
        'sts/sts13-sts',
        'sts/sts14-sts',
        'sts/sts15-sts',
        'sts/sts16-sts',
        'sickr-sts'
    ]

    # STS-B (Main benchmark)
    print(f"\n[1/{len(sts_datasets)}] Downloading STS-B dataset...")

    stsb = load_dataset("mteb/stsbenchmark-sts")
    stsb.save_to_disk(os.path.join(data_dir, 'STS-B'))
    print(f"STS-benchmark saved to {os.path.join(data_dir,'STS-B')}")

    # STS[12~16]
    for i in range(12, 17):
        print(f"\n[{i}/{len(sts_datasets)}] Downloading STS-{i} dataset...")
        ds_i = load_dataset(f"mteb/sts{i}-sts")
        ds_i.save_to_disk(os.path.join(data_dir, 'STS-' + str(i)))
        print(f"STS-{i} saved to {os.path.join(data_dir, 'STS-' + str(i))}")
    
    # SICKR
    print(f"\n[17/{len(sts_datasets)}] Downloading SICKR dataset...")
    sickr = load_dataset("mteb/sickr-sts")
    sickr.save_to_disk(os.path.join(data_dir, 'SICKR'))
    print(f"SICKR saved to {os.path.join(data_dir, 'SICKR')}")
    print("=" * 60)


def download_sst(data_dir='./data'):
    """
    Download Stanford SST-2 dataset.
    """
    print("\n" + "=" * 60)
    print("Downloading SST-2 Dataset")
    print("=" * 60)
    sst = load_dataset("stanfordnlp/sst2")
    sst.save_to_disk(os.path.join(data_dir, 'SST-2'))
    print(f"SST-2 saved to {os.path.join(data_dir, 'SST-2')}")
    print(f"  - Train: {len(sst['train'])} examples")
    print(f"  - Validation: {len(sst['validation'])} examples")
    print(f"  - Test: {len(sst['test'])} examples")


def download_mr(data_dir='./data'):
    """
    Download MR (Movie Review) dataset (Rotten Tomatoes).
    """
    print("\n" + "=" * 60)
    print("Downloading MR (Rotten Tomatoes) Dataset")
    print("=" * 60)
    mr = load_dataset("rotten_tomatoes")
    mr.save_to_disk(os.path.join(data_dir, 'MR'))
    print(f"MR saved to {os.path.join(data_dir, 'MR')}")
    print(f"  - Train: {len(mr['train'])} examples")
    print(f"  - Validation: {len(mr['validation'])} examples")
    print(f"  - Test: {len(mr['test'])} examples")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download datasets for SBERT')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory to save datasets')
    parser.add_argument('--nli_only', action='store_true',
                        help='Download only NLI datasets')
    parser.add_argument('--sts_only', action='store_true',
                        help='Download only STS datasets')
    parser.add_argument('--sst_only', action='store_true',
                        help='Download only SST-2 dataset')
    parser.add_argument('--mr_only', action='store_true',
                        help='Download only MR dataset')
    args = parser.parse_args()

    # make directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, 'SNLI'), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, 'MultiNLI'), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, 'STS-Benchmark'), exist_ok=True)

    # download
    if not (args.sts_only or args.sst_only or args.mr_only):
        download_nli_datasets(args.data_dir)

    if not args.nli_only and not args.sst_only and not args.mr_only:
        download_sts_benchmark(args.data_dir)

    if args.sst_only:
        download_sst(args.data_dir)
    elif not args.nli_only and not args.mr_only:
        download_sst(args.data_dir)
    if args.mr_only:
        download_mr(args.data_dir)
    elif not args.nli_only and not args.sst_only:
        download_mr(args.data_dir)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
