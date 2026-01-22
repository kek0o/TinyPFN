"""Main module for the priors package.

Usage (compatible with tfmplayground.priors):
    python -m priors \
        --num_batches 75000 --batch_size 4 \
        --min_features 5 --max_features 5 \
        --max_seq_len 150 \
        --min_classes 3 --max_classes 3 \
        --save_path tabicl_300k_150x5_multiclass_exact3.h5
"""

import argparse
import random

import numpy as np
import torch

from .dataloader import LocalPriorDataLoader
from .utils import dump_prior_to_h5


def main():
    parser = argparse.ArgumentParser(description="Generate and dump prior datasets into HDF5 format.")
    
    parser.add_argument("--save_path", type=str, required=False, help="Path to save the HDF5 file.")
    parser.add_argument("--num_batches", type=int, default=100, help="Number of batches to dump.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for dumping.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run prior sampling on.")
    parser.add_argument("--min_features", type=int, default=1, help="Minimum number of input features.")
    parser.add_argument("--max_features", type=int, default=100, help="Maximum number of input features.")
    parser.add_argument("--min_seq_len", type=int, default=None, help="Minimum number of data points per function.")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum number of data points per function.")
    parser.add_argument("--min_classes", type=int, default=2, help="Minimum number of classes (NEW!).")
    parser.add_argument("--max_classes", type=int, default=10, help="Maximum number of classes (classification only).")
    parser.add_argument("--np_seed", type=int, default=None, help="Random seed for NumPy.")
    parser.add_argument("--torch_seed", type=int, default=None, help="Random seed for PyTorch.")

    args = parser.parse_args()

    if args.np_seed is not None:
        np.random.seed(args.np_seed)
    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
        random.seed(args.torch_seed)

    device = torch.device(args.device)

    # Generate default save path if not provided
    if args.save_path is None:
        args.save_path = f"prior_{args.num_batches}x{args.batch_size}_{args.max_seq_len}x{args.max_features}.h5"

    # Handle min_seq_len (PriorDataset requires min_seq_len < max_seq_len or None)
    min_seq_len = args.min_seq_len
    if min_seq_len is not None and min_seq_len == args.max_seq_len:
        min_seq_len = None

    # Create data loader
    prior = LocalPriorDataLoader(
        num_steps=args.num_batches,
        batch_size=args.batch_size,
        num_datapoints_min=min_seq_len,
        num_datapoints_max=args.max_seq_len,
        min_features=args.min_features,
        max_features=args.max_features,
        min_num_classes=args.min_classes,
        max_num_classes=args.max_classes,
        device=device,
    )

    print(f"Generating {args.num_batches} batches × {args.batch_size} = {args.num_batches * args.batch_size} samples")
    print(f"Features: {args.min_features}-{args.max_features}, Seq len: {args.max_seq_len}, Classes: {args.min_classes}-{args.max_classes}")
    print(f"Saving to: {args.save_path}")

    dump_prior_to_h5(
        prior,
        max_classes=args.max_classes,
        batch_size=args.batch_size,
        save_path=args.save_path,
        problem_type="classification",
        max_seq_len=args.max_seq_len,
        max_features=args.max_features,
    )

    print(f"Done!")


if __name__ == "__main__":
    main()
