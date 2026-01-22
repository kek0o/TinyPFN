# Priors - Synthetic Dataset Generation for TinyPFN

Standalone implementation for generating synthetic tabular datasets.

Based on TabICL with improvements:
- **`min_classes` parameter** - control minimum number of classes
- **Fixed `MulticlassAssigner`** - guarantees exact class count

## Usage

```bash
# Generate with exactly 3 classes (like before but with min_classes)
python -m priors \
    --num_batches 75000 --batch_size 4 \
    --min_features 5 --max_features 5 \
    --max_seq_len 150 \
    --min_classes 3 --max_classes 3 \
    --save_path tabicl_300k_150x5_exact3class.h5
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--save_path` | auto | HDF5 output path |
| `--num_batches` | 100 | Number of batches |
| `--batch_size` | 8 | Samples per batch |
| `--min_features` | 1 | Min features |
| `--max_features` | 100 | Max features |
| `--min_seq_len` | None | Min sequence length |
| `--max_seq_len` | 1024 | Max sequence length |
| `--min_classes` | 2 | **NEW:** Min classes |
| `--max_classes` | 10 | Max classes |
| `--device` | cpu | cpu or cuda |

## Files

```
priors/
├── __init__.py, __main__.py, main.py  # Entry points
├── dataset.py      # PriorDataset (modified: min_classes)
├── reg2cls.py      # MulticlassAssigner (modified: quantile-based)
├── dataloader.py   # LocalPriorDataLoader
├── utils.py        # dump_prior_to_h5
├── mlp_scm.py, tree_scm.py  # SCM generators
├── scm_utils.py    # GaussianNoise, XSampler
├── prior_config.py, hp_sampling.py, activations.py
└── README.md
```
