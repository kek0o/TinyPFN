# TinyPFN

A lightweight Prior-Fitted Network for tabular classification on edge devices.

TinyPFN adapts meta-learning approaches (inspired by TabPFN/TabICL) for deployment on resource-constrained microcontrollers like ESP32. It uses linear attention (O(n) complexity) to enable zero-shot inference on these devices.

## Features

- **Zero-shot inference**: No retraining needed for new datasets
- **Linear attention**: O(n) complexity instead of O(n²)
- **Edge-ready**: Runs on ESP32-WROOM (520KB RAM (~320KB usable), no PSRAM)

## Project Structure

```
TinyPFN/
├── model.py                    # TinyPFN model architecture
├── train.py                    # Training script
├── model_analysis.py           # Benchmarking and memory analysis
│
├── TinyPFN_ESP32/
│   ├── TinyPFN_ESP32.ino       # ESP32 implementation (pure C)
│   ├── validate_esp32.py       # Validate Python vs ESP32 outputs
│   ├── export_weights_to_c.py  # Export weights for ESP32
│   └── tinypfn_weights.h       # Exported model weights
│
└── priors/                     # Synthetic data generation
    ├── main.py                 # CLI for generating priors
    ├── dataset.py              # PriorDataset class
    ├── dataloader.py           # Data loading utilities
    ├── mlp_scm.py              # MLP-based SCM generator
    ├── tree_scm.py             # Tree-based SCM generator
    ├── reg2cls.py              # Regression to classification
    └── ...
```

---

## 1. Model

The core architecture in `model.py`:

- **LinearAttention**: O(n) attention using kernel feature maps
- **TinyPFNModel**: Transformer with dual attention (features + datapoints) (based on [TabPFN](https://github.com/automl/TabPFN))
- **TinyPFNClassifier**: Scikit-learn compatible interface wrapper

```python
from model import TinyPFNModel, TinyPFNClassifier

model = TinyPFNModel(
    embedding_size=16,
    num_attention_heads=1,
    mlp_hidden_size=32,
    num_layers=4,
    num_outputs=3,
    use_linear_attention=True
)

# Scikit-learn style usage
classifier = TinyPFNClassifier(model, device='cpu')
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
```

---

## 2. Training

Train on synthetic prior data generated from structural causal models.

```bash
# Train with pre-generated prior (HDF5)
python train.py
```

Edit `train.py` to configure:

```python
model = TinyPFNModel(
    embedding_size=16,
    num_attention_heads=1,
    mlp_hidden_size=32,
    num_layers=4,
    num_outputs=3
)

prior = PriorDumpDataLoader(
    "tabicl_300k_150x5_exact3class.h5",
    num_steps=2500,
    batch_size=32,
    min_classes=3
)

model, history = train(model, prior, lr=4e-3, steps_per_eval=25)
```

Output: `tinypfn_prior_trained.pt`

---

## 3. ESP32 Deployment

### Export weights to C header

```bash
python export_weights_to_c.py \
    --checkpoint tinypfn_prior_trained.pt \
    --output TinyPFN_ESP32/tinypfn_weights.h
```

### Flash to ESP32

1. Open `TinyPFN_ESP32/TinyPFN_ESP32.ino` in Arduino IDE
2. Select board: ESP32 Dev Module
3. Upload

### Validate Python vs ESP32

```bash
python validate_esp32.py
```

### ESP32 Configuration

In `TinyPFN_ESP32.ino`:

```c
#define NUM_FEATURES 5
#define NUM_TRAIN 50
#define NUM_TEST 50
#define NUM_ROWS (NUM_TRAIN + NUM_TEST)
```

---

## 4. Synthetic Prior Generation

Generate training data using structural causal models (MLP-SCM and Tree-SCM).

```bash
# Generate 300k datasets with exactly 3 classes
python -m priors \
    --num_batches 75000 \
    --batch_size 4 \
    --min_features 5 \
    --max_features 5 \
    --max_seq_len 150 \
    --min_classes 3 \
    --max_classes 3 \
    --save_path tabicl_300k_150x5_exact3class.h5
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--save_path` | auto | Output HDF5 path |
| `--num_batches` | 100 | Number of batches |
| `--batch_size` | 8 | Datasets per batch |
| `--min_features` | 1 | Min features |
| `--max_features` | 100 | Max features |
| `--min_seq_len` | None | Min samples per dataset |
| `--max_seq_len` | 1024 | Max samples per dataset |
| `--min_classes` | 2 | Min classes |
| `--max_classes` | 10 | Max classes |
| `--device` | cpu | cpu or cuda |

---

## 5. Benchmarking

Compare TinyPFN against memory-matched baselines.

```bash
python model_analysis.py
```

Configurable at top of file:

```python
N_FEATURES = 5
N_TRAIN = 75
N_TEST = 75
PRECISION_BYTES = 4  # FP32
```

---

## Requirements

```
torch>=2.0
numpy
scipy
scikit-learn
h5py
schedulefree
xgboost
joblib
```

### Install

```bash
pip install torch numpy scipy scikit-learn h5py schedulefree xgboost joblib
```

### ESP32

- Arduino IDE 2.x
- ESP32 board package (`esp32` by Espressif)

---

## Quick Start

```bash
# 1. Generate synthetic training data
python -m priors \
    --num_batches 1000 \
    --batch_size 4 \
    --max_features 5 \
    --max_seq_len 150 \
    --min_classes 2 \
    --max_classes 3 \
    --save_path prior_4k.h5

# 2. Train model
python train.py

# 3. Export for ESP32
python export_weights_to_c.py --checkpoint tinypfn_prior_trained.pt

# 4. Validate
python validate_esp32.py
```

## License

MIT

## Acknowledgments

Based on ideas from [TabPFN](https://github.com/PriorLabs/TabPFN) and [TabICL](https://github.com/soda-inria/tabicl). Synthetic prior generation adapted from TabICL.