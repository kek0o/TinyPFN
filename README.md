# TinyPFN — Continual Learning for Tabular Data in resource-constrained devices (TinyML)

This project is a **re-implementation and extension of TabPFN**, designed to explore **continual learning**, **tiny models**, and **edge inference** on **non-stationary tabular data**.

The goal is to build a **tiny Prior-Fitted Network (PFN)** capable of:

- learning from **streaming** or **incremental** data,  
- operating under **TinyML constraints**,  
- adapting to **distribution shifts**,  
- and maintaining competitive performance on small tabular classification tasks.

---

## Main Modules

This repository contains the following main components:

### **1. `model.py`**
Defines the **TinyPFNModel** and its components:
- **TinyPFNModel**: Compact Transformer-based model for tabular data with optional adapters and learnable prompt pool.  
- **Adapters**: Small bottleneck layers inserted into Transformer blocks for continual learning fine-tuning.  
- **PromptPool**: Learnable prompts that can be prepended to inputs.  
- **Encoders**: `FeatureEncoder` and `TargetEncoder` for feature/target embedding.  
- **TransformerEncoderLayer**: Multihead attention between features and between datapoints, with layer normalization and MLP.  
- **Decoder**: Maps embeddings to outputs.  
- **TinyPFNClassifier**: sklearn-like wrapper for easy `.fit()`, `.predict()`, `.predict_proba()` usage.

### **2. `train.py`**
Training utilities and prior loader:
- `train()`: Function to train TinyPFNModel on synthetic prior data.  
- `PriorDumpDataLoader`: Efficient streaming of synthetic datasets from `.h5` files.  
- Evaluation functions for tabular datasets using metrics like ROC-AUC, accuracy, and balanced accuracy.  
- Script to pretrain TinyPFN on priors and save a checkpoint.

### **3. `drift.py`**
Continual learning and drift detection:
- **ReservoirBuffer**: Fixed-size buffer storing (x, y) pairs using reservoir sampling.  
- **PCAChangeDetector**: Incremental PCA-based drift detector.  
- **EntropyDetector**: Monitors predictive entropy for drift.  
- **DriftManager**: Combines PCA and entropy detectors, tracks drift events.

### **4. `continual_utils.py`**
Streaming inference and continual adaptation tools:
- Load pretrained TinyPFN models.  
- Extract feature embeddings.  
- Train adapters and prompts on memory for active adaptation.  
- `continual_inference_with_active_updates()`: Full streaming loop that detects drift, updates memory, retrains adapters/prompts, and performs predictions.

---

This structure enables:
- **Efficient TinyML-friendly training** on synthetic priors.  
- **Continual learning and active adaptation** on streaming tabular data.  
- **Drift detection** and automated model update using reservoir memory.

