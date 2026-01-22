"""
FAIR COMPARISON BENCHMARK - Memory Matched Version
==================================================

All models are matched to the same TOTAL memory as TinyPFN.

Configuration is easy to change at the top of the file:
- N_FEATURES: number of features to select
- N_TRAIN: number of training samples
- N_TEST: number of test samples
- PRECISION_BYTES: bytes per parameter (4 for FP32, 2 for FP16, 1 for INT8)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

import sys
sys.path.append('.')
try:
    from model import TinyPFNModel, TinyPFNClassifier
except ImportError:
    print("Warning: Could not import TinyPFN model.")
    TinyPFNModel = None
    TinyPFNClassifier = None

np.random.seed(42)
torch.manual_seed(42)

# =============================================================================
# CONFIGURATION - CHANGE THESE PARAMETERS
# =============================================================================

N_FEATURES = 5          # Number of features to select (max seen in prior: 5)
N_TRAIN = 75            # Number of training samples
N_TEST = 75             # Number of test samples
PRECISION_BYTES = 4     # 4 = FP32, 2 = FP16, 1 = INT8

# TinyPFN architecture (should match your trained model)
EMBEDDING_SIZE = 16
NUM_HEADS = 1
MLP_HIDDEN = 32
NUM_LAYERS = 4
NUM_OUTPUTS = 3


# =============================================================================
# MEMORY COMPUTATION FUNCTIONS
# =============================================================================

def compute_memory_tinypfn(n_rows, n_features, E=EMBEDDING_SIZE, M=MLP_HIDDEN, L=NUM_LAYERS,
                           H=NUM_HEADS, O=NUM_OUTPUTS, precision=PRECISION_BYTES):
    """
    Compute TinyPFN memory usage based on architecture.
    
    Weights breakdown:
    - Feature encoder: E weights + E bias = 2E
    - Target encoder: E weights + E bias = 2E
    - Per transformer layer:
        - 2 attention blocks (features + datapoints), each with:
            - Q, K, V, O projections: 4 * (E*E + E) = 4E² + 4E
        - 3 layer norms: 3 * 2E = 6E (gamma + beta each)
        - MLP: E*M + M + M*E + E = 2EM + M + E
        - Total per layer: 8E² + 8E + 6E + 2EM + M + E = 8E² + 2EM + 15E + M
    - Decoder: E*M + M + M*O + O = EM + M + MO + O
    """
    # Weights calculation
    encoder_params = 2 * E + 2 * E  # Feature + Target encoder
    layer_params = 8 * E * E + 2 * E * M + 15 * E + M  # Per transformer layer
    decoder_params = E * M + M + M * O + O
    
    total_params = encoder_params + L * layer_params + decoder_params
    
    weights_kb = total_params * precision / 1024
    
    # Activations (depends on input size)
    # Peak activation occurs during attention or MLP
    n_cols = n_features + 1  # features + target column
    
    # For linear attention, peak is during KV computation or MLP
    activation_candidates = [
        3 * n_rows * n_cols * E,      # Q, K, V tensors in feature attention
        3 * n_cols * n_rows * E,      # Q, K, V tensors in datapoint attention
        H * n_rows * E * E // H,      # KV cache: (BR, H, D, D) = BR * E * D = BR * E²/H
        H * n_cols * E * E // H,      # KV cache for datapoints
        n_rows * n_cols * M,          # MLP hidden activations
    ]
    activation_peak = max(activation_candidates)
    activations_kb = activation_peak * precision / 1024
    
    return {
        'params': total_params,
        'weights_kb': weights_kb,
        'activations_kb': activations_kb,
        'total_kb': weights_kb + activations_kb
    }


def compute_memory_mlp(input_dim, hidden_dim, num_classes, batch_size=1, precision=PRECISION_BYTES):
    """Compute MLP memory usage"""
    # Weights: input->hidden + hidden bias + hidden->output + output bias
    params = input_dim * hidden_dim + hidden_dim + hidden_dim * num_classes + num_classes
    weights_kb = params * precision / 1024
    
    # Activations: peak is the hidden layer
    activations_kb = batch_size * hidden_dim * precision / 1024
    
    return {
        'params': params,
        'weights_kb': weights_kb,
        'activations_kb': activations_kb,
        'total_kb': weights_kb + activations_kb
    }


def compute_memory_xgb(n_estimators, max_depth, n_features, batch_size=1, precision=PRECISION_BYTES):
    """Compute XGBoost memory usage"""
    # Each tree structure:
    # - Internal nodes: feature index (4 bytes) + threshold (4 bytes) + children pointers (8 bytes) = 16 bytes
    # - Leaf nodes: prediction value (4 bytes)
    internal_nodes = 2**max_depth - 1
    leaf_nodes = 2**max_depth
    bytes_per_tree = internal_nodes * 16 + leaf_nodes * 4
    weights_kb = n_estimators * bytes_per_tree / 1024
    
    # Activations: input features + traversal state
    activations_kb = (batch_size * n_features * precision + max_depth * 4) / 1024
    
    return {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'weights_kb': weights_kb,
        'activations_kb': activations_kb,
        'total_kb': weights_kb + activations_kb
    }


# =============================================================================
# FLOPS COMPUTATION FUNCTIONS
# =============================================================================

def compute_flops_tinypfn(n_rows, n_features, E=EMBEDDING_SIZE, M=MLP_HIDDEN, L=NUM_LAYERS,
                          H=NUM_HEADS, O=NUM_OUTPUTS, B=1):
    """
    Compute TinyPFN FLOPs for inference using linear attention.
    
    Based on detailed analysis:
    - Feature encoder: 2BRC(2+E)
    - Target encoder: BR_train + 2BRE
    - Per transformer layer:
        - Feature attention (linear): 12BRCE² + 10BRCE
        - Residual + LayerNorm 1: 9BRCE
        - Datapoint attention (linear): 9BCRE² + 8BCRE
        - Residual + LayerNorm 2: 9BRCE  (note: BRCE = BCRE)
        - MLP block: 4BRCEM + 5BRCM + 9BRCE
    - Decoder: BR_test * M * (2E + 5 + 2O)
    """
    C = n_features + 1  # columns = features + target
    R = n_rows
    R_train = R // 2
    R_test = R - R_train
    
    # Feature encoder: 2BRC(2+E)
    flops_feat_enc = 2 * B * R * C * (2 + E)
    
    # Target encoder: BR_train + 2BRE
    flops_tgt_enc = B * R_train + 2 * B * R * E
    
    # Per transformer layer
    # Feature attention (linear, H=1): 12BRCE² + 10BRCE
    flops_attn_feat = 12 * B * R * C * E * E + 10 * B * R * C * E
    
    # Residual + LayerNorm 1: BRCE + 8BRCE = 9BRCE
    flops_res_norm1 = 9 * B * R * C * E
    
    # Datapoint attention (linear, H=1): 9BCRE² + 8BCRE
    flops_attn_dp = 9 * B * C * R * E * E + 8 * B * C * R * E
    
    # Residual + LayerNorm 2: 9BRCE
    flops_res_norm2 = 9 * B * R * C * E
    
    # MLP block: 4BRCEM + 5BRCM + 9BRCE
    flops_mlp = 4 * B * R * C * E * M + 5 * B * R * C * M + 9 * B * R * C * E
    
    # Total per layer
    flops_per_layer = flops_attn_feat + flops_res_norm1 + flops_attn_dp + flops_res_norm2 + flops_mlp
    
    # Decoder: BR_test * M * (2E + 5 + 2O)
    flops_decoder = B * R_test * M * (2 * E + 5 + 2 * O)
    
    # Total
    total_flops = flops_feat_enc + flops_tgt_enc + L * flops_per_layer + flops_decoder
    
    return total_flops / 1e3  # Return in kFLOPs


def compute_flops_tinypfn_standard(n_rows, n_features, E=EMBEDDING_SIZE, M=MLP_HIDDEN, L=NUM_LAYERS,
                                    H=NUM_HEADS, O=NUM_OUTPUTS, B=1):
    """
    Compute TinyPFN FLOPs for inference using STANDARD (quadratic) attention.
    For comparison purposes.
    
    Per transformer layer:
    - Feature attention (standard): 8BRCE² + 4BRC²E + 4BRC²
    - Residual + LayerNorm 1: 9BRCE
    - Datapoint attention (standard): 8BCRE² + 2BCR²E + 2BCR²
    - Residual + LayerNorm 2: 9BRCE
    - MLP block: 4BRCEM + 5BRCM + 9BRCE
    """
    C = n_features + 1
    R = n_rows
    R_train = R // 2
    R_test = R - R_train
    
    # Feature encoder: 2BRC(2+E)
    flops_feat_enc = 2 * B * R * C * (2 + E)
    
    # Target encoder: BR_train + 2BRE
    flops_tgt_enc = B * R_train + 2 * B * R * E
    
    # Per transformer layer
    # Feature attention (standard, H=1): 8BRCE² + 4BRC²E + 4BRC²
    flops_attn_feat = 8 * B * R * C * E * E + 4 * B * R * C * C * E + 4 * B * R * C * C
    
    # Residual + LayerNorm 1: 9BRCE
    flops_res_norm1 = 9 * B * R * C * E
    
    # Datapoint attention (standard, H=1): 8BCRE² + 2BCR²E + 2BCR²
    flops_attn_dp = 8 * B * C * R * E * E + 2 * B * C * R * R * E + 2 * B * C * R * R
    
    # Residual + LayerNorm 2: 9BRCE
    flops_res_norm2 = 9 * B * R * C * E
    
    # MLP block: 4BRCEM + 5BRCM + 9BRCE
    flops_mlp = 4 * B * R * C * E * M + 5 * B * R * C * M + 9 * B * R * C * E
    
    # Total per layer
    flops_per_layer = flops_attn_feat + flops_res_norm1 + flops_attn_dp + flops_res_norm2 + flops_mlp
    
    # Decoder: BR_test * M * (2E + 5 + 2O)
    flops_decoder = B * R_test * M * (2 * E + 5 + 2 * O)
    
    # Total
    total_flops = flops_feat_enc + flops_tgt_enc + L * flops_per_layer + flops_decoder
    
    return total_flops / 1e3  # Return in kFLOPs


def compute_flops_mlp(input_dim, hidden_dim, num_classes, n_samples):
    """
    Compute MLP FLOPs for inference.
    
    - Linear 1: 2 * n_samples * input_dim * hidden_dim
    - ReLU: n_samples * hidden_dim (comparisons)
    - Linear 2: 2 * n_samples * hidden_dim * num_classes
    """
    flops_linear1 = 2 * n_samples * input_dim * hidden_dim
    flops_relu = n_samples * hidden_dim
    flops_linear2 = 2 * n_samples * hidden_dim * num_classes
    
    total_flops = flops_linear1 + flops_relu + flops_linear2
    return total_flops / 1e3  # kFLOPs


def compute_flops_xgb(n_estimators, max_depth, n_samples):
    """
    Compute XGBoost FLOPs for inference.
    
    Per sample per tree:
    - max_depth comparisons (feature threshold)
    - max_depth memory accesses (approximated as 1 FLOP each)
    - 1 addition for accumulating prediction
    """
    flops_per_sample_per_tree = 2 * max_depth + 1
    total_flops = n_samples * n_estimators * flops_per_sample_per_tree
    
    return total_flops / 1e3  # kFLOPs


# =============================================================================
# MEMORY-MATCHED MODEL CREATION
# =============================================================================

def create_mlp_with_memory_budget(target_total_kb, input_dim, num_classes, batch_size=1, 
                                   precision=PRECISION_BYTES):
    """
    Create an MLP that matches the target total memory budget.
    Uses binary search to find the right hidden dimension.
    """
    print(f"  Searching for MLP hidden_dim to match {target_total_kb:.2f} KB total memory...")
    
    low, high = 1, 10000
    best_hidden = 1
    best_diff = float('inf')
    
    while low <= high:
        mid = (low + high) // 2
        mem = compute_memory_mlp(input_dim, mid, num_classes, batch_size, precision)
        diff = abs(mem['total_kb'] - target_total_kb)
        
        if diff < best_diff:
            best_diff = diff
            best_hidden = mid
        
        if mem['total_kb'] < target_total_kb:
            low = mid + 1
        else:
            high = mid - 1
    
    # Verify final memory
    final_mem = compute_memory_mlp(input_dim, best_hidden, num_classes, batch_size, precision)
    print(f"  Found hidden_dim={best_hidden} -> {final_mem['total_kb']:.2f} KB "
          f"(weights={final_mem['weights_kb']:.2f} KB, activations={final_mem['activations_kb']:.2f} KB)")
    
    return best_hidden, final_mem


def create_xgboost_with_memory_budget(target_total_kb, n_features, batch_size=1, 
                                       precision=PRECISION_BYTES):
    """
    Create XGBoost parameters that match the target total memory budget.
    Searches over (n_estimators, max_depth) combinations.
    """
    print(f"  Searching for XGBoost params to match {target_total_kb:.2f} KB total memory...")
    
    best_params = (10, 3)
    best_diff = float('inf')
    best_mem = None
    
    # Search over reasonable ranges
    for max_depth in range(2, 15):
        # Binary search for n_estimators
        low, high = 1, 1000
        
        while low <= high:
            mid = (low + high) // 2
            mem = compute_memory_xgb(mid, max_depth, n_features, batch_size, precision)
            diff = abs(mem['total_kb'] - target_total_kb)
            
            if diff < best_diff:
                best_diff = diff
                best_params = (mid, max_depth)
                best_mem = mem
            
            if mem['total_kb'] < target_total_kb:
                low = mid + 1
            else:
                high = mid - 1
    
    n_estimators, max_depth = best_params
    final_mem = compute_memory_xgb(n_estimators, max_depth, n_features, batch_size, precision)
    print(f"  Found n_estimators={n_estimators}, max_depth={max_depth} -> {final_mem['total_kb']:.2f} KB "
          f"(weights={final_mem['weights_kb']:.2f} KB, activations={final_mem['activations_kb']:.2f} KB)")
    
    return n_estimators, max_depth, final_mem


# =============================================================================
# MLP CLASSIFIER
# =============================================================================

class SimpleMLPClassifier:
    def __init__(self, input_dim, hidden_dim, num_classes):
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        ]
        self.model = nn.Sequential(*layers)
        
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters())
    
    def fit(self, X_train, y_train, epochs=200, lr=0.01):
        X = torch.FloatTensor(X_train)
        y = torch.LongTensor(y_train)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(self.model(X), y)
            loss.backward()
            optimizer.step()
    
    def predict(self, X_test):
        self.model.eval()
        with torch.no_grad():
            return torch.max(self.model(torch.FloatTensor(X_test)), 1)[1].numpy()
    
    def predict_proba(self, X_test):
        self.model.eval()
        with torch.no_grad():
            return torch.softmax(self.model(torch.FloatTensor(X_test)), dim=1).numpy()


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_benchmark(X, y, dataset_name="Dataset"):
    """
    Run the full benchmark on a given dataset.
    
    Args:
        X: Feature matrix (will be reduced to N_FEATURES)
        y: Labels
        dataset_name: Name for display
    """
    print("=" * 80)
    print(f"FAIR COMPARISON BENCHMARK - {dataset_name}")
    print(f"Configuration: {N_FEATURES} features, {N_TRAIN} train, {N_TEST} test, FP{PRECISION_BYTES*8}")
    print("=" * 80)
    
    # Feature selection if needed
    if X.shape[1] > N_FEATURES:
        selector = SelectKBest(f_classif, k=N_FEATURES)
        X_selected = selector.fit_transform(X, y)
        print(f"\nFeature selection: {X.shape[1]} -> {N_FEATURES} features")
    else:
        X_selected = X
        print(f"\nUsing all {X.shape[1]} features")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.5, random_state=42, stratify=y
    )
    
    # Limit samples
    X_train, y_train = X_train[:N_TRAIN], y_train[:N_TRAIN]
    X_test, y_test = X_test[:N_TEST], y_test[:N_TEST]
    
    n_features = X_train.shape[1]
    n_rows = N_TRAIN + N_TEST
    num_classes = len(np.unique(y))
    
    print(f"\nDataset configuration:")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {num_classes}")
    
    results = {}
    
    # =========================================================================
    # 1. TinyPFN
    # =========================================================================
    print("\n" + "-" * 80)
    print("1. TinyPFN (zero-shot, pretrained on synthetic prior)")
    print("-" * 80)
    
    model_path = "tinypfn_prior_trained_multiclass_ticl_v4.pt"
    try:
        device = torch.device('cpu')
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model_kwargs = checkpoint.get('model_kwargs', {
            'embedding_size': EMBEDDING_SIZE, 
            'num_attention_heads': NUM_HEADS, 
            'mlp_hidden_size': MLP_HIDDEN,
            'num_layers': NUM_LAYERS, 
            'num_outputs': NUM_OUTPUTS, 
            'use_linear_attention': True
        })
        
        model = TinyPFNModel(**model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        classifier = TinyPFNClassifier(model, device)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)
        
        mem = compute_memory_tinypfn(n_rows, n_features)
        kflops = compute_flops_tinypfn(n_rows, n_features)
        kflops_std = compute_flops_tinypfn_standard(n_rows, n_features)
        
        results['TinyPFN'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba[:, 1]) if num_classes == 2 else None,
            'params': mem['params'],
            'weights_kb': mem['weights_kb'],
            'activations_kb': mem['activations_kb'],
            'total_kb': mem['total_kb'],
            'kflops': kflops,
            'kflops_standard': kflops_std
        }
        
        r = results['TinyPFN']
        print(f"  Params: {r['params']}")
        print(f"  Memory: Weights={r['weights_kb']:.2f} KB + Activations={r['activations_kb']:.2f} KB = {r['total_kb']:.2f} KB")
        print(f"  FLOPs (linear): {r['kflops']:.2f} kFLOPs")
        print(f"  FLOPs (standard): {r['kflops_standard']:.2f} kFLOPs (reduction: {r['kflops_standard']/r['kflops']:.2f}x)")
        print(f"  Accuracy: {r['accuracy']:.4f} | Balanced: {r['balanced_accuracy']:.4f}", end="")
        if r['roc_auc']:
            print(f" | ROC AUC: {r['roc_auc']:.4f}")
        else:
            print()
        
        target_memory_kb = r['total_kb']
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        target_memory_kb = 88  # Default fallback
    
    # =========================================================================
    # 2. MLP (memory-matched)
    # =========================================================================
    print("\n" + "-" * 80)
    print("2. MLP (trained on this dataset, memory-matched)")
    print("-" * 80)
    
    hidden_dim, mem_mlp = create_mlp_with_memory_budget(
        target_memory_kb, n_features, num_classes, batch_size=1, precision=PRECISION_BYTES
    )
    
    mlp = SimpleMLPClassifier(n_features, hidden_dim, num_classes)
    mlp.fit(X_train, y_train, epochs=200, lr=0.01)
    y_pred_mlp = mlp.predict(X_test)
    y_proba_mlp = mlp.predict_proba(X_test)
    
    kflops_mlp = compute_flops_mlp(n_features, hidden_dim, num_classes, N_TEST)
    
    results['MLP'] = {
        'accuracy': accuracy_score(y_test, y_pred_mlp),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_mlp),
        'roc_auc': roc_auc_score(y_test, y_proba_mlp[:, 1]) if num_classes == 2 else None,
        'params': mlp.count_parameters(),
        'hidden_dim': hidden_dim,
        'weights_kb': mem_mlp['weights_kb'],
        'activations_kb': mem_mlp['activations_kb'],
        'total_kb': mem_mlp['total_kb'],
        'kflops': kflops_mlp
    }
    
    r = results['MLP']
    print(f"  Params: {r['params']} (hidden_dim={hidden_dim})")
    print(f"  Memory: Weights={r['weights_kb']:.2f} KB + Activations={r['activations_kb']:.2f} KB = {r['total_kb']:.2f} KB")
    print(f"  FLOPs: {r['kflops']:.2f} kFLOPs")
    print(f"  Accuracy: {r['accuracy']:.4f} | Balanced: {r['balanced_accuracy']:.4f}", end="")
    if r['roc_auc']:
        print(f" | ROC AUC: {r['roc_auc']:.4f}")
    else:
        print()
    
    # =========================================================================
    # 3. XGBoost (memory-matched)
    # =========================================================================
    print("\n" + "-" * 80)
    print("3. XGBoost (trained on this dataset, memory-matched)")
    print("-" * 80)
    
    n_estimators, max_depth, mem_xgb = create_xgboost_with_memory_budget(
        target_memory_kb, n_features, batch_size=1, precision=PRECISION_BYTES
    )
    
    xgb = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_proba_xgb = xgb.predict_proba(X_test)
    
    kflops_xgb = compute_flops_xgb(n_estimators, max_depth, N_TEST)
    
    results['XGBoost'] = {
        'accuracy': accuracy_score(y_test, y_pred_xgb),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_xgb),
        'roc_auc': roc_auc_score(y_test, y_proba_xgb[:, 1]) if num_classes == 2 else None,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'weights_kb': mem_xgb['weights_kb'],
        'activations_kb': mem_xgb['activations_kb'],
        'total_kb': mem_xgb['total_kb'],
        'kflops': kflops_xgb
    }
    
    r = results['XGBoost']
    print(f"  Trees: {n_estimators}, Depth: {max_depth}")
    print(f"  Memory: Weights={r['weights_kb']:.2f} KB + Activations={r['activations_kb']:.2f} KB = {r['total_kb']:.2f} KB")
    print(f"  FLOPs: {r['kflops']:.2f} kFLOPs")
    print(f"  Accuracy: {r['accuracy']:.4f} | Balanced: {r['balanced_accuracy']:.4f}", end="")
    if r['roc_auc']:
        print(f" | ROC AUC: {r['roc_auc']:.4f}")
    else:
        print()
    
    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "=" * 110)
    print(f"SUMMARY - {dataset_name} ({N_FEATURES} features, {N_TRAIN}+{N_TEST} samples, batch_size=1)")
    print("=" * 110)
    print(f"{'Model':<12} {'Params':<10} {'Weights':<12} {'Activations':<12} {'Total':<12} {'kFLOPs':<12} {'Accuracy':<10}")
    print("-" * 110)
    
    for name in ['TinyPFN', 'MLP', 'XGBoost']:
        if name in results:
            r = results[name]
            params = r.get('params', '-')
            print(f"{name:<12} {str(params):<10} {r['weights_kb']:>10.2f} KB {r['activations_kb']:>10.2f} KB {r['total_kb']:>10.2f} KB {r['kflops']:>10.2f} {r['accuracy']:>9.4f}")
    
    print("=" * 110)
    
    # Linear vs Standard attention comparison
    if 'TinyPFN' in results and 'kflops_standard' in results['TinyPFN']:
        print("\n LINEAR vs STANDARD ATTENTION:")
        r = results['TinyPFN']
        print(f"  Linear attention:   {r['kflops']:.2f} kFLOPs")
        print(f"  Standard attention: {r['kflops_standard']:.2f} kFLOPs")
        print(f"  Reduction factor:   {r['kflops_standard']/r['kflops']:.2f}x")
    
    # Memory matching verification
    print("\n MEMORY MATCHING VERIFICATION:")
    tinypfn_mem = results['TinyPFN']['total_kb']
    for name in ['MLP', 'XGBoost']:
        if name in results:
            diff = abs(results[name]['total_kb'] - tinypfn_mem)
            pct = diff / tinypfn_mem * 100
            status = "✓" if pct < 5 else "⚠"
            print(f"  {name}: {results[name]['total_kb']:.2f} KB (diff: {diff:.2f} KB, {pct:.1f}%) {status}")
    
    # Key insights
    print("\n KEY INSIGHTS:")
    print(f"  - TinyPFN: {results['TinyPFN']['accuracy']:.2%} accuracy (zero-shot, no training on this data)")
    print(f"  - MLP:     {results['MLP']['accuracy']:.2%} accuracy (trained on this data, {results['MLP']['params']} params)")
    print(f"  - XGBoost: {results['XGBoost']['accuracy']:.2%} accuracy (trained on this data, {results['XGBoost']['n_estimators']} trees)")
    
    return results


def main():
    # Load Breast Cancer dataset
    X, y = load_breast_cancer(return_X_y=True)
    results = run_benchmark(X, y, dataset_name="Breast Cancer Wisconsin")
    
    X, y = load_iris(return_X_y=True)
    results_iris = run_benchmark(X, y, dataset_name="Iris")
    
    X, y = load_wine(return_X_y=True)
    results_wine = run_benchmark(X, y, dataset_name="Wine")


if __name__ == "__main__":
    main()