"""
Validate TinyPFN Python implementation with the EXACT same data as ESP32.
This uses the same 50 train + 50 test samples to compare accuracy.
"""

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import sys
sys.path.append('.')
from model import TinyPFNModel, TinyPFNClassifier

np.random.seed(42)
torch.manual_seed(42)


def main():
    print("=" * 60)
    print("VALIDATION: Python vs ESP32 with identical data")
    print("50 train + 50 test, 5 features")
    print("=" * 60)
    
    # Load and prepare data - EXACT same as ESP32
    X, y = load_breast_cancer(return_X_y=True)
    
    # Select top 5 features
    selector = SelectKBest(f_classif, k=5)
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    feature_names = load_breast_cancer().feature_names
    
    print(f"\nSelected features: {[feature_names[i] for i in selected_indices]}")
    
    # Split - same as benchmark
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_selected, y, test_size=0.5, random_state=42, stratify=y
    )
    
    # Limit to 50 each - SAME AS ESP32
    n_train, n_test = 50, 50
    X_train = X_train_full[:n_train]
    y_train = y_train_full[:n_train]
    X_test = X_test_full[:n_test]
    y_test = y_test_full[:n_test]
    
    print(f"\nData: {n_train} train + {n_test} test = {n_train + n_test} total")
    print(f"Train distribution: {sum(y_train == 0)} malignant, {sum(y_train == 1)} benign")
    print(f"Test distribution: {sum(y_test == 0)} malignant, {sum(y_test == 1)} benign")
    
    # Load model
    model_path = "tinypfn_prior_trained.pt"
    device = torch.device('cpu')
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_kwargs = checkpoint.get('model_kwargs', {
        'embedding_size': 8,
        'num_attention_heads': 1,
        'mlp_hidden_size': 8,
        'num_layers': 1,
        'num_outputs': 2,
        'use_linear_attention': True
    })
    
    print(f"\nModel config: {model_kwargs}")
    
    model = TinyPFNModel(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Run inference
    classifier = TinyPFNClassifier(model, device)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    # Confusion matrix components
    tp = sum((y_pred == 0) & (y_test == 0))  # Malignant correct
    tn = sum((y_pred == 1) & (y_test == 1))  # Benign correct
    fp = sum((y_pred == 0) & (y_test == 1))  # Benign misclassified as Malignant
    fn = sum((y_pred == 1) & (y_test == 0))  # Malignant misclassified as Benign
    
    # Print detailed results
    print("\n" + "-" * 60)
    print("PREDICTIONS (first 10 and last 5)")
    print("-" * 60)
    
    for t in range(n_test):
        prob_m = y_proba[t, 0]  # P(Malignant)
        prob_b = y_proba[t, 1]  # P(Benign)
        pred = y_pred[t]
        actual = y_test[t]
        
        if t < 10 or t >= n_test - 5:
            status = "OK" if pred == actual else "MISS"
            actual_str = "Malig" if actual == 0 else "Benign"
            pred_str = "Malig" if pred == 0 else "Benign"
            print(f"Test {t:2d}: P(M)={prob_m:.3f} P(B)={prob_b:.3f} | True:{actual_str:<6} Pred:{pred_str:<6} [{status}]")
        elif t == 10:
            print("... (skipping middle results) ...")
    
    # Summary
    print("\n" + "=" * 60)
    print("                   PYTHON RESULTS")
    print("=" * 60)
    print(f"Accuracy:          {sum(y_pred == y_test)}/{n_test} = {accuracy * 100:.2f}%")
    print(f"Balanced Accuracy: {balanced_acc * 100:.2f}%")
    print("-" * 60)
    print(f"True Positives:  {tp} (Malignant correctly identified)")
    print(f"True Negatives:  {tn} (Benign correctly identified)")
    print(f"False Positives: {fp} (Benign misclassified as Malignant)")
    print(f"False Negatives: {fn} (Malignant misclassified as Benign)")
    print("=" * 60)
    
    # Comparison with ESP32
    print("\n" + "=" * 60)
    print("                   COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'Python':<15} {'ESP32':<15}")
    print("-" * 60)
    print(f"{'Accuracy':<25} {accuracy * 100:.2f}%{'':<10} 94.00%")
    print(f"{'Balanced Accuracy':<25} {balanced_acc * 100:.2f}%{'':<10} 93.18%")
    print(f"{'True Positives':<25} {tp:<15} 19")
    print(f"{'True Negatives':<25} {tn:<15} 28")
    print(f"{'False Positives':<25} {fp:<15} 0")
    print(f"{'False Negatives':<25} {fn:<15} 3")
    print("=" * 60)
    
    if abs(accuracy - 0.94) < 0.02:
        print("\n Python and ESP32 results MATCH! Implementation is correct.")
    else:
        print(f"\n Results differ by {abs(accuracy - 0.94) * 100:.1f}%. Check implementation.")


if __name__ == "__main__":
    main()
