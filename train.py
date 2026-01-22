import random
import time

import h5py
import numpy as np
import schedulefree
import torch
from model import TinyPFNClassifier, TinyPFNModel
from sklearn.datasets import *
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_randomness_seed(0)

def get_default_device():
    device = "cpu"
    if torch.backends.mps.is_available(): device = "mps"
    if torch.cuda.is_available(): device = "cuda"
    return device

datasets = []
datasets.append(train_test_split(*load_breast_cancer(return_X_y=True), test_size=0.5, random_state=0))

def eval(classifier):
    scores = {
        "roc_auc": 0,
        "acc": 0,
        "balanced_acc": 0
    }
    for X_train, X_test, y_train, y_test in datasets:
         classifier.fit(X_train, y_train)
         prob = classifier.predict_proba(X_test)
         pred = prob.argmax(axis=1)
         if prob.shape[1] == 2:
            prob = prob[:, 1]

         scores["roc_auc"] += float(roc_auc_score(y_test, prob))
         scores["acc"] += float(accuracy_score(y_test, pred))
         scores["balanced_acc"] += float(balanced_accuracy_score(y_test, pred))
    scores = {k:v/len(datasets) for k,v in scores.items()}
    return scores

def train(model: TinyPFNModel, prior: DataLoader,
          lr: float = 1e-4, device: torch.device = None, steps_per_eval=10, eval_func=None):
    
    if not device:
        device = get_default_device()
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    model.train()
    optimizer.train()

    train_time = 0
    eval_history = []
    try:
        for step, full_data in enumerate(prior):
            step_start_time = time.time()
            train_test_split_index = full_data["train_test_split_index"]
            x = full_data["x"].to(device)
            y = full_data["y"].to(device)
            
            if torch.isnan(x).any() or torch.isnan(y).any():
                print(f"Step {step}: NaN in input data, skip..")
                continue
            
            data = (x, y[:, :train_test_split_index])
            targets = y[:, train_test_split_index:]

            output = model(data, train_test_split_index=train_test_split_index)
            
            if torch.isnan(output).any():
                print(f"Step {step}: NaN in model output, skip..")
                continue

            targets = targets.reshape((-1,)).to(torch.long)
            output = output.view(-1, output.shape[-1])

            loss = criterion(output, targets).mean()
            if torch.isnan(loss):
                print(f"Step {step}: NaN in loss, skip..")
                continue
            
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"Step {step}: NaN in gradient {name}")

            total_loss = loss.cpu().detach().item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            optimizer.zero_grad()
            step_train_duration = time.time() - step_start_time
            train_time += step_train_duration

            if step % steps_per_eval == steps_per_eval-1 and eval_func is not None:
                model.eval()
                optimizer.eval()

                classifier = TinyPFNClassifier(model, device)
                scores = eval_func(classifier)
                eval_history.append((train_time, scores))
                score_str = " | ".join([f"{k} {v:7.4f}" for k,v in scores.items()])
                print(f"time {train_time:7.1f}s | loss {total_loss:7.4f} | {score_str}")

                model.train()
                optimizer.train()
            elif step % steps_per_eval == steps_per_eval-1 and eval_func is None:
                print(f"time {train_time:7.1f}s | loss {total_loss:7.4f}")
    except KeyboardInterrupt:
        pass

    return model, eval_history


class PriorDumpDataLoader(DataLoader):
    """DataLoader that loads synthetic prior data from an HDF5 dump with shuffle and class filtering."""

    def __init__(self, filename, num_steps, batch_size, device=None, 
                 min_classes=1, shuffle=True):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.device = device
        self.min_classes = min_classes
        self.shuffle = shuffle
        
        if device is None:
            device = get_default_device()
        
        # Pre-compute valid indices (datasets with >= min_classes)
        print(f"Scanning prior for datasets with >= {min_classes} classes...")
        with h5py.File(self.filename, "r") as f:
            self.max_num_classes = f["max_num_classes"][0]
            total_datasets = f["X"].shape[0]
            
            if min_classes > 1:
                self.valid_indices = []
                y_all = f["y"][:]
                for i in range(total_datasets):
                    n_classes = len(np.unique(y_all[i][~np.isnan(y_all[i])]))
                    if n_classes >= min_classes:
                        self.valid_indices.append(i)
                self.valid_indices = np.array(self.valid_indices)
                print(f"Found {len(self.valid_indices)} valid datasets out of {total_datasets} ({100*len(self.valid_indices)/total_datasets:.1f}%)")
            else:
                self.valid_indices = np.arange(total_datasets)
                print(f"Using all {total_datasets} datasets")

    def __iter__(self):
        # Shuffle indices at the start of each epoch
        indices = self.valid_indices.copy()
        if self.shuffle:
            np.random.shuffle(indices)
        
        pointer = 0
        
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                # Get batch indices
                if pointer + self.batch_size > len(indices):
                    # Reshuffle and restart if we run out
                    if self.shuffle:
                        np.random.shuffle(indices)
                    pointer = 0
                
                batch_indices = indices[pointer:pointer + self.batch_size]
                pointer += self.batch_size
                
                # Sort indices for h5py (requires increasing order)
                sorted_order = np.argsort(batch_indices)
                sorted_indices = batch_indices[sorted_order]
                
                # Load data with sorted indices
                num_features = int(f["num_features"][sorted_indices].max())
                num_datapoints_batch = f["num_datapoints"][sorted_indices]
                max_seq_in_batch = int(num_datapoints_batch.max())
                
                # Load batch data
                x_sorted = f["X"][sorted_indices, :max_seq_in_batch, :num_features]
                y_sorted = f["y"][sorted_indices, :max_seq_in_batch]
                
                # Restore original shuffle order
                inverse_order = np.argsort(sorted_order)
                x = torch.from_numpy(x_sorted[inverse_order])
                y = torch.from_numpy(y_sorted[inverse_order])
                
                train_test_split_index = f["single_eval_pos"][sorted_indices[0]]

                yield dict(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    train_test_split_index=int(train_test_split_index),
                )

    def __len__(self):
        return self.num_steps


if __name__ == "__main__":
    device = get_default_device()
    model = TinyPFNModel(
        embedding_size=16,
        num_attention_heads=1,
        mlp_hidden_size=32, 
        num_layers=4,
        num_outputs=3,
        use_linear_attention=True
    )
    
    prior = PriorDumpDataLoader(
        "tabicl_300k_150x5_exact3class.h5",  
        num_steps=2500,              
        batch_size=32, 
        device=device,
        min_classes=3,                # use only datasets with 3+ classes for multiclass training
        shuffle=True                  # shuffle to avoid repeating same data sequence between epochs
    )
    
    model, history = train(model, prior, lr=4e-3, steps_per_eval=25)
    
    print("\n" + "="*60)
    print("Final evaluation:")
    print("="*60)
    print(eval(TinyPFNClassifier(model, device)))

    for name, loader in [("Breast Cancer", load_breast_cancer), 
                        ("Iris", load_iris), 
                        ("Wine", load_wine)]:
        X, y = loader(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        
        X_train, X_test = X_train[:, :10], X_test[:, :10]
        
        classifier = TinyPFNClassifier(model, device)
        classifier.fit(X_train[:100], y_train[:100])
        probs = classifier.predict_proba(X_test[:100])
        preds = probs.argmax(axis=1)
        
        print(f"\n{name}:")
        print(f"  Predictions: {np.bincount(preds, minlength=4)}")
        print(f"  True labels: {np.bincount(y_test[:100], minlength=4)}")
        print(f"  Accuracy: {accuracy_score(y_test[:100], preds):.4f}")
        print(f"  Sample probs:\n{probs[:3]}")

    # ---- save pretrained model ----
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_kwargs": {
            "embedding_size": 16,
            "num_attention_heads": 1,
            "mlp_hidden_size": 32,
            "num_layers": 4,
            "num_outputs": 3
        }
    }, "tinypfn_prior_trained_multiclass_ticl_v4.pt")

    print("\nModel saved as tinypfn_prior_trained_multiclass_ticl_v4.pt")