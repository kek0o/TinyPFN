import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# TinyPFN
from model import TinyPFNModel, TinyPFNClassifier
# TabPFN classic
from tabpfn import TabPFNClassifier

# Drift y memoria
from drift import ReservoirBuffer, DriftManager

# -------------------------
# CONFIG
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STREAM_BATCH_SIZE = 32
MEMORY_CAPACITY = 500
DRIFT_WARMUP = 5

# -------------------------
# FUNCIONES AUXILIARES
# -------------------------
def extract_embeddings(model, X: np.ndarray):
    """Extrae embeddings de features usando el encoder del modelo."""
    return model.extract_feature_embeddings(X, device=DEVICE)

def continual_inference_cl_metrics(model_clf, stream_X, stream_y):
    """
    Simula aprendizaje incremental y calcula métricas CL:
    - Accuracy batch
    - Promedio Accuracy
    - Forgetting promedio
    """
    memory = ReservoirBuffer(capacity=MEMORY_CAPACITY, seed=42)
    drift_manager = DriftManager()

    pred_list, true_list, drift_flags = [], [], []
    batch_acc_list = []
    past_batches = []

    for start in range(0, stream_X.shape[0], STREAM_BATCH_SIZE):
        end = min(start + STREAM_BATCH_SIZE, stream_X.shape[0])
        batch_X = stream_X[start:end]
        batch_y = stream_y[start:end]

        # Embeddings
        if hasattr(model_clf.model, "extract_feature_embeddings"):
            embeddings = extract_embeddings(model_clf.model, batch_X)
            if embeddings.ndim == 3:
                embeddings_flat = embeddings.mean(axis=(1,2))
            else:
                embeddings_flat = embeddings
        else:
            embeddings_flat = None

        # drift detection
        if len(memory.storage) >= DRIFT_WARMUP and embeddings_flat is not None:
            drift_info = drift_manager.check(window_embeddings=embeddings_flat)
            is_drift = drift_info["drift"]
        else:
            is_drift = False
            drift_info = {}

        # memory update
        memory.add_batch(batch_X, batch_y)

        # incremental prediction
        mem_X, mem_y = memory.all()
        if len(mem_X) > 0:
            model_clf.fit(mem_X, mem_y)
            preds = model_clf.predict(batch_X)
        else:
            preds = np.zeros(batch_X.shape[0], dtype=int)

        batch_acc = (preds == batch_y).mean()
        batch_acc_list.append(batch_acc)

        pred_list.append(preds)
        true_list.append(batch_y)
        drift_flags.append(is_drift)
        past_batches.append((batch_X, batch_y))

        print(f"Batch {start}-{end} | Drift: {is_drift} | PCA dist: {drift_info.get('pca_dist')} | Entropy: {drift_info.get('entropy')} | Batch Acc: {batch_acc:.4f}")

    # compute forgetting
    forgetting_list = []
    for i, (bX, bY) in enumerate(past_batches):
        preds_i = model_clf.predict(bX)
        acc_i = (preds_i == bY).mean()
        forget = batch_acc_list[i] - acc_i
        forgetting_list.append(forget)

    predictions = np.concatenate(pred_list)
    truths = np.concatenate(true_list)
    drift_flags = np.array(drift_flags)
    avg_acc = np.mean(batch_acc_list)
    avg_forgetting = np.mean(forgetting_list)

    return predictions, truths, drift_flags, avg_acc, avg_forgetting

# -------------------------
# load pretrained models
# -------------------------
def load_checkpoint(path, model_class):
    checkpoint = torch.load(path, map_location=DEVICE)
    model_kwargs = checkpoint["model_kwargs"]
    state_dict = checkpoint["model_state_dict"]
    model = model_class(**model_kwargs).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    clf = TinyPFNClassifier(model, DEVICE)
    return clf
tiny_clf = load_checkpoint("tinypfn_prior_trained.pt", TinyPFNModel)

# TabPFN classic
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
tabpfn_clf = TabPFNClassifier()
tabpfn_clf.fit(X_train, y_train)

# -------------------------
# global avaluation
# -------------------------
def evaluate_global(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    if isinstance(clf, TabPFNClassifier):
        prob = clf.predict_proba(X_test)[:,1]
        pred = clf.predict(X_test)
    else:
        prob = clf.predict_proba(X_test)[:,1]
        pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    roc = roc_auc_score(y_test, prob)
    return acc, roc
def eval(classifier):
    scores = {
        "roc_auc": 0,
        "acc": 0,
        "balanced_acc": 0
    }
    for  X_train, X_test, y_train, y_test in datasets:
         classifier.fit(X_train, y_train)
         prob = classifier.predict_proba(X_test)
         pred = prob.argmax(axis=1) # avoid a second forward pass by not calling predict
         if prob.shape[1]==2:
             prob = prob[:,:1]
         scores["roc_auc"] += float(roc_auc_score(y_test, prob, multi_class="ovr"))
         scores["acc"] += float(accuracy_score(y_test, pred))
         scores["balanced_acc"] += float(balanced_accuracy_score(y_test, pred))
    scores = {k:v/len(datasets) for k,v in scores.items()}
    return scores
print("=== global avaluation ===")
for name, clf in [("TinyPFN", tiny_clf), ("TabPFN", tabpfn_clf)]:
    acc, roc = evaluate_global(clf, X_train, y_train, X_test, y_test)
    print(f"{name}: Accuracy {acc:.4f}, ROC AUC {roc:.4f}")

# -------------------------
# global continual learning
# -------------------------
print("\n=== Continual learning avaluation (simulated) ===")
for name, clf in [("TinyPFN", tiny_clf)]:
    preds, truths, drifts, avg_acc, avg_forgetting = continual_inference_cl_metrics(clf, X_test, y_test)
    roc = roc_auc_score(truths, clf.predict_proba(X_test)[:,1])
    proportion_drift = drifts.mean()
    print(f"{name}: Accuracy stream: {avg_acc:.4f}, ROC AUC global: {roc:.4f}, Total drifts: {drifts.sum()}, Proporción batches con drift: {proportion_drift:.4f}, Avg Forgetting: {avg_forgetting:.4f}")
