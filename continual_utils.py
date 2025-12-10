import numpy as np
import torch
from typing import Tuple
from drift import ReservoirBuffer, DriftManager
from model import TinyPFNModel, TinyPFNClassifier

# -------------------------
# DEFAULT CONFIG
# -------------------------
STREAM_BATCH_SIZE = 32
MEMORY_CAPACITY = 500
DRIFT_WARMUP = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Model loader
# -------------------------
def load_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"]
    model_kwargs = checkpoint["model_kwargs"]
    model = TinyPFNModel(**model_kwargs).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    classifier = TinyPFNClassifier(model, DEVICE)
    return model, classifier, model_kwargs

# -------------------------
# Embeddings extractor (wrapper)
# -------------------------
def extract_embeddings(model: TinyPFNModel, batch_X: np.ndarray, device=DEVICE) -> np.ndarray:
    """
    Returns embeddings from model.extract_feature_embeddings.
    Accepts batch_X shape (N_rows, n_features) or (B, R, C) depending on your extractor.
    We normalize to return (N_rows, embedding_dim) or (B, E) depending on model behavior.
    """
    # model.extract_feature_embeddings expects (N_rows, num_features) -> returns (1,R,C,E)
    emb = model.extract_feature_embeddings(batch_X, device=device)  # (1, R, C, E) per your model
    emb = np.asarray(emb)
    if emb.ndim == 4:
        # collapse rows x cols to a single per-row vector by mean over columns and embedding
        # Here we produce one vector per original row: shape (R, E)
        B, R, C, E = emb.shape
        # average over columns to get (B, R, E), squeeze batch
        per_row = emb.reshape(B, R, C, E).mean(axis=2).squeeze(0)  # (R, E)
        return per_row
    elif emb.ndim == 3:
        return emb.squeeze(0)
    else:
        return emb

# -------------------------
# Train adapters/prompts on memory
# -------------------------
def train_adapters_prompts(model: TinyPFNModel, classifier: TinyPFNClassifier,
                           mem_X: np.ndarray, mem_y: np.ndarray,
                           epochs: int = 5, lr: float = 1e-3, batch_size: int = 64,
                           device=DEVICE, verbose: bool = False) -> dict:
    """
    Trains ONLY adapters and prompts (if present). Expects mem_y to contain numeric labels or np.nan.
    Returns training stats dict.
    """
    # filter labeled
    if mem_X.size == 0:
        return {"trained": False, "reason": "no_memory"}
    labeled_mask = ~np.isnan(mem_y)
    if labeled_mask.sum() == 0:
        return {"trained": False, "reason": "no_labels"}
    X_l = mem_X[labeled_mask]
    y_l = mem_y[labeled_mask].astype(int)

    # freeze base - keep adapters and prompt params trainable
    model.freeze_base(freeze_feature_encoder=True, freeze_transformer=True, freeze_decoder=False)

    # collect parameters that require grad
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    if len(params) == 0:
        return {"trained": False, "reason": "no_trainable_params"}

    model.train()
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    n = len(X_l)
    for ep in range(epochs):
        # simple shuffling batches
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = idx[start:end]
            x_batch = torch.from_numpy(X_l[batch_idx]).unsqueeze(0).float().to(device)  # PFN expects batch dim
            y_train = torch.from_numpy(y_l[batch_idx]).unsqueeze(0).float().to(device)
            # PFN style: we need to present the model with a train_test_split_index
            train_test_split_index = x_batch.shape[1]  # treat all as train for supervised fine-tune
            optimizer.zero_grad()
            out = model((x_batch, y_train), train_test_split_index=train_test_split_index)
            # out shape: (B, num_targets, num_outputs) -> flatten
            out = out.view(-1, out.shape[-1])
            # targets: use zeros because y_train was used as train; instead we want to supervise next positions.
            # For simplicity create dummy targets equal to y_batch last positions (we use y_train itself)
            targets = y_train.view(-1).long()
            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
    # unfreeze base (optionally)
    model.unfreeze_all()
    model.eval()
    return {"trained": True, "n_samples": int(labeled_mask.sum())}

# -------------------------
# Continual inference + active update
# -------------------------
def continual_inference_with_active_updates(model: TinyPFNModel, classifier: TinyPFNClassifier,
                                            stream_X: np.ndarray, stream_y: np.ndarray = None,
                                            memory_capacity: int = MEMORY_CAPACITY,
                                            stream_batch_size: int = STREAM_BATCH_SIZE,
                                            drift_warmup: int = DRIFT_WARMUP,
                                            pca_batch_size: int = 128,
                                            adapter_train_cfg: dict = None):
    """
    Stream simulation:
      - extract embeddings per incoming batch
      - populate reservoir memory
      - after warmup fit PCA reference using memory embeddings
      - detect drift each step (PCA + optional entropy)
      - if drift -> train adapters/prompts using labeled samples in memory (if available)
      - update PCA reference from memory after adaptation
    Returns: dict with predictions, truths, drift_flags and logs.
    """
    memory = ReservoirBuffer(capacity=memory_capacity, seed=42)
    drift_manager = DriftManager()
    logs = {"pca_dist": [], "entropy": [], "drift": [], "batch_acc": []}

    num_samples = stream_X.shape[0]
    pred_list = []
    true_list = []
    drift_flags = []

    # adapter training cfg defaults
    if adapter_train_cfg is None:
        adapter_train_cfg = {"epochs": 3, "lr": 1e-3, "batch_size": 32}

    # streaming loop
    for start in range(0, num_samples, stream_batch_size):
        end = min(start + stream_batch_size, num_samples)
        batch_X = stream_X[start:end]
        batch_y = None if stream_y is None else stream_y[start:end]

        # 1) extract embeddings (per-row vectors)
        emb_rows = extract_embeddings(model, batch_X)  # shape (R, E)
        if emb_rows is None or emb_rows.size == 0:
            window_emb = np.zeros((1, 1))
        else:
            # we can feed the detector with per-row embeddings (better) or mean per-window
            window_emb = emb_rows  # (M, D)

        # 2) if we have enough memory and PCA not fitted, fit PCA using memory
        if len(memory) >= drift_warmup and not drift_manager.pca._fitted:
            # build embeddings for memory (sample all or a subset)
            mem_X_all, _ = memory.all()
            # compute embeddings for mem samples (careful: memory stores raw X rows)
            mem_emb_list = []
            for xi in mem_X_all:
                mem_e = extract_embeddings(model, xi)
                # if per-row, take mean to represent dataset sample
                mem_emb_list.append(mem_e.mean(axis=0))
            mem_emb = np.vstack(mem_emb_list)
            drift_manager.fit_memory(mem_emb, batch_size=pca_batch_size)

        # 3) detect drift (PCA + entropy optional)
        probs = None
        if hasattr(classifier, "predict_proba"):
            try:
                probs = classifier.predict_proba(batch_X)
            except Exception:
                probs = None
        if len(memory) >= drift_warmup and drift_manager.pca._fitted:
            pca_drift_info = drift_manager.check(window_embeddings=window_emb, probs=probs)
            is_drift = pca_drift_info["drift"]
            pca_dist = pca_drift_info.get("pca_dist")
            entropy = pca_drift_info.get("entropy")
        else:
            is_drift = False
            pca_dist = None
            entropy = None

        # 4) update memory
        memory.add_batch(batch_X, batch_y)

        # 5) prediction: fit classifier on labeled mem only (if any), else skip fit
        mem_X_all, mem_y_all = memory.all()
        preds = np.zeros(len(batch_X), dtype=int)
        trained_flag = False
        if len(mem_X_all) > 0:
            # only use labeled examples for fit
            labeled_mask = ~np.isnan(mem_y_all)
            if labeled_mask.sum() > 0:
                try:
                    classifier.fit(mem_X_all[labeled_mask], mem_y_all[labeled_mask].astype(int))
                    trained_flag = True
                except Exception:
                    # fallback: attempt to convert labels
                    try:
                        classifier.fit(mem_X_all, np.nan_to_num(mem_y_all, nan=0).astype(int))
                        trained_flag = True
                    except Exception:
                        trained_flag = False
            # predict
            try:
                preds = classifier.predict(batch_X)
            except Exception:
                preds = np.zeros(len(batch_X), dtype=int)
        else:
            preds = np.zeros(len(batch_X), dtype=int)

        # 6) if drift -> active update adapters/prompts (supervised) and update PCA reference
        if is_drift:
            # train adapters/prompts using memory (if labeled samples exist)
            mem_X_all, mem_y_all = memory.all()
            train_stats = train_adapters_prompts(model, classifier, mem_X_all, mem_y_all,
                                                 epochs=adapter_train_cfg.get("epochs", 3),
                                                 lr=adapter_train_cfg.get("lr", 1e-3),
                                                 batch_size=adapter_train_cfg.get("batch_size", 32),
                                                 device=DEVICE)
            # after adaptation update PCA reference using memory embeddings (representative)
            # build mem embeddings
            mem_emb_list = []
            for xi in mem_X_all:
                me = extract_embeddings(model, xi)
                mem_emb_list.append(me.mean(axis=0))
            mem_emb = np.vstack(mem_emb_list)
            drift_manager.update_reference(mem_emb)

        # 7) logging & store
        pred_list.append(preds)
        if batch_y is not None:
            true_list.append(batch_y)
            batch_acc = float((preds == batch_y).mean())
        else:
            batch_acc = None
        drift_flags.append(bool(is_drift))
        logs["pca_dist"].append(pca_dist)
        logs["entropy"].append(entropy)
        logs["drift"].append(bool(is_drift))
        logs["batch_acc"].append(batch_acc)

    predictions = np.concatenate(pred_list) if len(pred_list) > 0 else np.array([])
    truths = np.concatenate(true_list) if len(true_list) > 0 else np.array([])
    drift_arr = np.array(drift_flags, dtype=bool)
    return {
        "predictions": predictions,
        "truths": truths,
        "drift_flags": drift_arr,
        "logs": logs
    }
