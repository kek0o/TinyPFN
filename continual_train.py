import os
import numpy as np
import torch
from continual_utils import (
    load_model,
    continual_inference_with_active_updates,
    STREAM_BATCH_SIZE,
    MEMORY_CAPACITY,
    DRIFT_WARMUP,
    DEVICE
)

def continual_train(stream_X: np.ndarray, stream_y: np.ndarray,
                    checkpoint_dir: str = "checkpoints",
                    model_checkpoint: str = None,
                    memory_capacity: int = MEMORY_CAPACITY,
                    stream_batch_size: int = STREAM_BATCH_SIZE,
                    drift_warmup: int = DRIFT_WARMUP,
                    adapter_train_cfg: dict = None,
                    save_every: int = 100,
                    verbose: bool = True):
    """
    Perform continual training on a streaming dataset with active drift adaptation.
    
    Args:
        stream_X, stream_y: np.ndarray of data and labels.
        checkpoint_dir: folder to save checkpoints.
        model_checkpoint: path to a pre-trained model checkpoint.
        memory_capacity: reservoir memory size.
        stream_batch_size: batch size for streaming.
        drift_warmup: number of batches before PCA drift detection.
        adapter_train_cfg: dict with 'epochs', 'lr', 'batch_size' for adapter training.
        save_every: save checkpoint every N batches.
        verbose: print progress.
    Returns:
        dict with 'predictions', 'truths', 'drift_flags', 'logs'.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1) Load or initialize model
    if model_checkpoint:
        model, classifier, model_kwargs = load_model(model_checkpoint)
        if verbose:
            print(f"[INFO] Loaded model from {model_checkpoint}")
    else:
        raise ValueError("Please provide a model checkpoint path to initialize the model.")

    # 2) Run streaming + continual inference
    results = continual_inference_with_active_updates(
        model=model,
        classifier=classifier,
        stream_X=stream_X,
        stream_y=stream_y,
        memory_capacity=memory_capacity,
        stream_batch_size=stream_batch_size,
        drift_warmup=drift_warmup,
        adapter_train_cfg=adapter_train_cfg
    )

    # 3) Optionally save final checkpoint
    final_ckpt_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_kwargs": model_kwargs
    }, final_ckpt_path)
    if verbose:
        print(f"[INFO] Final model saved to {final_ckpt_path}")

    return results


if __name__ == "__main__":
    # -----------------------------
    # Demo / test
    # -----------------------------
    # Generar datos de prueba
    N, F = 500, 10
    stream_X = np.random.randn(N, F).astype(np.float32)
    stream_y = np.random.randint(0, 2, size=N).astype(np.float32)

    # Path a checkpoint pre-entrenado de TinyPFN
    ckpt_path = "tinypfn_prior_trained.pt"

    results = continual_train(stream_X, stream_y, model_checkpoint=ckpt_path)

    print("Predictions:", results["predictions"][:10])
    print("Drift flags:", results["drift_flags"][:10])
