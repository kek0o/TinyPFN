import numpy as np
from sklearn.decomposition import IncrementalPCA
import math

# -------------------------
# Reservoir Buffer
# -------------------------
class ReservoirBuffer:
    """
    Reservoir sampling buffer storing (x, y) pairs.
    Labels for unlabeled entries are stored as np.nan (float).
    """
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self.storage = []  # list of tuples (x, y)
        self.n_seen = 0
        self.rng = np.random.RandomState(seed)

    def add_batch(self, X_batch, y_batch=None):
        """
        X_batch: ndarray shape [N, ...]
        y_batch: ndarray shape [N,] or None
        Stores copies to avoid side-effects.
        """
        for i in range(len(X_batch)):
            x_item = np.copy(X_batch[i])
            if y_batch is None:
                y_item = np.nan
            else:
                # convert to float and np.nan for missing
                yi = y_batch[i]
                y_item = float(yi) if (yi is not None and not (isinstance(yi, float) and np.isnan(yi))) else np.nan
            self.n_seen += 1
            item = (x_item, y_item)
            if len(self.storage) < self.capacity:
                self.storage.append(item)
            else:
                j = self.rng.randint(0, self.n_seen)
                if j < self.capacity:
                    self.storage[j] = item

    def sample(self, k: int):
        k = min(k, len(self.storage))
        if k == 0:
            return np.array([]), np.array([])
        idx = self.rng.choice(len(self.storage), k, replace=False)
        Xs = np.array([self.storage[i][0] for i in idx])
        ys = np.array([self.storage[i][1] for i in idx], dtype=float)
        return Xs, ys

    def all(self):
        if not self.storage:
            return np.array([]), np.array([])
        Xs = np.array([t[0] for t in self.storage])
        ys = np.array([t[1] for t in self.storage], dtype=float)
        return Xs, ys

    def __len__(self):
        return len(self.storage)

# -------------------------
# PCA-based change detector (incremental)
# -------------------------
class PCAChangeDetector:
    """
    Incremental PCA-based detector.
    - Keeps an IncrementalPCA subspace (k components).
    - Computes Mahalanobis distance between the mean of projected current window
      and the mean of memory projection, using covariance estimated from memory.
    - Adaptive thresholding using median + k * std or median + k * MAD if requested.
    """
    def __init__(self, n_components: int = 3, threshold_multiplier: float = 3.0,
                 warmup: int = 5, batch_size_for_ipca: int = 64, use_mad: bool = False):
        self.n_components = int(n_components)
        self.threshold_multiplier = float(threshold_multiplier)
        self.warmup = int(warmup)
        self.ipca = None
        self.mem_projected = None
        self.mem_mean = None
        self.mem_cov_inv = None
        self.history_distances = []
        self.batch_size_for_ipca = int(batch_size_for_ipca)
        self.use_mad = use_mad
        self._fitted = False

    def fit_memory(self, embeddings, batch_size=None):
        """
        Fit/initialize IncrementalPCA using embeddings (N_samples, D).
        embeddings can be a big array; we partial_fit in chunks.
        """
        embeddings = np.asarray(embeddings)
        if embeddings.shape[0] < max(2, self.n_components):
            # not enough data
            self._fitted = False
            return False

        batch_size = batch_size or self.batch_size_for_ipca
        self.ipca = IncrementalPCA(n_components=min(self.n_components, embeddings.shape[1]))
        # partial_fit in chunks
        for i in range(0, embeddings.shape[0], batch_size):
            chunk = embeddings[i: i + batch_size]
            self.ipca.partial_fit(chunk)
        # project memory
        self.mem_projected = self.ipca.transform(embeddings)
        self.mem_mean = np.mean(self.mem_projected, axis=0)
        cov = np.cov(self.mem_projected, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        try:
            self.mem_cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self.mem_cov_inv = np.linalg.pinv(cov)
        self.history_distances = []
        self._fitted = True
        return True

    def incremental_update_reference(self, new_embeddings, recompute_cov=True):
        """
        Optionally update the reference memory stats with new embeddings.
        This does NOT keep full memory; it updates PCA by partial_fit and recomputes mem stats from provided
        embeddings argument (which should come from memory sample).
        """
        if self.ipca is None:
            return self.fit_memory(new_embeddings)
        new_embeddings = np.asarray(new_embeddings)
        # update ipca
        self.ipca.partial_fit(new_embeddings)
        # recompute projected memory stats if user provides a representative set
        proj = self.ipca.transform(new_embeddings)
        self.mem_projected = proj
        self.mem_mean = np.mean(proj, axis=0)
        cov = np.cov(proj, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        try:
            self.mem_cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self.mem_cov_inv = np.linalg.pinv(cov)
        return True

    def update(self, window_embeddings):
        """
        window_embeddings: ndarray (M, D)
        returns: (is_drift: bool, distance: float)
        """
        if not self._fitted or self.ipca is None:
            return False, 0.0
        W = np.asarray(window_embeddings)
        if W.shape[0] == 0:
            return False, 0.0
        proj = self.ipca.transform(W)
        mean_w = np.mean(proj, axis=0)
        diff = mean_w - self.mem_mean
        dist_sq = float(diff.T.dot(self.mem_cov_inv).dot(diff))
        dist = math.sqrt(max(dist_sq, 0.0))
        self.history_distances.append(dist)

        if len(self.history_distances) < self.warmup:
            return False, dist

        hist = np.array(self.history_distances[-500:])  # keep recent
        if self.use_mad:
            med = np.median(hist)
            mad = np.median(np.abs(hist - med)) + 1e-12
            thr = med + self.threshold_multiplier * mad
        else:
            thr = np.median(hist) + self.threshold_multiplier * np.std(hist)
        is_drift = dist > thr
        return bool(is_drift), float(dist)

# -------------------------
# Entropy-based detector
# -------------------------
def mean_entropy(probs: np.ndarray, eps: float = 1e-12):
    p = np.clip(probs, eps, 1.0)
    ent = -np.sum(p * np.log(p), axis=1)
    return float(np.mean(ent))

class EntropyDetector:
    """
    Detects rise in predictive entropy. Stores history and sets baseline after warmup.
    """
    def __init__(self, baseline_entropy: float = None, multiplier: float = 2.0, warmup: int = 20):
        self.baseline = baseline_entropy
        self.multiplier = multiplier
        self.warmup = warmup
        self.history = []

    def update(self, probs: np.ndarray):
        ent = mean_entropy(probs)
        self.history.append(ent)
        if self.baseline is None and len(self.history) >= self.warmup:
            self.baseline = float(np.median(self.history[-self.warmup:]))
            return False, ent
        if self.baseline is None:
            return False, ent
        thr = self.baseline * self.multiplier
        return (ent > thr), ent

# -------------------------
# DriftManager
# -------------------------
class DriftManager:
    """
    Wrapper combining PCA detector and optional entropy detector.
    """
    def __init__(self, pca_detector: PCAChangeDetector = None, entropy_detector: EntropyDetector = None,
                 use_entropy_as_secondary: bool = True):
        self.pca = pca_detector if pca_detector is not None else PCAChangeDetector()
        self.ent = entropy_detector if entropy_detector is not None else EntropyDetector()
        self.use_entropy_as_secondary = use_entropy_as_secondary

    def fit_memory(self, embeddings, batch_size=None):
        return self.pca.fit_memory(embeddings, batch_size=batch_size)

    def update_reference(self, embeddings):
        return self.pca.incremental_update_reference(embeddings)

    def check(self, window_embeddings=None, probs=None):
        res = {"drift": False, "pca_dist": None, "entropy": None, "entropy_drift": False}
        if window_embeddings is not None:
            pca_drift, pca_dist = self.pca.update(window_embeddings)
            res["pca_dist"] = pca_dist
            res["drift"] = res["drift"] or pca_drift
        if probs is not None and self.use_entropy_as_secondary:
            ent_drift, ent = self.ent.update(probs)
            res["entropy"] = ent
            res["entropy_drift"] = ent_drift
            res["drift"] = res["drift"] or ent_drift
        return res
