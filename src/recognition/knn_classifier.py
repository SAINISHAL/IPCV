"""PCA + K-Nearest Neighbors Classifier with Unknown Rejection."""
import pickle
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from collections import Counter
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("KNNClassifier")


class KNNClassifier:
    """K-Nearest Neighbors classifier operating on PCA feature weights."""

    UNKNOWN_LABEL = "UNKNOWN"

    def __init__(self, n_neighbors: int = 3, threshold: Optional[float] = None):
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.train_weights: Optional[np.ndarray] = None
        self.train_labels: Optional[np.ndarray] = None
        self.is_trained = False

    def fit(self, X_weights: np.ndarray, y_labels: np.ndarray) -> "KNNClassifier":
        """Fit KNN model with training projections."""
        self.train_weights = np.asarray(X_weights, dtype=np.float32)
        self.train_labels = np.asarray(y_labels)
        self.is_trained = True
        logger.info(f"Fitted KNN (k={self.n_neighbors}) with {len(self.train_weights)} samples.")
        return self

    def calibrate_threshold(self, val_weights: np.ndarray, val_labels: np.ndarray,
                            factor: float = 1.3) -> float:
        """Calibrate rejection threshold on validation data."""
        distances = []
        for w, true_lbl in zip(val_weights, val_labels):
            dists = np.linalg.norm(self.train_weights - w, axis=1)
            k_indices = np.argsort(dists)[:self.n_neighbors]
            k_labels = [self.train_labels[i] for i in k_indices]
            majority_label = Counter(k_labels).most_common(1)[0][0]
            if majority_label == true_lbl:
                distances.append(np.mean(dists[k_indices]))

        if distances:
            self.threshold = float(np.mean(distances) * factor)
        else:
            self.threshold = 5.0
        logger.info(f"Calibrated KNN threshold: {self.threshold:.3f}")
        return self.threshold

    def predict_single(self, weight_vector: np.ndarray) -> Tuple[str, float]:
        """Predict class and average neighbor distance for a single weight vector."""
        if not self.is_trained:
            raise ValueError("Classifier is not trained.")
            
        dists = np.linalg.norm(self.train_weights - weight_vector, axis=1)
        k_indices = np.argsort(dists)[:self.n_neighbors]
        k_dists = dists[k_indices]
        avg_dist = float(np.mean(k_dists))

        if self.threshold is not None and avg_dist > self.threshold:
            return self.UNKNOWN_LABEL, avg_dist

        k_labels = [str(self.train_labels[i]) for i in k_indices]
        vote_counts = Counter(k_labels)
        winner_label = vote_counts.most_common(1)[0][0]
        return winner_label, avg_dist

    def predict(self, X_weights: np.ndarray) -> List[str]:
        """Predict labels for batch of feature weights."""
        return [self.predict_single(w)[0] for w in X_weights]

    def save(self, filepath: str or Path) -> Path:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({
                "n_neighbors": self.n_neighbors,
                "threshold": self.threshold,
                "train_weights": self.train_weights,
                "train_labels": self.train_labels
            }, f)
        return p

    def load(self, filepath: str or Path) -> "KNNClassifier":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.n_neighbors = data["n_neighbors"]
        self.threshold = data["threshold"]
        self.train_weights = data["train_weights"]
        self.train_labels = data["train_labels"]
        self.is_trained = True
        return self

