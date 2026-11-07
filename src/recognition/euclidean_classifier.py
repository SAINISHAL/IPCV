"""PCA + Euclidean Nearest Neighbor Classifier with Calibrated Unknown Rejection."""
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("EuclideanClassifier")


class EuclideanClassifier:
    """Classifies PCA projection weights using minimum Euclidean distance to enrolled prototypes."""

    UNKNOWN_LABEL = "UNKNOWN"

    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold
        self.train_weights: Optional[np.ndarray] = None  # (N_train, K)
        self.train_labels: Optional[np.ndarray] = None   # (N_train,)
        self.is_trained = False

    def fit(self, X_weights: np.ndarray, y_labels: np.ndarray) -> "EuclideanClassifier":
        """Store training weights and class labels."""
        self.train_weights = np.asarray(X_weights, dtype=np.float32)
        self.train_labels = np.asarray(y_labels)
        self.is_trained = True
        logger.info(f"Fitted Euclidean classifier with {len(self.train_weights)} prototypes.")
        return self

    def calibrate_threshold(self, val_weights: np.ndarray, val_labels: np.ndarray,
                            std_factor: float = 1.5) -> float:
        """Calibrate rejection threshold on validation data: threshold = mean(correct_dist) + factor * std(correct_dist)."""
        if not self.is_trained:
            raise ValueError("Classifier must be fitted before calibration.")
            
        distances = []
        for w, true_lbl in zip(val_weights, val_labels):
            dists = np.linalg.norm(self.train_weights - w, axis=1)
            nearest_idx = np.argmin(dists)
            pred_lbl = self.train_labels[nearest_idx]
            if pred_lbl == true_lbl:
                distances.append(dists[nearest_idx])

        if len(distances) > 0:
            mean_dist = float(np.mean(distances))
            std_dist = float(np.std(distances))
            self.threshold = mean_dist + std_factor * std_dist
            logger.info(f"Calibrated Euclidean threshold: {self.threshold:.3f} (mean={mean_dist:.3f}, std={std_dist:.3f})")
        else:
            # Fallback based on training intra-class distances
            all_dists = []
            for i, w in enumerate(self.train_weights):
                same_cls = (self.train_labels == self.train_labels[i])
                if np.sum(same_cls) > 1:
                    d = np.linalg.norm(self.train_weights[same_cls] - w, axis=1)
                    d = d[d > 0]
                    if len(d) > 0:
                        all_dists.append(np.min(d))
            self.threshold = float(np.mean(all_dists) * 1.5) if all_dists else 5.0
            logger.info(f"Fallback Euclidean threshold: {self.threshold:.3f}")
            
        return self.threshold

    def predict_single(self, weight_vector: np.ndarray) -> Tuple[str, float]:
        """Predict student identity for a single projection weight vector.
        
        Returns:
            Tuple of (predicted_identity, distance). If distance > threshold, returns (UNKNOWN, distance).
        """
        if not self.is_trained:
            raise ValueError("Classifier is not trained.")
            
        dists = np.linalg.norm(self.train_weights - weight_vector, axis=1)
        nearest_idx = int(np.argmin(dists))
        min_dist = float(dists[nearest_idx])
        nearest_label = str(self.train_labels[nearest_idx])

        if self.threshold is not None and min_dist > self.threshold:
            return self.UNKNOWN_LABEL, min_dist
            
        return nearest_label, min_dist

    def predict(self, X_weights: np.ndarray) -> List[str]:
        """Predict labels for a batch of weight vectors."""
        preds = []
        for w in X_weights:
            pred_lbl, _ = self.predict_single(w)
            preds.append(pred_lbl)
        return preds

    def save(self, filepath: str or Path) -> Path:
        """Save classifier state."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({
                "threshold": self.threshold,
                "train_weights": self.train_weights,
                "train_labels": self.train_labels
            }, f)
        return p

    def load(self, filepath: str or Path) -> "EuclideanClassifier":
        """Load classifier state."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.threshold = data["threshold"]
        self.train_weights = data["train_weights"]
        self.train_labels = data["train_labels"]
        self.is_trained = True
        return self

