"""PCA + Support Vector Machine (SVM) Classifier with Calibrated Probabilistic Unknown Rejection."""
import pickle
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.logger import setup_logger

logger = setup_logger("SVMClassifier")


class SVMClassifier:
    """Support Vector Machine classifier operating on PCA feature weights."""

    UNKNOWN_LABEL = "UNKNOWN"

    def __init__(self, kernel: str = "rbf", C: float = 10.0, gamma: str = "scale",
                 probability: bool = True, confidence_threshold: Optional[float] = 0.40,
                 scale_features: bool = False):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability
        self.confidence_threshold = confidence_threshold
        self.scale_features = scale_features
        self.model: Optional[SVC | Pipeline] = None
        self.classes_: Optional[np.ndarray] = None
        self.is_trained = False

    def fit(self, X_weights: np.ndarray, y_labels: np.ndarray,
            tune_hyperparameters: bool = False) -> "SVMClassifier":
        """Train SVM classifier on PCA projection weights."""
        X = np.asarray(X_weights, dtype=np.float32)
        y = np.asarray(y_labels)
        estimator = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma,
                        probability=self.probability, random_state=42)
        if self.scale_features:
            estimator = make_pipeline(StandardScaler(), estimator)

        if tune_hyperparameters and len(X) >= 30:
            logger.info("Tuning SVM hyperparameters via Stratified 3-Fold Cross-Validation...")
            param_grid = {
                "C": [1.0, 10.0, 50.0],
                "gamma": ["scale", "auto", 0.01, 0.001],
                "kernel": ["rbf", "linear"]
            }
            if self.scale_features:
                param_grid = {f"svc__{key}": values for key, values in param_grid.items()}
            # Stratified K-Fold
            cv = StratifiedKFold(n_splits=min(3, np.min(np.bincount(np.unique(y, return_inverse=True)[1]))))
            grid = GridSearchCV(estimator, param_grid, cv=cv, n_jobs=-1)
            grid.fit(X, y)
            self.model = grid.best_estimator_
            logger.info(f"Best SVM parameters: {grid.best_params_}")
        else:
            self.model = estimator.fit(X, y)

        self.classes_ = self.model.classes_
        self.is_trained = True
        logger.info(f"Fitted SVM with {len(X)} samples across {len(self.classes_)} classes.")
        return self

    def calibrate_threshold(self, val_weights: np.ndarray, val_labels: np.ndarray,
                            target_far: float = 0.05) -> float:
        """Calibrate confidence threshold using validation set class probabilities."""
        if not self.is_trained or not self.probability:
            return self.confidence_threshold or 0.40

        probs = self.model.predict_proba(val_weights)
        max_probs = np.max(probs, axis=1)
        preds = self.classes_[np.argmax(probs, axis=1)]

        correct_probs = [max_probs[i] for i, (p, t) in enumerate(zip(preds, val_labels)) if p == t]
        if correct_probs:
            # Set threshold to capture ~90% of correct classifications
            calibrated = float(np.percentile(correct_probs, 10))
            # Keep within sensible bounds [0.25, 0.65]
            self.confidence_threshold = max(0.25, min(0.65, calibrated))
        else:
            self.confidence_threshold = 0.40

        logger.info(f"Calibrated SVM confidence threshold: {self.confidence_threshold:.3f}")
        return self.confidence_threshold

    def predict_single(self, weight_vector: np.ndarray) -> Tuple[str, float]:
        """Predict student identity and class confidence score for a single projection."""
        if not self.is_trained:
            raise ValueError("Classifier is not trained.")

        X = np.atleast_2d(weight_vector)
        if self.probability:
            prob_dist = self.model.predict_proba(X)[0]
            max_idx = int(np.argmax(prob_dist))
            confidence = float(prob_dist[max_idx])
            pred_label = str(self.classes_[max_idx])

            if self.confidence_threshold is not None and confidence < self.confidence_threshold:
                return self.UNKNOWN_LABEL, confidence
            return pred_label, confidence
        else:
            pred = str(self.model.predict(X)[0])
            return pred, 1.0

    def predict(self, X_weights: np.ndarray) -> List[str]:
        """Predict labels for batch of feature weights."""
        return [self.predict_single(w)[0] for w in X_weights]

    def save(self, filepath: str or Path) -> Path:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({
                "model": self.model,
                "classes_": self.classes_,
                "confidence_threshold": self.confidence_threshold
            }, f)
        return p

    def load(self, filepath: str or Path) -> "SVMClassifier":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.classes_ = data["classes_"]
        self.confidence_threshold = data["confidence_threshold"]
        self.is_trained = True
        return self

