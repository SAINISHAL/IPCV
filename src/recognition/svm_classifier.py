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

    def predict_single(self, weight_vector: np.ndarray, *, apply_rejection: bool = True) -> Tuple[str, float]:
        """Predict student identity and class confidence score for a single projection."""
        return self.predict_batch(np.atleast_2d(weight_vector), apply_rejection=apply_rejection)[0]

    def predict_batch(self, weights: np.ndarray, *, apply_rejection: bool = True) -> List[Tuple[str, float]]:
        if not self.is_trained:
            raise ValueError("Classifier is not trained.")

        if not self.probability:
            return [(str(label), 1.0) for label in self.model.predict(weights)]
        probs = self.model.predict_proba(weights)
        best = probs.argmax(axis=1)
        cutoff = self.confidence_threshold if apply_rejection else None
        return [(self.UNKNOWN_LABEL if cutoff is not None and row[i] < cutoff else str(self.classes_[i]),
                 float(row[i])) for row, i in zip(probs, best)]

    def prediction_details(self, weights: np.ndarray) -> Dict[str, Any]:
        """Report calibrated rankings and the separate raw SVM decision scores."""
        sample = np.atleast_2d(weights)
        probs = self.model.predict_proba(sample)[0] if self.probability else None
        top_two = [(str(self.classes_[i]), float(probs[i])) for i in np.argsort(probs)[-2:][::-1]] if probs is not None else []
        estimator = self.model
        if hasattr(estimator, "calibrated_classifiers_"):
            estimator = estimator.calibrated_classifiers_[0].estimator
        raw = estimator.decision_function(sample).ravel() if hasattr(estimator, "decision_function") else np.empty(0)
        if len(raw) == 1 and len(self.classes_) == 2:
            raw = np.array([-raw[0], raw[0]])
        return {"top_two": top_two, "svm_decision_scores": dict(zip(map(str, self.classes_), map(float, raw))),
                "cutoff": self.confidence_threshold}

    def predict(self, X_weights: np.ndarray) -> List[str]:
        """Predict labels for batch of feature weights."""
        return [label for label, _ in self.predict_batch(X_weights)] if len(X_weights) else []

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

