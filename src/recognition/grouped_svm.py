"""Calibrate an SVM using pairwise margins from separate source-photo groups."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


class GroupedMarginSVM:
    """Combine held-out pairwise SVM margins into one probability per identity."""

    def __init__(self, estimator, folds):
        self.estimator = estimator
        self.folds = folds

    def fit(self, weights, labels):
        margins = cross_val_predict(self.estimator, weights, labels, cv=self.folds, method="decision_function")
        self.calibrator = LogisticRegression(C=1.0, max_iter=1000).fit(self._as_matrix(margins), labels)
        self.estimator.fit(weights, labels)
        self.classes_ = self.calibrator.classes_
        self.n_features_in_ = weights.shape[1]
        return self

    @staticmethod
    def _as_matrix(margins):
        return margins.reshape(-1, 1) if margins.ndim == 1 else margins

    def predict_proba(self, weights):
        margins = self.estimator.decision_function(weights)
        return self.calibrator.predict_proba(self._as_matrix(margins))

    def decision_function(self, weights):
        """Return a raw sum of signed pairwise margins per class for diagnostics."""
        margins = self.estimator.decision_function(weights)
        if margins.ndim == 1:
            return np.column_stack((-margins, margins))
        scores = np.zeros((len(margins), len(self.classes_)))
        column = 0
        for left in range(len(self.classes_)):
            for right in range(left + 1, len(self.classes_)):
                scores[:, left] += margins[:, column]
                scores[:, right] -= margins[:, column]
                column += 1
        return scores
