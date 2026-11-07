"""Unit tests for recognition classifiers and unknown rejection."""
import pytest
import numpy as np
from src.recognition.euclidean_classifier import EuclideanClassifier
from src.recognition.knn_classifier import KNNClassifier
from src.recognition.svm_classifier import SVMClassifier
from src.recognition.unknown_detector import UnknownDetector


def test_euclidean_classifier():
    X = np.array([
        [1.0, 0.0],
        [1.1, 0.1],
        [-1.0, 0.0],
        [-0.9, -0.1]
    ])
    y = np.array(["StudentA", "StudentA", "StudentB", "StudentB"])

    clf = EuclideanClassifier(threshold=1.0)
    clf.fit(X, y)

    # In-distribution test
    pred, dist = clf.predict_single(np.array([1.05, 0.05]))
    assert pred == "StudentA"
    assert dist < 1.0

    # Far out-of-distribution (unknown)
    pred_unk, dist_unk = clf.predict_single(np.array([50.0, 50.0]))
    assert pred_unk == "UNKNOWN"
    assert dist_unk > 1.0


def test_knn_classifier():
    X = np.array([
        [1.0, 1.0], [1.1, 0.9], [0.9, 1.1],
        [-1.0, -1.0], [-1.1, -0.9], [-0.9, -1.1]
    ])
    y = np.array(["A", "A", "A", "B", "B", "B"])

    knn = KNNClassifier(n_neighbors=3, threshold=2.0)
    knn.fit(X, y)

    pred, dist = knn.predict_single(np.array([1.0, 0.95]))
    assert pred == "A"

    pred_unk, dist_unk = knn.predict_single(np.array([100.0, 100.0]))
    assert pred_unk == "UNKNOWN"


def test_svm_classifier():
    X = np.array([
        [1.0, 1.0], [1.2, 0.8], [0.8, 1.2], [1.1, 1.1],
        [-1.0, -1.0], [-1.2, -0.8], [-0.8, -1.2], [-1.1, -1.1]
    ])
    y = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

    svm = SVMClassifier(kernel="rbf", C=1.0, confidence_threshold=0.5)
    svm.fit(X, y)

    pred, conf = svm.predict_single(np.array([1.0, 1.0]))
    assert pred == "A"
    assert conf >= 0.5


def test_unknown_detector_metrics():
    y_true = ["A", "A", "B", "UNKNOWN", "UNKNOWN"]
    y_pred = ["A", "UNKNOWN", "B", "UNKNOWN", "A"]  # 1 FR, 1 FA
    enrolled = {"A", "B"}

    metrics = UnknownDetector.evaluate_unknown_rejection(y_true, y_pred, enrolled)
    # Known: 3 (2 accepted, 1 rejected -> FRR = 1/3)
    # Unknown: 2 (1 rejected, 1 accepted -> FAR = 1/2)
    assert abs(metrics["FRR"] - 1/3) < 1e-4
    assert abs(metrics["FAR"] - 1/2) < 1e-4

