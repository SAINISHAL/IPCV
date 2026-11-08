"""Group-calibrated Eigenfaces/SVM experiments and open-set evaluation."""
from collections import Counter
from dataclasses import dataclass
from itertools import product

import cv2
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.eigenfaces.eigenfaces_model import EigenfacesModel
from src.preprocessing.image_preprocessing import ImagePreprocessor
from src.recognition.svm_classifier import SVMClassifier


PREPROCESSING = {
    "A": ("box", "gray", "Grayscale + resize"),
    "B": ("box", "clahe", "Grayscale + CLAHE"),
    "C": ("box", "gamma", "Gamma + grayscale"),
    "D": ("box", "clahe_gamma", "CLAHE + gamma"),
    "E": ("sface", "clahe", "Five-point alignment + grayscale + CLAHE"),
}
CUTOFFS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.625, 0.65, 0.70, 0.75, 0.80)


@dataclass(frozen=True)
class SVMParams:
    k: int = 80
    kernel: str = "rbf"
    c: float = 10
    gamma: str | float = "scale"
    balancing: str | None = None


@dataclass
class FitData:
    pca: EigenfacesModel
    processor: ImagePreprocessor
    weights: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    folds: list


def augment(crop, index, rng):
    """Use one mild variation per original in addition to a mirror."""
    operation = index % 6
    if operation == 0:  # brightness
        return np.clip(crop.astype(np.int16) + rng.choice([-22, 22]), 0, 255).astype(np.uint8)
    if operation == 1:  # contrast
        return np.clip((crop.astype(np.float32) - 127.5) * rng.choice([0.82, 1.17]) + 127.5,
                       0, 255).astype(np.uint8)
    if operation == 2:  # gamma
        lut = np.uint8(np.round(255 * (np.arange(256) / 255) ** rng.choice([0.80, 1.22])))
        return cv2.LUT(crop, lut)
    if operation == 3:  # rotation
        h, w = crop.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), rng.choice([-6, 6]), 1)
        return cv2.warpAffine(crop, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    if operation == 4:  # sensor noise
        return np.clip(crop.astype(np.float32) + rng.normal(0, 4, (*crop.shape[:2], 1)), 0, 255).astype(np.uint8)
    # A soft shadow over one side of the face.
    light = np.linspace(0.70, 1, crop.shape[1], dtype=np.float32)
    if rng.integers(2):
        light = light[::-1]
    return np.clip(crop * light[None, :, None], 0, 255).astype(np.uint8)


def prepare_fit(rows, method, *, augmented, extensive=False):
    alignment, mode, _ = PREPROCESSING[method]
    processor = ImagePreprocessor(contrast_mode=mode)
    pca = EigenfacesModel(n_components=120, face_alignment=alignment, preprocessing_mode=mode, webcam_retry=False)
    rng = np.random.default_rng(42)
    vectors, labels, groups = [], [], []
    for i, row in enumerate(rows):
        crop = row.crop(alignment)
        images = [crop]
        if augmented:
            images.extend((cv2.flip(crop, 1), augment(crop, i, rng)))
        if extensive:
            # A bounded set gives every original rotation, exposure, gamma, noise, and shadow coverage.
            images = [crop, cv2.flip(crop, 1)]
            images.extend(augment(crop, operation, rng) for operation in (i % 2, 2, 3, 4, 5))
        for image in images:
            vectors.append(processor.process_and_flatten(image))
            labels.append(row.name)
            groups.append(row.group)
    pca.train(np.stack(vectors))
    labels, groups = np.asarray(labels), np.asarray(groups)
    enrolled = set(labels)
    n_folds = min(3, min(len(set(groups[labels == name])) for name in enrolled))
    if n_folds < 2:
        raise ValueError("Each identity needs at least two independent training photo groups for calibration.")
    for seed in range(42, 62):
        folds = list(StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed).split(vectors, labels, groups))
        if all(set(labels[train]) == enrolled for train, _ in folds):
            break
    else:
        raise ValueError("Some identities have too few independent photo groups for grouped calibration.")
    assert all(not (set(groups[train]) & set(groups[cal])) for train, cal in folds)
    weights = pca.pca.project(np.stack(vectors), k=120)
    return FitData(pca, processor, weights, labels, groups, folds)


def project(fit, rows, k):
    if not rows:
        return np.empty((0, k))
    X = np.stack([fit.processor.process_and_flatten(row.crop(fit.pca.face_alignment)) for row in rows])
    return fit.pca.pca.project(X, k=k)


def create_svm(params):
    return make_pipeline(StandardScaler(), SVC(kernel=params.kernel, C=params.c, gamma=params.gamma,
                                                class_weight=params.balancing, decision_function_shape="ovr"))


def calibrate(fit, params):
    model = CalibratedClassifierCV(create_svm(params), method="sigmoid", cv=fit.folds, ensemble=False)
    model.fit(fit.weights[:, :params.k], fit.labels)
    return model


def count_predictions(expected, predictions):
    counts = Counter()
    for truth, name in zip(expected, predictions, strict=True):
        if truth == "UNKNOWN":
            counts["unassigned_named"] += int(name != "UNKNOWN")
            counts["unassigned_total"] += 1
        else:
            counts["correct" if name == truth else "unknown" if name == "UNKNOWN" else "wrong"] += 1
    return counts


def predict_scores(model, weights):
    if not len(weights):
        return np.array([], dtype=str), np.array([], dtype=float)
    probabilities = model.predict_proba(weights)
    return model.classes_[probabilities.argmax(axis=1)], probabilities.max(axis=1)


def with_cutoff(best_names, confidence, cutoff):
    return np.where(confidence >= cutoff, best_names, "UNKNOWN")


def evaluate_cutoffs(model, val_weights, expected, incumbent):
    names, scores = predict_scores(model, val_weights)
    counts = {cutoff: count_predictions(expected, with_cutoff(names, scores, cutoff)) for cutoff in CUTOFFS}
    allowed = {c: v for c, v in counts.items()
               if v["wrong"] <= incumbent["wrong"] and v["unassigned_named"] <= incumbent["unassigned_named"]}
    if not allowed:
        allowed = {max(counts): counts[max(counts)]}
    cutoff = max(allowed, key=lambda c: rank(allowed[c], c))
    return cutoff, counts


def rank(counts, cutoff):
    correct, wrong, unassigned = counts["correct"], counts["wrong"], counts["unassigned_named"]
    return correct - 2 * wrong - unassigned, -wrong, -unassigned, correct, cutoff


def search(fit, validation, incumbent):
    """Screen the whole grid, then calibrate competitive candidates for rejection."""
    X = project(fit, validation, 120)
    y = np.array([row.name for row in validation])
    known = y != "UNKNOWN"
    records = []
    for k, c, balance in product((20, 40, 60, 80, 100, 120), (0.1, 1, 10, 100), (None, "balanced")):
        if k > fit.weights.shape[1]:
            continue
        for kernel, gamma in [("linear", "scale"), ("rbf", "scale"), ("rbf", 0.01), ("rbf", 0.05)]:
            params = SVMParams(k, kernel, c, gamma, balance)
            estimator = create_svm(params).fit(fit.weights[:, :k], fit.labels)
            prediction = estimator.predict(X[known, :k])
            records.append((params, int(np.count_nonzero(prediction == y[known]))))
    ordered = sorted(records, key=lambda pair: -pair[1])
    finalists = {params for params, _ in ordered[:16]}
    finalists.add(SVMParams())
    for key in ("kernel", "k", "balancing"):
        seen = set()
        for params, _ in ordered:
            value = getattr(params, key)
            if value not in seen:
                finalists.add(params)
                seen.add(value)
    scored = []
    for params, _ in ordered:
        if params not in finalists:
            continue
        model = calibrate(fit, params)
        cutoff, counts = evaluate_cutoffs(model, X[:, :params.k], y, incumbent)
        scored.append((params, model, cutoff, counts))
    return max(scored, key=lambda row: rank(row[3][row[2]], row[2])), records, scored


def model_classifier(model, cutoff):
    classifier = SVMClassifier(confidence_threshold=cutoff)
    classifier.model = model
    classifier.classes_ = model.classes_
    classifier.is_trained = True
    return classifier


def performance(rows, predictions, roster):
    truth = np.array([row.name for row in rows])
    predictions = np.array(predictions)
    known = truth != "UNKNOWN"
    counts = count_predictions(truth, predictions)
    total = int(known.sum())
    return {
        "total": total,
        "correct": counts["correct"], "wrong": counts["wrong"], "unknown": counts["unknown"],
        "accuracy": counts["correct"] / total if total else None,
        "macro_f1": float(f1_score(truth[known], predictions[known], labels=sorted(set(truth[known])),
                                   average="macro", zero_division=0)) if total else None,
        "coverage": (counts["correct"] + counts["wrong"]) / total if total else None,
        "unknown_fraction": counts["unknown"] / total if total else None,
        "unassigned_named": counts["unassigned_named"], "unassigned_total": counts["unassigned_total"],
        "recall": {name: {"correct": int(np.sum((truth == name) & (predictions == name))),
                          "total": int(np.sum(truth == name))} for name in roster},
        "matrix": confusion_matrix(truth[known], predictions[known], labels=[*roster, "UNKNOWN"]).tolist(),
    }
