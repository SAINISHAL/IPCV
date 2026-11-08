"""Train a more varied Eigenfaces/SVM candidate and test lower rejection cutoffs."""
import argparse
from collections import Counter
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.photo_groups import fingerprint, near_duplicate
from src.detection.unified_detector import UnifiedFaceDetector
from src.evaluation.recognition_study import (
    SVMParams, count_predictions, create_svm, model_classifier, performance, predict_scores, prepare_fit, project,
    rank, with_cutoff,
)
from src.evaluation.recognition_study_data import FaceSample, load_camera_sessions, load_classroom, select_study_rows
from src.recognition.euclidean_classifier import EuclideanClassifier
from src.recognition.grouped_svm import GroupedMarginSVM
from src.recognition.knn_classifier import KNNClassifier
from src.recognition.prediction import RecognitionPipeline


LOWER_CUTOFFS = (0.60, 0.575, 0.55, 0.525, 0.50, 0.45, 0.40, 0.35)


def within_validation_limits(counts, baseline):
    return counts.get("wrong", 0) <= baseline["wrong"] and counts.get("unassigned_named", 0) <= baseline["unassigned_named"]


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    detector = UnifiedFaceDetector(engine="yunet")
    parts, flags, _, fingerprints = load_classroom(ROOT, detector)
    parts = select_study_rows(parts, flags)
    train = [row for row in parts["train"] if row.name != "UNKNOWN"]
    validation, test = parts["validation"], parts["test"]
    roster = sorted({row.name for row in train})
    assert all(not ({r.group for r in a} & {r.group for r in b}) for a, b in
               ((train, validation), (train, test), (validation, test)))

    occupied = list(fingerprints.values())
    fresh_train = load_camera_sessions(ROOT, detector, "train", occupied, set(roster))
    enrollment = list(fresh_train)
    for path in sorted((ROOT / "data/enrollment").glob("*/*.png")):
        image = cv2.imread(str(path))
        if path.parent.name in roster and image is not None and image.shape[:2] == (112, 112):
            enrollment.append(FaceSample(path.parent.name, str(path), path.name, image, None, str(path.parent)))
    originals = train + enrollment
    print(f"Training {len(originals)} eligible original crops with 6 variations each.", flush=True)

    incumbent = RecognitionPipeline.load_pipeline(ROOT / "models/eigenfaces/pca_model.pkl",
                                                 ROOT / "models/classifiers/svm_classifier.pkl")
    saved_cutoff = incumbent.classifier.confidence_threshold
    legacy_trials = []
    valid_names = [row.name for row in validation]

    def incumbent_predict(rows, cutoff):
        incumbent.classifier.confidence_threshold = cutoff
        results = incumbent.recognize_faces([r.crop(incumbent.eigenfaces_model.face_alignment) for r in rows],
                                            retry_webcam=True)
        return [r["identity"] for r in results]

    saved_validation = count_predictions(valid_names, incumbent_predict(validation, saved_cutoff))
    for cutoff in LOWER_CUTOFFS:
        if cutoff < saved_cutoff:
            stats = count_predictions(valid_names, incumbent_predict(validation, cutoff))
            legacy_trials.append((cutoff, stats))
    eligible_legacy = [(cutoff, stats) for cutoff, stats in legacy_trials
                       if within_validation_limits(stats, saved_validation)]
    # Compare a modest decrease even if no legacy cutoff meets the validation limits. It will not be promoted.
    lower_cutoff, lower_validation = max(eligible_legacy or legacy_trials,
                                         key=lambda row: rank(row[1], row[0]), default=(saved_cutoff, saved_validation))
    print(f"Saved validation: {dict(saved_validation)}. Lower saved-model cutoff {lower_cutoff}: {dict(lower_validation)}", flush=True)

    fit = prepare_fit(originals, "E", augmented=True, extensive=True)
    X_val = project(fit, validation, 120)
    trials, candidate = [], None
    for k, c, gamma in product((40, 60, 80, 120), (1, 10), ("scale", 0.01)):
        params = SVMParams(k, "rbf", c, gamma)
        estimator = create_svm(params).set_params(svc__decision_function_shape="ovo")
        model = GroupedMarginSVM(estimator, fit.folds).fit(fit.weights[:, :k], fit.labels)
        names, scores = predict_scores(model, X_val[:, :k])
        for cutoff in (*LOWER_CUTOFFS, saved_cutoff):
            stats = count_predictions(valid_names, with_cutoff(names, scores, cutoff))
            allowed = within_validation_limits(stats, saved_validation)
            trials.append({**asdict(params), "cutoff": cutoff, "within_limits": allowed, **dict(stats)})
            key = (allowed, *rank(stats, cutoff))
            if candidate is None or key > candidate[0]:
                candidate = (key, params, model, cutoff, stats)
        print(f"Validated K={k}, C={c}, gamma={gamma}.", flush=True)

    _, params, model, cutoff, stats = candidate
    print(f"Selected on validation: {params}, cutoff {cutoff}, {dict(stats)}", flush=True)
    # Do not use the separate webcam test or classroom test to choose thresholds or hyperparameters.
    fresh_test = load_camera_sessions(ROOT, detector, "test", occupied, set(roster))
    unknown_test = load_camera_sessions(ROOT, detector, "unknown", occupied, set(roster))
    camera_fingerprints = [fingerprint(r.aligned) for r in enrollment]
    fresh_test = [r for r in fresh_test if not any(near_duplicate(fingerprint(r.aligned), f, max_difference=3)
                                                   for f in camera_fingerprints)]

    def candidate_predict(rows):
        return with_cutoff(*predict_scores(model, project(fit, rows, params.k)), cutoff)

    variants = {
        "saved": (saved_cutoff, saved_validation, lambda rows: incumbent_predict(rows, saved_cutoff)),
        "lower_cutoff": (lower_cutoff, lower_validation, lambda rows: incumbent_predict(rows, lower_cutoff)),
        "augmented": (cutoff, stats, candidate_predict),
    }
    results = {}
    for name, (threshold, validation_counts, predict) in variants.items():
        metrics = performance(test, predict(test), roster)
        webcam = performance(fresh_test, predict(fresh_test), roster) if fresh_test else None
        false_accept = sum(identity != "UNKNOWN" for identity in predict(unknown_test)) if unknown_test else None
        results[name] = {"cutoff": threshold, "validation": dict(validation_counts), "test": metrics,
                         "webcam": webcam, "verified_false_accepts": false_accept}
        print(f"{name}: {metrics['correct']}/{metrics['total']} correct, {metrics['wrong']} wrong, "
              f"{metrics['unknown']} UNKNOWN, {metrics['unassigned_named']}/{metrics['unassigned_total']} unassigned named", flush=True)

    saved_test = results["saved"]["test"]
    eligible = []
    for name in ("lower_cutoff", "augmented"):
        item = results[name]
        held_out = item["test"]
        if (item["cutoff"] < saved_cutoff and within_validation_limits(item["validation"], saved_validation)
                and held_out["correct"] > saved_test["correct"] and held_out["wrong"] <= saved_test["wrong"]
                and held_out["unassigned_named"] <= saved_test["unassigned_named"]
                and (not unknown_test or item["verified_false_accepts"] <= results["saved"]["verified_false_accepts"])):
            eligible.append(name)
    recommendation = max(eligible, key=lambda name: rank(Counter(results[name]["validation"]), results[name]["cutoff"]),
                         default="saved")

    stage = out / "candidate"
    fit.pca.n_components = params.k
    fit.pca.save(stage / "eigenfaces/pca_model.pkl")
    model_classifier(model, cutoff).save(stage / "classifiers/svm_classifier.pkl")
    known = [r for r in validation if r.name != "UNKNOWN"]
    known_weights, known_names = project(fit, known, params.k), np.array([r.name for r in known])
    for kind, classifier in (("euclidean", EuclideanClassifier()), ("knn", KNNClassifier())):
        classifier.fit(fit.weights[:, :params.k], fit.labels)
        classifier.calibrate_threshold(known_weights, known_names)
        classifier.save(stage / f"classifiers/{kind}_classifier.pkl")

    payload = {"recommendation": recommendation, "training_original_crops": len(originals),
               "training_examples": len(fit.labels), "fresh_training_captures": len(fresh_train),
               "verified_unenrolled_test": len(unknown_test), "fresh_webcam_test": len(fresh_test),
               "parameters": asdict(params), "legacy_cutoffs": [(c, dict(v)) for c, v in legacy_trials],
               "candidate_trials": trials, "results": results}
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out, payload)


def write_report(out, payload):
    lines = ["# Augmented webcam recognition study", "",
             f"{payload['training_original_crops']} original crops produced {payload['training_examples']} total training examples. "
             "Each original contributed a mirror, exposure or contrast variation, gamma change, small rotation, mild noise, and shadow. "
             "Generated examples are not independent photos. Photo groups stayed together in training, validation, testing, and SVM calibration.", "",
             f"New confirmed webcam training photos: {payload['fresh_training_captures']}. "
             f"Fresh separate webcam test photos: {payload['fresh_webcam_test']}.", "",
             "The augmented model uses five-point alignment, CLAHE, Eigenfaces, and an RBF SVM. Probabilities were calibrated "
             "from pairwise SVM margins scored in held-out training-photo groups. Cutoffs were selected using the validation photos. "
             "The locked classroom holdout is the same regression set used in the earlier study.", "",
             "| Model | Cutoff | Correct | Wrong name | UNKNOWN | Unassigned named |", "|---|---:|---:|---:|---:|---:|"]
    for name, result in payload["results"].items():
        m = result["test"]
        lines.append(f"| {name.replace('_', ' ')} | {result['cutoff']:.3f} | {m['correct']}/{m['total']} | "
                     f"{m['wrong']} | {m['unknown']} | {m['unassigned_named']}/{m['unassigned_total']} |")
    lines += ["", f"Recommended configuration: **{payload['recommendation'].replace('_', ' ')}**. "
              "Promotion requires more correct test matches, with no additional wrong names or unassigned naming on validation and test. "
              "The training script stages model artifacts and does not replace the active model files.", ""]
    if not payload["verified_unenrolled_test"]:
        lines.append("A true false acceptance rate remains unmeasured. Unassigned classroom faces are only a proxy.")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/recognition_augmentation")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    with threadpool_limits(limits=2):
        run(args.output)
