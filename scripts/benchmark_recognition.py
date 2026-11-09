"""Compare five Eigenfaces/SVM experiments without touching the active model files."""
import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.photo_groups import fingerprint, near_duplicate
from src.detection.unified_detector import UnifiedFaceDetector
from src.evaluation.recognition_study import (
    PREPROCESSING, SVMParams, calibrate, count_predictions, evaluate_cutoffs,
    model_classifier, performance, predict_scores, prepare_fit, project, rank, search, with_cutoff,
)
from src.evaluation.recognition_study_data import (
    FaceSample, audit_identities, load_camera_sessions, load_classroom, select_study_rows,
)
from src.recognition.euclidean_classifier import EuclideanClassifier
from src.recognition.knn_classifier import KNNClassifier
from src.recognition.prediction import RecognitionPipeline
from src.webcam.video_processor import VideoProcessor


def log(message):
    print(message, flush=True)


def csv_file(path, headers, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def select_fit(train, validation, method, augmented, incumbent, params=SVMParams()):
    fit = prepare_fit(train, method, augmented=augmented)
    model = calibrate(fit, params)
    cutoff, sweep = evaluate_cutoffs(model, project(fit, validation, params.k), [row.name for row in validation], incumbent)
    return fit, params, model, cutoff, sweep


def build_pipeline(experiment):
    fit, params, model, cutoff, _ = experiment
    fit.pca.n_components = params.k
    pipeline = RecognitionPipeline(fit.processor, fit.pca, k_components=params.k)
    pipeline.set_classifier(model_classifier(model, cutoff), "svm")
    return pipeline


def time_replay(detector, pipeline, source):
    original = cv2.imread(str(ROOT / source))
    h, w = original.shape[:2]
    scale = min(1280 / w, 720 / h)
    frame = cv2.resize(original, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    processor = VideoProcessor(detector, pipeline)
    processor.process_frame(frame)
    start = perf_counter()
    for dx in (-2, -1, 0, 1, 2, 0):
        moved = cv2.warpAffine(frame, np.float32([[1, 0, dx], [0, 1, 0]]), frame.shape[1::-1])
        annotated, _ = processor.process_frame(moved)
        cv2.imencode(".jpg", annotated)
    return 6 / (perf_counter() - start)


def study(out):
    out.mkdir(parents=True, exist_ok=True)
    detector = UnifiedFaceDetector(engine="yunet")
    parts, flags, groups, fingerprints = load_classroom(ROOT, detector)
    original_rows = [row for rows in parts.values() for row in rows]
    audit_model = cv2.FaceRecognizerSF.create(str(ROOT / "models/pretrained/sface.onnx"), "")
    audit_identities(original_rows, audit_model, flags)
    log(f"Audit: {len(groups)} source photos in {len(set(groups.values()))} independent groups; "
        f"flags {dict((key, len(items)) for key, items in flags.items())}")
    parts = select_study_rows(parts, flags)

    train = [row for row in parts["train"] if row.name != "UNKNOWN"]
    validation = parts["validation"]
    roster = sorted({row.name for row in original_rows if row.name != "UNKNOWN"})
    if set(roster) != {row.name for row in train}:
        raise ValueError("A named class is absent from training after duplicate and quality checks.")
    split_groups = {key: {row.group for row in rows} for key, rows in parts.items()}
    assert not any(split_groups[a] & split_groups[b] for a, b in
                   [("train", "validation"), ("train", "test"), ("validation", "test")])
    log(f"Locked photo groups: { {split: len(values) for split, values in split_groups.items()} }")
    (out / "dataset_audit.json").write_text(json.dumps({"flags": flags, "photo_groups": groups,
                                                       "split_groups": {k: sorted(v) for k, v in split_groups.items()}},
                                                      indent=2), encoding="utf-8")

    pca_path = ROOT / "models/eigenfaces/pca_model.pkl"
    svm_path = ROOT / "models/classifiers/svm_classifier.pkl"
    baseline = RecognitionPipeline.load_pipeline(pca_path, svm_path)
    assert set(baseline.classifier.classes_) == set(roster), "Saved SVM labels disagree with the identity mapping."
    log(f"Existing PCA K={baseline.k_components}, SVM classes={len(roster)}, cutoff={baseline.classifier.confidence_threshold}")
    incumbent = count_predictions(
        [row.name for row in validation],
        [result["identity"] for result in baseline.recognize_faces([row.aligned for row in validation], retry_webcam=True)],
    )
    log(f"Saved model validation: {dict(incumbent)}")

    prep_rows = []
    selected = None
    for method, (_, _, title) in PREPROCESSING.items():
        log(f"Preprocessing {method}: {title}")
        exp = select_fit(train, validation, method, augmented=False, incumbent=incumbent)
        _, _, _, cutoff, sweep = exp
        counts = sweep[cutoff]
        prep_rows.append((method, title, cutoff, counts))
        if selected is None or rank(counts, cutoff) > selected[0]:
            selected = (rank(counts, cutoff), method, exp)
    _, method, e2 = selected
    log(f"Validation selected preprocessing {method}; evaluating bounded augmentation.")
    e3 = select_fit(train, validation, method, augmented=True, incumbent=incumbent)
    alternatives, grid, finalists = [], [], []
    for key in dict.fromkeys((method, "E")):
        log(f"Screening the PCA and SVM grid for preprocessing {key}.")
        fit = e3[0] if key == method else prepare_fit(train, key, augmented=True)
        (params, model, cutoff, sweep), screened, calibrated = search(fit, validation, incumbent)
        alternatives.append((key, (fit, params, model, cutoff, sweep)))
        grid.extend((key, p, correct) for p, correct in screened)
        finalists.extend((key, p, c, sweep) for p, _, c, sweep in calibrated)
    tuned_method, e4 = max(alternatives, key=lambda entry: rank(entry[1][4][entry[1][3]], entry[1][3]))
    params = e4[1]
    csv_file(out / "svm_grid.csv", ["preprocessing", "pca_components", "kernel", "C", "gamma", "class_balance", "validation_top1_correct"],
             [(key, *asdict(p).values(), correct) for key, p, correct in grid])
    csv_file(out / "validation_cutoffs.csv", ["preprocessing", "pca_components", "kernel", "C", "gamma", "class_balance",
                                             "cutoff", "selected_for_configuration", "correct", "wrong", "unknown", "unassigned_named"],
             [(key, *asdict(p).values(), c, c == chosen, counts["correct"], counts["wrong"], counts["unknown"],
               counts["unassigned_named"]) for key, p, chosen, sweep in finalists for c, counts in sweep.items()])

    occupied = list(fingerprints.values())
    new_train = load_camera_sessions(ROOT, detector, "train", occupied, set(roster))
    camera_train = list(new_train)
    if PREPROCESSING[tuned_method][0] == "sface":
        for path in sorted((ROOT / "data/enrollment").glob("*/*.png")):
            if path.parent.name in roster:
                crop = cv2.imread(str(path))
                if crop is not None and crop.shape[:2] == (112, 112):
                    camera_train.append(FaceSample(path.parent.name, str(path), path.name, crop, None, str(path.parent)))
    e5 = select_fit(train + camera_train, validation, tuned_method, augmented=True,
                    incumbent=incumbent, params=params) if camera_train else None

    # Pick a configuration before reading any final-test capture sessions or evaluating test predictions.
    candidates = [(2, e2), (3, e3), (4, e4)] + ([(5, e5)] if e5 else [])
    chosen_num, chosen = max(candidates, key=lambda item: rank(item[1][4][item[1][3]], item[1][3]))
    log(f"Validation selected E{chosen_num}; K={chosen[1].k}, kernel={chosen[1].kernel}, cutoff={chosen[3]}")
    webcam_test = load_camera_sessions(ROOT, detector, "test", occupied, set(roster))
    webcam_unknown = load_camera_sessions(ROOT, detector, "unknown", occupied, set(roster))
    prior_camera_crops = [fingerprint(row.aligned) for row in camera_train]
    webcam_test = [row for row in webcam_test if not any(near_duplicate(fingerprint(row.aligned), prior, max_difference=3)
                                                       for prior in prior_camera_crops)]

    test = parts["test"]
    source = next(row.source for row in test if row.name != "UNKNOWN")
    results = {}
    for number, exp in [(1, None), *candidates]:
        pipeline = baseline if exp is None else build_pipeline(exp)
        def predict_rows(rows):
            crops = [row.crop(pipeline.eigenfaces_model.face_alignment) for row in rows]
            return [r["identity"] for r in pipeline.recognize_faces(crops, retry_webcam=True)]
        metrics = performance(test, predict_rows(test), roster)
        camera_metrics = performance(webcam_test, predict_rows(webcam_test), roster) if webcam_test else None
        false_acceptances = sum(name != "UNKNOWN" for name in predict_rows(webcam_unknown))
        results[number] = {"classroom": metrics, "webcam_test": camera_metrics,
                           "false_acceptances": false_acceptances if webcam_unknown else None,
                           "verified_unenrolled": len(webcam_unknown),
                           "replay_fps": time_replay(detector, pipeline, source)}
        csv_file(out / f"confusion_e{number}.csv", ["True / Predicted", *roster, "UNKNOWN"],
                 [[label, *row] for label, row in zip([*roster, "UNKNOWN"], metrics["matrix"], strict=True)])
        log(f"E{number} test: {metrics['correct']}/{metrics['total']} correct, {metrics['wrong']} wrong, "
            f"{metrics['unknown']} UNKNOWN; replay {results[number]['replay_fps']:.1f} FPS")

    candidate = out / "candidate"
    fit, params, model, cutoff, _ = chosen
    fit.pca.n_components = params.k
    fit.pca.save(candidate / "eigenfaces/pca_model.pkl")
    model_classifier(model, cutoff).save(candidate / "classifiers/svm_classifier.pkl")
    known_val = [row for row in validation if row.name != "UNKNOWN"]
    val_weights = project(fit, known_val, params.k)
    val_names = np.array([row.name for row in known_val])
    for name, clf in (("knn", KNNClassifier()), ("euclidean", EuclideanClassifier())):
        clf.fit(fit.weights[:, :params.k], fit.labels)
        clf.calibrate_threshold(val_weights, val_names)
        clf.save(candidate / f"classifiers/{name}_classifier.pkl")
    payload = {"chosen_experiment": chosen_num, "parameters": asdict(params), "cutoff": cutoff,
               "preprocessing": method if chosen_num in {2, 3} else tuned_method,
               "e2_preprocessing": method, "e4_preprocessing": tuned_method,
               "new_training_captures": len(new_train), "camera_training_captures": len(camera_train),
               "webcam_test_captures": len(webcam_test), "incumbent_validation": dict(incumbent), "results": results}
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out, payload, prep_rows, original_rows, train, parts, flags, roster, len(grid), len(finalists))


def pct(value):
    return f"{100 * value:.1f}%" if value is not None else "—"


def write_report(out, payload, prep_rows, original_rows, train, parts, flags, roster, grid_count, calibrated_count):
    labels = {1: "Saved model", 2: "Selected preprocessing", 3: "+ bounded augmentation",
              4: "+ tuned SVM/PCA", 5: "+ verified webcam enrollment"}
    lines = ["# Webcam recognition study", "", "## Dataset and leakage checks", "",
             "Faces use the local cluster-to-name assignments. YOLO annotations were not used as identity labels. "
             "Visually similar source photos and strict duplicate face crops were grouped before splitting. "
             "Groups that touched the old training split stayed in training. Confirmed same-photo conflicts and "
             "cross-label duplicate crops were excluded. Visual identity flags need human review.", "",
             "| Identity | Independent source groups | Usable training groups |", "|---|---:|---:|"]
    for name in roster:
        lines.append(f"| {name} | {len({r.group for r in original_rows if r.name == name})} | "
                     f"{len({r.group for r in train if r.name == name})} |")
    lines += ["", "Audit flags and split group IDs are in `dataset_audit.json`. "
              "Low-support identities with fewer than 20 independent training groups need more captures.", ""]
    for key, items in flags.items():
        lines.append(f"- {key.replace('_', ' ')}: {len(items)}")
    lines += ["", "## Preprocessing on the same validation faces", "",
              "| Method | Cutoff | Correct | Wrong name | UNKNOWN | Unassigned named |", "|---|---:|---:|---:|---:|---:|"]
    for code, title, cutoff, counts in prep_rows:
        lines.append(f"| {code}. {title} | {cutoff:.3f} | {counts['correct']} | {counts['wrong']} | "
                     f"{counts['unknown']} | {counts['unassigned_named']} |")
    lines += ["", f"Saved model on the same validation faces: `{payload['incumbent_validation']}`.", "",
              f"SVM screening evaluated {grid_count} configurations. {calibrated_count} finalists used grouped sigmoid "
              "calibration. The legacy 0.625 cutoff was tested along with a wider score range for the newly calibrated model. "
              "Thresholds were eligible only if they added no wrong names or unassigned acceptances relative to the saved model on validation. "
              "Selection penalized wrong names twice and "
              "unassigned acceptances once per correct match. All threshold trials are in `validation_cutoffs.csv`.", "",
              f"Selected before test evaluation: E{payload['chosen_experiment']}, preprocessing {payload['preprocessing']}, "
              f"parameters `{payload['parameters']}`, cutoff {payload['cutoff']:.3f}.", "",
              f"E2/E3 used preprocessing {payload['e2_preprocessing']}. The tuned E4/E5 branch used {payload['e4_preprocessing']}.", "",
              "## Untouched classroom test photos", "",
              "| Experiment | Accuracy | Macro F1 | Correct | Wrong | UNKNOWN | Coverage | Unassigned named | Verified FAR | Replay FPS |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for number, result in payload["results"].items():
        m = result["classroom"]
        far = (f"{result['false_acceptances']}/{result['verified_unenrolled']}" if result["verified_unenrolled"] else "Not measured")
        lines.append(f"| E{number} {labels[number]} | {pct(m['accuracy'])} | {pct(m['macro_f1'])} | {m['correct']} | "
                     f"{m['wrong']} | {m['unknown']} ({pct(m['unknown_fraction'])}) | {pct(m['coverage'])} | "
                     f"{m['unassigned_named']}/{m['unassigned_total']} | {far} | {result['replay_fps']:.1f} |")
    lines += ["", "Unassigned crops may include enrolled students. Their naming rate is a proxy, not a false acceptance rate. "
              "Replay FPS includes detection, classification, drawing, and JPEG encoding on the same group image at a "
              "maximum of 1280×720. It is not a hardware-camera or browser refresh measurement.", "",
              "## Per-student recall on the classroom test split", "",
              "| Identity | " + " | ".join(f"E{n}" for n in payload["results"]) + " |",
              "|---|" + "---:|" * len(payload["results"])]
    for name in roster:
        cells = [result["classroom"]["recall"][name] for result in payload["results"].values()]
        lines.append(f"| {name} | " + " | ".join(f"{r['correct']}/{r['total']}" if r['total'] else "—" for r in cells) + " |")
    lines += ["", "The confusion matrices are in `confusion_e*.csv`. An em dash means no independent test face for that identity.",
              "", "## Separate webcam sessions", "",
              f"New confirmed training captures: {payload['new_training_captures']}. Total eligible webcam training captures "
              f"including existing enrollment: {payload['camera_training_captures']}. Fresh final-test captures: {payload['webcam_test_captures']}."]
    for number, result in payload["results"].items():
        m = result["webcam_test"]
        if m:
            lines.append(f"- E{number}: {m['correct']}/{m['total']} correct, {m['wrong']} wrong, {m['unknown']} UNKNOWN.")
    if not payload["webcam_test_captures"]:
        lines.append("No independent fresh webcam accuracy can be reported until a separate final-test session is saved.")
    if 5 not in payload["results"]:
        lines.append("E5 was not run in this study because there were no additional webcam crops compatible with the selected preprocessing. "
                     "The earlier enrollment crops are already aligned and cannot be treated as raw box crops.")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "recognition_study")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    with threadpool_limits(limits=2):
        study(args.output)
