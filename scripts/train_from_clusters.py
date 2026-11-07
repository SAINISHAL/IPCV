"""Train Eigenfaces and Classifiers from user-confirmed identity_mapping.json."""
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.preprocessing.image_preprocessing import ImagePreprocessor
from src.eigenfaces.eigenfaces_model import EigenfacesModel
from src.recognition.euclidean_classifier import EuclideanClassifier
from src.recognition.knn_classifier import KNNClassifier
from src.recognition.svm_classifier import SVMClassifier
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.confusion_matrix import ConfusionMatrixVisualizer
from src.evaluation.visualization import EvaluationVisualizer

logger = setup_logger("TrainFromClusters")


def load_enrollment_crops(directory, enrolled_names, preprocessor):
    """Load verified, SFace-aligned webcam crops from identity-named folders."""
    vectors = []
    labels = []
    for path in sorted(directory.glob("*/*.png")):
        name = path.parent.name
        if name not in enrolled_names:
            raise ValueError(f"Enrollment crop uses an unrecognized identity folder: {path}")
        crop = cv2.imread(str(path))
        if crop is None or crop.shape[:2] != (112, 112):
            raise ValueError(f"Expected an aligned 112x112 enrollment crop: {path}")

        # Give each verified capture extra weight, with small changes a webcam can produce.
        center = (crop.shape[1] / 2, crop.shape[0] / 2)
        for angle in (-5, 0, 5):
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(crop, matrix, (112, 112), borderMode=cv2.BORDER_REFLECT_101)
            for brightness in (-22, 0, 22):
                variation = np.clip(rotated.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
                for sample in (variation, cv2.flip(variation, 1)):
                    vectors.append(preprocessor.process_and_flatten(sample))
                    labels.append(name)
    return vectors, labels


def score_svm(svm, val_weights, val_labels, unassigned_weights):
    """Prefer correct names while penalizing wrong names and uncertain assignments."""
    scores = svm.model.predict_proba(val_weights)
    top_scores = scores.max(axis=1)
    correct_name = svm.classes_[scores.argmax(axis=1)] == val_labels
    unassigned_scores = (svm.model.predict_proba(unassigned_weights).max(axis=1)
                         if len(unassigned_weights) else np.empty(0))

    cutoff = svm.confidence_threshold
    accepted = top_scores >= cutoff
    correct = np.count_nonzero(accepted & correct_name)
    wrong = np.count_nonzero(accepted & ~correct_name)
    unassigned_accepted = np.count_nonzero(unassigned_scores >= cutoff)
    return (correct - 2 * wrong - 0.5 * unassigned_accepted, correct, -wrong, -unassigned_accepted)


def main(output_root: Path = PROJECT_ROOT):
    config = load_config()
    mapping_path = PROJECT_ROOT / "data" / "identity_mapping.json"
    clusters_path = PROJECT_ROOT / "data" / "clusters.json"
    models_dir = output_root / "models"
    figures_dir = output_root / config["paths"]["figures_dir"]
    reports_dir = output_root / config["paths"]["reports_dir"]

    if not mapping_path.exists():
        logger.error(f"Identity mapping not found at {mapping_path}! Please confirm cluster identities first.")
        print(f"[ERROR] Please assign identities to clusters in {mapping_path} before training.")
        return

    with open(mapping_path, "r", encoding="utf-8") as f:
        identity_map = json.load(f)

    with open(clusters_path, "r", encoding="utf-8") as f:
        clusters_data = json.load(f)

    target_size = tuple(config["preprocessing"]["target_size"])
    preprocessor = ImagePreprocessor(target_size=target_size, enable_clahe=True)

    # 1. Group labeled faces by original photo to prevent leakage
    logger.info("Loading confirmed cluster crops and grouping by original photograph...")
    photo_groups = defaultdict(list)
    unassigned_faces = defaultdict(list)

    for cid_str, items in clusters_data["clusters"].items():
        assigned_name = identity_map.get(cid_str, "(Unassigned)").strip()
        is_valid = (assigned_name not in ["(Unassigned)", "REMOVE / FALSE DETECTION", "UNKNOWN", ""] and cid_str != "-1")

        for it in items:
            crop_path = PROJECT_ROOT / it["crop_path"]
            if not crop_path.exists():
                continue
            crop_img = cv2.imread(str(crop_path))
            if crop_img is None:
                continue

            flat_vec = preprocessor.process_and_flatten(crop_img)
            photo_id = it.get("original_image", "unknown_source")

            if is_valid:
                photo_groups[photo_id].append((flat_vec, assigned_name, photo_id))
            elif cid_str == "-1":
                unassigned_faces[photo_id].append(flat_vec)

    unique_photos = sorted(list(photo_groups.keys()))
    np.random.seed(42)
    np.random.shuffle(unique_photos)

    n_photos = len(unique_photos)
    n_train = max(1, int(round(n_photos * 0.70)))
    n_val = max(1, int(round(n_photos * 0.15)))

    train_photos = set(unique_photos[:n_train])
    val_photos = set(unique_photos[n_train:n_train + n_val])
    test_photos = set(unique_photos[n_train + n_val:])

    train_samples = []
    val_samples = []
    test_samples = []

    for p_id in unique_photos:
        items = photo_groups[p_id]
        if p_id in train_photos:
            # Two faces in one photo with the same assigned name are ambiguous.
            name_counts = Counter(item[1] for item in items)
            train_samples.extend(item for item in items if name_counts[item[1]] == 1)
        elif p_id in val_photos:
            val_samples.extend(items)
        else:
            test_samples.extend(items)

    logger.info(f"Splits: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    expected_names = {s[1] for p in train_photos for s in photo_groups[p]}
    if not train_samples or expected_names != {s[1] for s in train_samples}:
        raise ValueError("Some identities have no usable training samples. Review the cluster assignments.")
    width, height = target_size
    X_train = np.stack([crop for sample, _, _ in train_samples
                        for crop in (sample, sample.reshape(height, width)[:, ::-1].ravel())])
    y_train = np.repeat([s[1] for s in train_samples], 2)
    camera_vectors, camera_labels = load_enrollment_crops(
        PROJECT_ROOT / "data" / "enrollment", expected_names, preprocessor
    )
    if camera_vectors:
        X_train = np.concatenate((X_train, np.stack(camera_vectors)))
        y_train = np.concatenate((y_train, camera_labels))
        logger.info(f"Added {len(camera_vectors)} training examples from verified webcam crops and their variations.")

    X_val = np.array([s[0] for s in val_samples])
    y_val = np.array([s[1] for s in val_samples])

    X_test = np.array([s[0] for s in test_samples])
    y_test = np.array([s[1] for s in test_samples])
    X_unassigned_val = np.array([x for p in unique_photos if p in val_photos for x in unassigned_faces[p]])
    X_unassigned_test = np.array([x for p in unique_photos if p in test_photos for x in unassigned_faces[p]])

    enrolled_classes = sorted(list(set(y_train)))
    logger.info(f"Enrolled identities ({len(enrolled_classes)}): {enrolled_classes}")

    # 2. Train PCA from scratch
    pca_model = EigenfacesModel(target_shape=target_size, face_alignment="sface")
    pca_model.train(X_train)

    # Visualizations
    pca_model.plot_eigenfaces_grid(n_eigenfaces=10, save_path=figures_dir / "06_eigenfaces_basis.png")
    EvaluationVisualizer.plot_variance_curves(pca_model.pca, save_path=figures_dir / "07_pca_variance_curves.png")
    EvaluationVisualizer.plot_face_reconstructions(
        X_train[0], pca_model.pca, target_shape=target_size,
        k_steps=[5, 10, 15, 20, 25], save_path=figures_dir / "08_face_reconstructions.png"
    )

    # 3. Benchmark on validation set and select best K
    max_k = len(pca_model.pca.eigenvalues)
    valid_k = sorted({min(k, max_k) for k in (20, 40, 80)})
    best_k = valid_k[0]
    best_rank = None
    best_models = {}
    k_results = {"Euclidean": [], "KNN": [], "SVM": []}
    svm_cutoff = config["recognition"]["unknown_rejection"]["svm_confidence_threshold"]

    for k in valid_k:
        W_train = pca_model.pca.project(X_train, k=k)
        W_val = pca_model.pca.project(X_val, k=k)
        W_unassigned_val = (pca_model.pca.project(X_unassigned_val, k=k)
                            if len(X_unassigned_val) else np.empty((0, k)))

        euc = EuclideanClassifier()
        euc.fit(W_train, y_train)
        euc.calibrate_threshold(W_val, y_val)
        acc_euc = np.mean([1 if p == t else 0 for p, t in zip(euc.predict(W_val), y_val)])
        k_results["Euclidean"].append(acc_euc)

        knn = KNNClassifier(n_neighbors=3)
        knn.fit(W_train, y_train)
        knn.calibrate_threshold(W_val, y_val)
        acc_knn = np.mean([1 if p == t else 0 for p, t in zip(knn.predict(W_val), y_val)])
        k_results["KNN"].append(acc_knn)

        svm = SVMClassifier(kernel="rbf", C=10.0, scale_features=True, confidence_threshold=svm_cutoff)
        svm.fit(W_train, y_train)
        val_rank = score_svm(svm, W_val, y_val, W_unassigned_val)
        acc_svm = np.mean([1 if p == t else 0 for p, t in zip(svm.predict(W_val), y_val)])
        k_results["SVM"].append(acc_svm)

        if best_rank is None or val_rank > best_rank:
            best_rank = val_rank
            best_k = k
            best_models = {"Euclidean": euc, "KNN": knn, "SVM": svm}

    EvaluationVisualizer.plot_k_experiment_comparison(
        valid_k, k_results, metric_name="Validation Accuracy",
        save_path=figures_dir / "09_k_component_comparison.png"
    )

    logger.info(f"Selected K={best_k} and SVM cutoff={best_models['SVM'].confidence_threshold:.3f}")

    # 4. Evaluate on unseen Test Set
    W_test_opt = pca_model.pca.project(X_test, k=best_k)
    W_unassigned_test = (pca_model.pca.project(X_unassigned_test, k=best_k)
                         if len(X_unassigned_test) else np.empty((0, best_k)))

    print("\n" + "=" * 60)
    print("FINAL BENCHMARK ON UNSEEN TEST SET (FROM CONFIRMED CLUSTERS)")
    print("=" * 60)

    final_metrics = {}
    for m_name in ["Euclidean", "KNN", "SVM"]:
        clf = best_models[m_name]
        preds = np.array(clf.predict(W_test_opt))
        metrics = EvaluationMetrics.compute_all_metrics(y_test, preds, enrolled_classes)
        rejected = preds == "UNKNOWN"

        final_metrics[m_name] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "wrong_names": int(np.count_nonzero(~rejected & (preds != y_test))),
            "rejected": int(np.count_nonzero(rejected)),
            "unassigned_named": sum(name != "UNKNOWN" for name in clf.predict(W_unassigned_test))
        }

        print(f"Model ({m_name:10s}): Test Acc={metrics['accuracy']:.4f} | "
              f"Wrong names={final_metrics[m_name]['wrong_names']} | Rejected={final_metrics[m_name]['rejected']}")

        # Confusion matrices
        ConfusionMatrixVisualizer.plot_matrix(
            y_test, preds, classes=enrolled_classes + ["UNKNOWN"],
            title=f"Confusion Matrix: Eigenfaces + {m_name} (Confirmed Identities, K={best_k})",
            save_path=figures_dir / f"10_confusion_matrix_{m_name.lower()}.png"
        )

    # 5. Save updated models
    pca_model.n_components = best_k
    pca_model.save(models_dir / "eigenfaces" / "pca_model.pkl")
    best_models["SVM"].save(models_dir / "classifiers" / "svm_classifier.pkl")
    best_models["KNN"].save(models_dir / "classifiers" / "knn_classifier.pkl")
    best_models["Euclidean"].save(models_dir / "classifiers" / "euclidean_classifier.pkl")
    logger.info("Updated models saved successfully!")

    # 6. Update Model Evaluation Report Markdown
    eval_report_path = reports_dir / "model_evaluation_report.md"
    lines = [
        "# Model Evaluation Report (Trained on Confirmed Cluster Identities)",
        "",
        "## 1. Experimental Overview",
        "",
        "- **Face Detection**: Pretrained YuNet Face Detector (independent of legacy YOLO labels)",
        "- **Identity Enrollment**: SFace/DBSCAN clusters with names from `data/identity_mapping.json`",
        "- **Training Data**: Excludes faces with conflicting same-name assignments in a single training photo and adds mirrored crops",
        f"- **Webcam Enrollment**: {len(camera_vectors)} training examples from labeled webcam crops and their variations. These captures are excluded from classroom validation and test splits.",
        "- **Feature Extraction**: Mathematical Eigenfaces (Pure NumPy Dual PCA)",
        f"- **Optimal Subspace Dimensionality (K)**: {best_k} principal components",
        "- **Data Splitting**: 70% train, 15% validation, and 15% test by source photograph",
        f"- **SVM**: RBF classifier on standardized PCA weights, with a configured cutoff of {svm_cutoff:.3f}",
        "",
        "## 2. Quantitative Model Benchmark on Unseen Test Split",
        "",
        f"Results on {len(y_test)} labeled face crops and {len(W_unassigned_test)} unassigned crops.",
        "",
        "| Classifier | Accuracy on labeled crops | Macro F1 | Wrong name | Returned UNKNOWN | Unassigned given a name |",
        "|---|---|---|---|---|---|",
    ]

    for model_name, m in final_metrics.items():
        lines.append(
            f"| {model_name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
            f"{m['wrong_names']} | {m['rejected']} | {m['unassigned_named']} |"
        )

    lines.extend([
        "",
        "Unassigned crops have no verified identity and may contain enrolled people. The final column counts how often the model assigned them a name. Measuring false acceptance requires photos of verified unenrolled people.",
        ""
    ])

    with open(eval_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Updated model evaluation report at: {eval_report_path}")


if __name__ == "__main__":
    main()
