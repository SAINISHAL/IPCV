"""Experimental Protocol, Model Training, and Evaluation Orchestrator."""
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time
import cv2
import numpy as np
from collections import defaultdict

from src.data.archive_extractor import ArchiveExtractor
from src.data.annotation_parser import AnnotationParser
from src.data.identity_mapper import IdentityMapper
from src.preprocessing.image_preprocessing import ImagePreprocessor
from src.preprocessing.quality_filter import QualityFilter
from src.eigenfaces.eigenfaces_model import EigenfacesModel
from src.eigenfaces.pca_from_scratch import PCAFromScratch
from src.recognition.euclidean_classifier import EuclideanClassifier
from src.recognition.knn_classifier import KNNClassifier
from src.recognition.svm_classifier import SVMClassifier
from src.recognition.unknown_detector import UnknownDetector
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.confusion_matrix import ConfusionMatrixVisualizer
from src.evaluation.visualization import EvaluationVisualizer
from src.utils.logger import setup_logger

logger = setup_logger("ExperimentRunner")


class ExperimentRunner:
    """Orchestrates leakage-free dataset splitting, training, calibration, and benchmarking."""

    def __init__(self, raw_data_dir: str or Path, label_path: str or Path,
                 models_dir: str or Path, figures_dir: str or Path,
                 target_size: Tuple[int, int] = (100, 100)):
        self.raw_data_dir = Path(raw_data_dir)
        self.label_path = Path(label_path)
        self.models_dir = Path(models_dir)
        self.figures_dir = Path(figures_dir)
        self.target_size = target_size

        self.identity_mapper = IdentityMapper(self.label_path)
        self.preprocessor = ImagePreprocessor(target_size=self.target_size, enable_clahe=True)

        self.train_samples = []  # list of (face_vec, student_id, photo_id)
        self.val_samples = []
        self.test_samples = []

        self.unknown_test_samples = []  # un-enrolled / unverified faces for FAR testing

    def prepare_dataset_splits(self, train_ratio: float = 0.70,
                               val_ratio: float = 0.15,
                               seed: int = 42) -> Dict[str, int]:
        """Partition faces into Train, Val, Test strictly by photograph to avoid data leakage."""
        logger.info("Loading images and preparing leakage-free photograph-level splits...")
        np.random.seed(seed)
        
        # Group faces by photo identifier: photo_id -> list of (crop, student_id)
        photo_groups = defaultdict(list)
        unknown_faces = []

        student_dirs = sorted([d for d in self.raw_data_dir.iterdir() if d.is_dir()])
        photo_idx = 0

        for sdir in student_dirs:
            archive_name = sdir.name
            is_verified = self.identity_mapper.is_archive_verified(archive_name)
            pairs = ArchiveExtractor.resolve_pairs(sdir)

            for img_path, txt_path in pairs:
                photo_idx += 1
                photo_id = f"{archive_name}_p{photo_idx}"
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]

                if txt_path and txt_path.exists():
                    boxes = AnnotationParser.parse_file(txt_path, deduplicate=True)
                    for box in boxes:
                        x1, y1, x2, y2 = box.to_pixel_coords(w, h)
                        if x2 > x1 + 10 and y2 > y1 + 10:
                            crop = img[y1:y2, x1:x2]
                            # Check quality
                            ok, _ = QualityFilter.is_acceptable_quality(crop, min_size=(24, 24), min_laplacian_var=12.0)
                            if not ok:
                                continue

                            flat = self.preprocessor.process_and_flatten(crop)
                            student_label = self.identity_mapper.map_class_to_student(box.class_id, archive_name=archive_name)

                            if is_verified and student_label != IdentityMapper.UNKNOWN_LABEL:
                                photo_groups[photo_id].append((flat, student_label, photo_id))
                            else:
                                unknown_faces.append((flat, IdentityMapper.UNKNOWN_LABEL, photo_id))

        # Shuffle unique photos (not individual crops!)
        unique_photos = sorted(list(photo_groups.keys()))
        np.random.shuffle(unique_photos)

        n_photos = len(unique_photos)
        n_train = max(1, int(round(n_photos * train_ratio)))
        n_val = max(1, int(round(n_photos * val_ratio)))

        train_photos = set(unique_photos[:n_train])
        val_photos = set(unique_photos[n_train:n_train + n_val])
        test_photos = set(unique_photos[n_train + n_val:])

        self.train_samples = []
        self.val_samples = []
        self.test_samples = []

        for p_id in unique_photos:
            items = photo_groups[p_id]
            if p_id in train_photos:
                self.train_samples.extend(items)
            elif p_id in val_photos:
                self.val_samples.extend(items)
            else:
                self.test_samples.extend(items)

        # Use unknown faces in test set for unknown rejection evaluation
        self.unknown_test_samples = unknown_faces[:25]
        self.test_samples.extend(self.unknown_test_samples)

        stats = {
            "train_faces": len(self.train_samples),
            "val_faces": len(self.val_samples),
            "test_faces": len(self.test_samples),
            "test_unknown_faces": len(self.unknown_test_samples),
            "total_photos": n_photos
        }
        logger.info(f"Dataset split completed: {stats}")
        return stats

    def run_experiments(self, candidate_k_values: List[int] = [5, 10, 15, 20, 25, 30, 40, 50]) -> Dict[str, Any]:
        """Train PCA model and benchmark Euclidean, KNN, and SVM across various K components."""
        if not self.train_samples:
            raise ValueError("Dataset has not been split. Call prepare_dataset_splits() first.")

        X_train = np.array([s[0] for s in self.train_samples])
        y_train = np.array([s[1] for s in self.train_samples])

        X_val = np.array([s[0] for s in self.val_samples])
        y_val = np.array([s[1] for s in self.val_samples])

        X_test = np.array([s[0] for s in self.test_samples])
        y_test = np.array([s[1] for s in self.test_samples])

        enrolled_classes = sorted(list(set(y_train)))
        logger.info(f"Training Eigenfaces on {len(X_train)} samples across {len(enrolled_classes)} enrolled identities...")

        # 1. Fit Master PCA model from scratch
        eigenfaces_model = EigenfacesModel(target_shape=self.target_size, n_components=max(candidate_k_values))
        eigenfaces_model.train(X_train)

        # Save eigenfaces visualization & variance curves
        eigenfaces_model.plot_eigenfaces_grid(n_eigenfaces=10, save_path=self.figures_dir / "06_eigenfaces_basis.png")
        EvaluationVisualizer.plot_variance_curves(eigenfaces_model.pca, save_path=self.figures_dir / "07_pca_variance_curves.png")

        # Sample face reconstruction visualization
        sample_orig = X_train[0]
        EvaluationVisualizer.plot_face_reconstructions(
            sample_orig, eigenfaces_model.pca, target_shape=self.target_size,
            k_steps=[5, 10, 20, 30, 40], save_path=self.figures_dir / "08_face_reconstructions.png"
        )

        # 2. Benchmark K values on Validation Set
        k_results = {"Euclidean": [], "KNN": [], "SVM": []}
        valid_k_values = [k for k in candidate_k_values if k <= len(eigenfaces_model.pca.eigenvalues)]

        best_k = 25
        best_val_score = -1.0
        best_models = {}

        for k in valid_k_values:
            W_train = eigenfaces_model.pca.project(X_train, k=k)
            W_val = eigenfaces_model.pca.project(X_val, k=k)

            # A. Euclidean
            clf_euc = EuclideanClassifier()
            clf_euc.fit(W_train, y_train)
            clf_euc.calibrate_threshold(W_val, y_val)
            pred_euc = clf_euc.predict(W_val)
            acc_euc = np.mean([1 if p == t else 0 for p, t in zip(pred_euc, y_val)])
            k_results["Euclidean"].append(acc_euc)

            # B. KNN
            clf_knn = KNNClassifier(n_neighbors=3)
            clf_knn.fit(W_train, y_train)
            clf_knn.calibrate_threshold(W_val, y_val)
            pred_knn = clf_knn.predict(W_val)
            acc_knn = np.mean([1 if p == t else 0 for p, t in zip(pred_knn, y_val)])
            k_results["KNN"].append(acc_knn)

            # C. SVM
            clf_svm = SVMClassifier(kernel="rbf", C=10.0)
            clf_svm.fit(W_train, y_train)
            clf_svm.calibrate_threshold(W_val, y_val)
            pred_svm = clf_svm.predict(W_val)
            acc_svm = np.mean([1 if p == t else 0 for p, t in zip(pred_svm, y_val)])
            k_results["SVM"].append(acc_svm)

            if acc_svm > best_val_score:
                best_val_score = acc_svm
                best_k = k
                best_models = {
                    "Euclidean": clf_euc,
                    "KNN": clf_knn,
                    "SVM": clf_svm
                }

        # Plot K experiment comparison
        EvaluationVisualizer.plot_k_experiment_comparison(
            valid_k_values, k_results, metric_name="Validation Accuracy",
            save_path=self.figures_dir / "09_k_component_comparison.png"
        )

        logger.info(f"Optimal K selected on validation split: K = {best_k} (Val Acc: {best_val_score:.3f})")

        # 3. Final Evaluation on UNSEEN Test Set using Optimal K
        W_train_opt = eigenfaces_model.pca.project(X_train, k=best_k)
        W_test_opt = eigenfaces_model.pca.project(X_test, k=best_k)

        final_evaluation = {}
        for m_name in ["Euclidean", "KNN", "SVM"]:
            clf = best_models[m_name]
            
            # Measure inference latency
            t0 = time.perf_counter()
            test_preds = clf.predict(W_test_opt)
            latency_ms = ((time.perf_counter() - t0) / len(W_test_opt)) * 1000.0
            fps = 1000.0 / latency_ms if latency_ms > 0 else 0

            # General metrics
            metrics = EvaluationMetrics.compute_all_metrics(y_test, test_preds, enrolled_classes)

            # Unknown rejection metrics
            unk_metrics = UnknownDetector.evaluate_unknown_rejection(y_test, test_preds, set(enrolled_classes))

            # Confusion Matrix
            cm_fig_path = self.figures_dir / f"10_confusion_matrix_{m_name.lower()}.png"
            ConfusionMatrixVisualizer.plot_matrix(
                y_test, test_preds, classes=enrolled_classes + [IdentityMapper.UNKNOWN_LABEL],
                title=f"Confusion Matrix: Eigenfaces + {m_name} (K={best_k})", save_path=cm_fig_path
            )

            final_evaluation[m_name] = {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "FAR": unk_metrics["FAR"],
                "FRR": unk_metrics["FRR"],
                "latency_ms": latency_ms,
                "fps": fps
            }

        # 4. Save Final Production Models
        eigenfaces_model.n_components = best_k
        eigenfaces_model.save(self.models_dir / "eigenfaces" / "pca_model.pkl")
        best_models["SVM"].save(self.models_dir / "classifiers" / "svm_classifier.pkl")
        best_models["KNN"].save(self.models_dir / "classifiers" / "knn_classifier.pkl")
        best_models["Euclidean"].save(self.models_dir / "classifiers" / "euclidean_classifier.pkl")

        logger.info("All models evaluated and saved successfully.")
        return {
            "best_k": best_k,
            "k_experiments": k_results,
            "final_evaluation": final_evaluation,
            "enrolled_classes": enrolled_classes
        }

