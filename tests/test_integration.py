"""End-to-End Integration Pipeline Test."""
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.utils.config import load_config
from src.data.identity_mapper import IdentityMapper
from src.preprocessing.image_preprocessing import ImagePreprocessor
from src.eigenfaces.eigenfaces_model import EigenfacesModel
from src.recognition.prediction import RecognitionPipeline
from src.recognition.svm_classifier import SVMClassifier


def test_full_pipeline_integration():
    config = load_config()
    label_path = PROJECT_ROOT / config["paths"]["label_file"]
    pca_path = PROJECT_ROOT / config["paths"]["pca_model_dir"] / "pca_model.pkl"
    svm_path = PROJECT_ROOT / config["paths"]["classifier_model_dir"] / "svm_classifier.pkl"

    assert label_path.exists(), "label.txt must exist"
    assert pca_path.exists(), "Trained PCA model must exist"
    assert svm_path.exists(), "Trained SVM classifier must exist"

    # 1. Load trained components
    pca_model = EigenfacesModel(target_shape=tuple(config["preprocessing"]["target_size"]))
    pca_model.load(pca_path)
    assert pca_model.is_trained is True

    svm_clf = SVMClassifier()
    svm_clf.load(svm_path)
    assert svm_clf.is_trained is True

    preprocessor = ImagePreprocessor(target_size=tuple(config["preprocessing"]["target_size"]))

    # 2. Setup recognition pipeline
    pipeline = RecognitionPipeline(
        preprocessor=preprocessor,
        eigenfaces_model=pca_model,
        classifier_type="svm",
        k_components=pca_model.n_components
    )
    pipeline.set_classifier(svm_clf, "svm")

    # 3. Simulate recognizing a synthetic face crop
    dummy_face = np.random.randint(50, 200, (120, 100, 3), dtype=np.uint8)
    res = pipeline.recognize_face(dummy_face)

    assert "identity" in res
    assert "confidence" in res
    assert "weights" in res
    assert len(res["weights"]) == pca_model.n_components

