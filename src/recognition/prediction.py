"""End-to-End Face Recognition Pipeline Module."""
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import numpy as np

from src.preprocessing.image_preprocessing import ImagePreprocessor
from src.eigenfaces.eigenfaces_model import EigenfacesModel
from src.recognition.svm_classifier import SVMClassifier
from src.recognition.knn_classifier import KNNClassifier
from src.recognition.euclidean_classifier import EuclideanClassifier
from src.utils.logger import setup_logger

logger = setup_logger("RecognitionPipeline")


class RecognitionPipeline:
    """Full inference pipeline combining Preprocessing -> Eigenfaces Projection -> Classifier."""

    def __init__(self, preprocessor: ImagePreprocessor,
                 eigenfaces_model: EigenfacesModel,
                 classifier_type: str = "svm",
                 k_components: Optional[int] = None):
        self.preprocessor = preprocessor
        self.eigenfaces_model = eigenfaces_model
        self.classifier_type = classifier_type.lower()
        self.k_components = k_components or eigenfaces_model.n_components
        
        self.classifier = None
        self.is_ready = False

    @classmethod
    def load_pipeline(cls, pca_path: Path, classifier_path: Path,
                      classifier_type: str = "svm",
                      target_size: Tuple[int, int] = (100, 100)) -> "RecognitionPipeline":
        """Factory method to construct a fully ready pipeline from saved model artifacts."""
        preprocessor = ImagePreprocessor(target_size=target_size)
        eigen_model = EigenfacesModel(target_shape=target_size).load(pca_path)

        clf_type = classifier_type.lower()
        if clf_type == "svm":
            clf = SVMClassifier().load(classifier_path)
        elif clf_type == "knn":
            clf = KNNClassifier().load(classifier_path)
        elif clf_type == "euclidean":
            clf = EuclideanClassifier().load(classifier_path)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        pipeline = cls(preprocessor=preprocessor, eigenfaces_model=eigen_model, classifier_type=clf_type)
        pipeline.set_classifier(clf, clf_type)
        return pipeline

    def set_classifier(self, classifier: Any, classifier_type: str) -> None:
        """Attach a trained classifier."""
        self.classifier = classifier
        self.classifier_type = classifier_type.lower()
        self.is_ready = True

    def recognize_face(self, face_crop: np.ndarray) -> Dict[str, Any]:
        """Recognize an identity from a raw face crop.
        
        Returns:
            Dict containing:
              - 'identity': student_id string or 'UNKNOWN'
              - 'confidence': confidence score (or distance)
              - 'distance': distance metric (if applicable)
              - 'weights': PCA projection vector
        """
        if not self.is_ready or self.classifier is None:
            raise ValueError("Recognition pipeline is not initialized with a trained model.")

        # 1. Preprocess & flatten
        flat_vec = self.preprocessor.process_and_flatten(face_crop)

        # 2. Project into Eigenfaces subspace
        weights = self.eigenfaces_model.pca.project(flat_vec, k=self.k_components)

        # 3. Classify
        pred_label, score = self.classifier.predict_single(weights)

        result = {
            "identity": pred_label,
            "classifier": self.classifier_type,
            "weights": weights
        }

        if self.classifier_type == "svm":
            result["confidence"] = score
            result["distance"] = None
        else:
            result["confidence"] = 1.0 / (1.0 + score)  # Inverted distance as confidence
            result["distance"] = score

        return result

