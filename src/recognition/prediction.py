"""End-to-End Face Recognition Pipeline Module."""
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import cv2
import numpy as np

from src.preprocessing.image_preprocessing import ImagePreprocessor
from src.eigenfaces.eigenfaces_model import EigenfacesModel
from src.recognition.svm_classifier import SVMClassifier
from src.recognition.knn_classifier import KNNClassifier
from src.recognition.euclidean_classifier import EuclideanClassifier
from src.utils.logger import setup_logger

logger = setup_logger("RecognitionPipeline")
WEBCAM_EXPOSURE_LUT = np.round(255 * (np.arange(256) / 255) ** 1.2).astype(np.uint8)
WEBCAM_RETRY_MIN_ORIGINAL_SCORE = 0.40
WEBCAM_RETRY_MIN_CORRECTED_SCORE = 0.70


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

    @property
    def face_alignment(self):
        return self.eigenfaces_model.face_alignment

    @classmethod
    def load_pipeline(cls, pca_path: Path, classifier_path: Path,
                      classifier_type: str = "svm",
                      target_size: Tuple[int, int] = (100, 100)) -> "RecognitionPipeline":
        """Factory method to construct a fully ready pipeline from saved model artifacts."""
        eigen_model = EigenfacesModel(target_shape=target_size).load(pca_path)
        preprocessor = ImagePreprocessor.for_model(eigen_model)

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

    def _project_crop(self, face_crop: np.ndarray) -> np.ndarray:
        vector = self.preprocessor.process_and_flatten(face_crop)
        return self.eigenfaces_model.pca.project(vector, k=self.k_components)

    def _retry_webcam_crop(self, crop, original_weights):
        # A correction may support the original candidate, but must not pick a different name.
        candidate, _ = self.classifier.predict_single(original_weights, apply_rejection=False)
        corrected_crops = (
            lambda: cv2.LUT(crop, WEBCAM_EXPOSURE_LUT),
            lambda: cv2.fastNlMeansDenoisingColored(crop, None, 7, 7, 7, 21),
        )
        for corrected_crop in corrected_crops:
            weights = self._project_crop(corrected_crop())
            identity, score = self.classifier.predict_single(weights)
            if identity != "UNKNOWN" and identity == candidate and score >= WEBCAM_RETRY_MIN_CORRECTED_SCORE:
                return identity, score, weights
        return None

    def recognize_face(self, face_crop: np.ndarray, *, retry_webcam: bool = False) -> Dict[str, Any]:
        return self.recognize_faces([face_crop], retry_webcam=retry_webcam)[0]

    def recognize_faces(self, crops: list[np.ndarray], *, retry_webcam: bool = False) -> list[Dict[str, Any]]:
        """Project all valid face crops in a frame together, keeping each result independent."""
        if not crops:
            return []
        if not self.is_ready or self.classifier is None:
            raise ValueError("Recognition pipeline is not initialized with a trained model.")

        vectors = np.stack([self.preprocessor.process_and_flatten(crop) for crop in crops])
        projections = self.eigenfaces_model.pca.project(vectors, k=self.k_components)
        predictions = (self.classifier.predict_batch(projections) if self.classifier_type == "svm"
                       else [self.classifier.predict_single(weights) for weights in projections])
        return [self._face_result(crop, weights, identity, score, retry_webcam)
                for crop, weights, (identity, score) in zip(crops, projections, predictions)]

    def _face_result(self, face_crop, weights, pred_label, score, retry_webcam):
        """Recognize an identity from a raw face crop.
        
        Returns:
            Dict containing:
              - 'identity': student_id string or 'UNKNOWN'
              - 'confidence': confidence score (or distance)
              - 'distance': distance metric (if applicable)
              - 'weights': PCA projection vector
        """
        if (retry_webcam and self.classifier_type == "svm" and pred_label == "UNKNOWN"
                and getattr(self.eigenfaces_model, "webcam_retry", True)
                and score >= WEBCAM_RETRY_MIN_ORIGINAL_SCORE
                and face_crop.shape == (112, 112, 3)):
            recovered = self._retry_webcam_crop(face_crop, weights)
            if recovered is not None:
                pred_label, score, weights = recovered

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

