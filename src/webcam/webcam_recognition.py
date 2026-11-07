"""Live Webcam Face Recognition Application using Unified Detector and Shared Pipeline."""
from pathlib import Path
from typing import Optional, Tuple
import cv2

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.detection.unified_detector import UnifiedFaceDetector
from src.preprocessing.image_preprocessing import ImagePreprocessor
from src.eigenfaces.eigenfaces_model import EigenfacesModel
from src.recognition.prediction import RecognitionPipeline
from src.recognition.svm_classifier import SVMClassifier
from src.recognition.knn_classifier import KNNClassifier
from src.recognition.euclidean_classifier import EuclideanClassifier
from src.webcam.video_processor import VideoProcessor

logger = setup_logger("WebcamRecognition")


class WebcamRecognitionApp:
    """Manages camera lifecycle and real-time face recognition video loop."""

    def __init__(self, config_path: Optional[str or Path] = None):
        self.config = load_config(config_path)
        self.root = Path(__file__).resolve().parent.parent.parent
        self.video_processor: Optional[VideoProcessor] = None
        self.cap: Optional[cv2.VideoCapture] = None

    def initialize_models(self) -> None:
        """Load trained Unified Detector, Eigenfaces model, and classifier once."""
        logger.info("Initializing models for live recognition...")

        # 1. Unified Face Detector (YOLOv8-Face + YuNet Ensemble with Affine Alignment)
        yunet_path = self.root / "models" / "pretrained" / "yunet.onnx"
        yolo_face_path = self.root / "models" / "yolo" / "yolov8n-face.pt"
        detector = UnifiedFaceDetector(
            engine="ensemble",
            yolo_face_path=yolo_face_path,
            yunet_model_path=yunet_path,
            conf_thresh=self.config["detection"].get("confidence_threshold", 0.35),
            align_faces=True
        )

        # 2. Preprocessor (100x100 CLAHE)
        target_size = tuple(self.config["preprocessing"]["target_size"])
        preprocessor = ImagePreprocessor(target_size=target_size, enable_clahe=True)

        # 3. PCA Model
        pca_path = self.root / self.config["paths"]["pca_model_dir"] / "pca_model.pkl"
        eigenfaces_model = EigenfacesModel(target_shape=target_size)
        if not pca_path.exists():
            raise FileNotFoundError(f"PCA model not found at: {pca_path}. Please run train.py first.")
        eigenfaces_model.load(pca_path)

        # 4. Classifier
        clf_type = self.config["recognition"]["classifier_type"].lower()
        clf_dir = self.root / self.config["paths"]["classifier_model_dir"]

        if clf_type == "knn":
            clf_path = clf_dir / "knn_classifier.pkl"
            classifier = KNNClassifier()
        elif clf_type == "euclidean":
            clf_path = clf_dir / "euclidean_classifier.pkl"
            classifier = EuclideanClassifier()
        else:
            clf_path = clf_dir / "svm_classifier.pkl"
            classifier = SVMClassifier()

        if not clf_path.exists():
            raise FileNotFoundError(f"Classifier not found at: {clf_path}. Please run train.py first.")
        classifier.load(clf_path)

        # 5. Full Pipeline
        pipeline = RecognitionPipeline(
            preprocessor=preprocessor,
            eigenfaces_model=eigenfaces_model,
            classifier_type=clf_type,
            k_components=eigenfaces_model.n_components
        )
        pipeline.set_classifier(classifier, clf_type)

        # 6. Video Processor
        self.video_processor = VideoProcessor(
            detector=detector,
            recognition_pipeline=pipeline,
            enable_smoothing=True,
            smoothing_window=self.config["webcam"].get("smoothing_window_size", 5)
        )
        logger.info("Live recognition pipeline initialized successfully.")

    def run(self, device_id: Optional[int] = None, video_source: Optional[str] = None) -> None:
        """Run real-time recognition loop."""
        if self.video_processor is None:
            self.initialize_models()

        if video_source is not None:
            logger.info(f"Opening video source: {video_source}")
            self.cap = cv2.VideoCapture(video_source)
        else:
            dev = device_id if device_id is not None else self.config["webcam"]["device_id"]
            logger.info(f"Opening camera device ID: {dev}...")
            self.cap = cv2.VideoCapture(dev, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(dev)

        if not self.cap.isOpened():
            logger.error(f"Failed to open video source (device {device_id} or file {video_source}).")
            print("\n[ERROR] Camera device could not be opened. Check connection or try another device ID.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["webcam"]["frame_width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["webcam"]["frame_height"])

        window_name = "Classroom Face Recognition (Press 'q' or 'ESC' to exit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        logger.info("Entering live video loop. Press 'q' to exit.")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break

                annotated_frame, meta = self.video_processor.process_frame(frame, is_video_stream=True)

                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    logger.info("Exit key pressed.")
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Webcam released and windows closed.")
