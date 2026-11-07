"""YOLO Face Detector Module."""
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from src.utils.logger import setup_logger

logger = setup_logger("YOLODetector")


class YOLODetector:
    """Detects faces in classroom images and video frames using YOLO."""

    def __init__(self, model_path: Optional[str or Path] = None,
                 fallback_path: Optional[str or Path] = None,
                 conf_thresh: float = 0.25,
                 iou_thresh: float = 0.45):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model = None
        self.model_source = "None"
        
        # Check custom weights first
        if model_path and Path(model_path).exists():
            try:
                logger.info(f"Loading custom YOLO weights: {model_path}")
                self.model = YOLO(str(model_path))
                self.model_source = f"Custom: {Path(model_path).name}"
            except Exception as e:
                logger.warning(f"Could not load custom weights {model_path}: {e}")

        # Fallback weights
        if self.model is None and fallback_path and Path(fallback_path).exists():
            try:
                logger.info(f"Loading fallback YOLO weights: {fallback_path}")
                self.model = YOLO(str(fallback_path))
                self.model_source = f"Fallback: {Path(fallback_path).name}"
            except Exception as e:
                logger.warning(f"Could not load fallback weights {fallback_path}: {e}")

        # Final fallback: OpenCV Haar cascade face detector if YOLO fails
        self.haar_detector = None
        if self.model is None:
            logger.warning("No YOLO weights available! Initializing OpenCV Haar face detector fallback.")
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.haar_detector = cv2.CascadeClassifier(cascade_path)
            self.model_source = "OpenCV Haar Cascade Fallback"

    def detect_faces(self, image: np.ndarray,
                     expand_ratio: float = 0.08,
                     min_size: Tuple[int, int] = (24, 24)) -> List[Dict[str, Any]]:
        """Detect faces in an RGB/BGR image.
        
        Returns:
            List of dicts:
              - 'box': (x1, y1, x2, y2) in pixel space
              - 'confidence': detection score float
              - 'class_id': detection class
        """
        if image is None or image.size == 0:
            return []
            
        h, w = image.shape[:2]
        min_w, min_h = min_size
        detections = []

        if self.model is not None:
            try:
                results = self.model(image, conf=self.conf_thresh, iou=self.iou_thresh, verbose=False)
                for res in results:
                    boxes = res.boxes
                    for b in boxes:
                        coords = b.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, coords)
                        conf = float(b.conf[0].cpu().numpy())
                        cls_id = int(b.cls[0].cpu().numpy())

                        # Apply expand ratio for natural face margins
                        bw = x2 - x1
                        bh = y2 - y1
                        dx = int(bw * expand_ratio)
                        dy = int(bh * expand_ratio)
                        x1 = max(0, x1 - dx)
                        y1 = max(0, y1 - dy)
                        x2 = min(w, x2 + dx)
                        y2 = min(h, y2 + dy)

                        if (x2 - x1) >= min_w and (y2 - y1) >= min_h:
                            detections.append({
                                "box": (x1, y1, x2, y2),
                                "confidence": conf,
                                "class_id": cls_id
                            })
            except Exception as e:
                logger.error(f"YOLO inference error: {e}")

        # If YOLO detected nothing or unavailable, and Haar fallback is present
        if len(detections) == 0 and self.haar_detector is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            faces = self.haar_detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=min_size)
            for (x, y, fw, fh) in faces:
                detections.append({
                    "box": (x, y, x + fw, y + fh),
                    "confidence": 0.85,
                    "class_id": 0
                })

        return detections

