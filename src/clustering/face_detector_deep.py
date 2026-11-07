"""Pretrained Face Detector and Crop Extractor (Independent of YOLO annotations)."""
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from src.utils.logger import setup_logger

logger = setup_logger("DeepFaceDetector")


class DeepFaceDetector:
    """Detects all faces in images using YuNet (OpenCV Zoo) and extracts aligned normalized crops."""

    def __init__(self, yunet_model_path: str or Path, sface_model_path: str or Path,
                 conf_threshold: float = 0.50, nms_threshold: float = 0.30):
        self.yunet_model_path = Path(yunet_model_path)
        self.sface_model_path = Path(sface_model_path)
        
        if not self.yunet_model_path.exists():
            raise FileNotFoundError(f"YuNet model not found at {self.yunet_model_path}")
        if not self.sface_model_path.exists():
            raise FileNotFoundError(f"SFace model not found at {self.sface_model_path}")
            
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        self.detector = cv2.FaceDetectorYN.create(
            str(self.yunet_model_path), "", (320, 320),
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            top_k=500
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(self.sface_model_path), "")

    def detect_in_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect all faces in an image and return raw detections and aligned crops."""
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)

        if faces is None or len(faces) == 0:
            return []

        results = []
        for face in faces:
            x, y, bw, bh = face[:4].astype(int)
            conf = float(face[-1])
            
            # Margins padding for square crop
            margin_x = int(bw * 0.15)
            margin_y = int(bh * 0.15)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(w, x + bw + margin_x)
            y2 = min(h, y + bh + margin_y)

            crop = image[y1:y2, x1:x2].copy()
            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                continue

            # Aligned 112x112 portrait crop using 5 facial landmarks
            try:
                aligned = self.recognizer.alignCrop(image, face)
            except Exception:
                aligned = cv2.resize(crop, (112, 112))

            # Feature embedding (128-D)
            feat = self.recognizer.feature(aligned)
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm  # L2 normalize

            results.append({
                "box": [int(x), int(y), int(bw), int(bh)],
                "padded_box": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": conf,
                "crop": crop,
                "aligned": aligned,
                "embedding": feat.flatten().tolist()
            })
            
        return results

