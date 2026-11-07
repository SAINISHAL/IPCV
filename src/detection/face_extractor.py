"""Face Extractor Module for cropping and storing face images from annotations or detection."""
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np

from src.data.annotation_parser import AnnotationParser, BoundingBox
from src.data.identity_mapper import IdentityMapper
from src.utils.logger import setup_logger

logger = setup_logger("FaceExtractor")


class FaceExtractor:
    """Extracts, filters, and organizes face crops by student ID."""

    def __init__(self, output_dir: str or Path, min_face_size: Tuple[int, int] = (24, 24)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_face_size = min_face_size

    def extract_crop(self, image: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Safely crop face from image with boundary checks."""
        if image is None or image.size == 0:
            return None
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        bw = x2 - x1
        bh = y2 - y1
        min_w, min_h = self.min_face_size
        if bw < min_w or bh < min_h:
            return None

        crop = image[y1:y2, x1:x2].copy()
        return crop

    def save_crop(self, crop: np.ndarray, student_id: str, filename: str) -> Path:
        """Save cropped face into student_id subfolder."""
        student_folder = self.output_dir / student_id
        student_folder.mkdir(parents=True, exist_ok=True)
        target_path = student_folder / filename
        cv2.imwrite(str(target_path), crop)
        return target_path

