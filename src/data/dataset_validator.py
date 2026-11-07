"""Dataset validation and integrity checking module."""
import hashlib
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from src.data.annotation_parser import AnnotationParser, BoundingBox
from src.utils.logger import setup_logger

logger = setup_logger("DatasetValidator")


class DatasetValidator:
    """Validates images, bounding boxes, file integrity, and dataset consistency."""

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Compute MD5 hash of a file."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def validate_image(file_path: Path) -> Tuple[bool, Optional[Tuple[int, int]], Optional[str]]:
        """Validate if an image can be successfully read and return (valid, (width, height), error_msg)."""
        try:
            # Read image using OpenCV
            img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                return False, None, "Failed to decode image bytes"
            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                return False, None, f"Invalid image dimensions: {w}x{h}"
            return True, (w, h), None
        except Exception as e:
            return False, None, str(e)

    @classmethod
    def validate_box(cls, box: BoundingBox, img_w: int, img_h: int) -> Tuple[bool, Optional[str]]:
        """Validate an individual bounding box."""
        if not (0.0 <= box.x_center <= 1.0 and 0.0 <= box.y_center <= 1.0):
            return False, f"Center ({box.x_center}, {box.y_center}) out of [0, 1]"
        if not (0.0 < box.width <= 1.0 and 0.0 < box.height <= 1.0):
            return False, f"Dimensions ({box.width}, {box.height}) invalid"
            
        x1, y1, x2, y2 = box.to_pixel_coords(img_w, img_h, clamp=False)
        if x2 <= x1 or y2 <= y1:
            return False, f"Degenerate box: ({x1}, {y1}) to ({x2}, {y2})"
            
        return True, None

