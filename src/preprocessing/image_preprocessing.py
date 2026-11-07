"""Image Preprocessing Module for Eigenfaces Pipeline."""
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import cv2
import numpy as np

from src.preprocessing.face_alignment import FaceAligner
from src.preprocessing.quality_filter import QualityFilter
from src.utils.logger import setup_logger

logger = setup_logger("ImagePreprocessor")


class ImagePreprocessor:
    """Provides identical deterministic preprocessing for training, evaluation, and live webcam inference."""

    def __init__(self, target_size: Tuple[int, int] = (100, 100),
                 enable_clahe: bool = True,
                 clahe_clip_limit: float = 2.0,
                 clahe_grid_size: Tuple[int, int] = (8, 8),
                 enable_alignment: bool = False,
                 normalize_intensity: bool = True):
        self.target_size = target_size
        self.enable_clahe = enable_clahe
        self.enable_alignment = enable_alignment
        self.normalize_intensity = normalize_intensity
        
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
        self.aligner = FaceAligner() if enable_alignment else None

    def process(self, image: np.ndarray) -> np.ndarray:
        """Process a single raw face crop.
        
        Steps:
          1. Grayscale conversion.
          2. Optional alignment.
          3. Resize to target dimensions (e.g. 100x100).
          4. CLAHE contrast equalization.
          5. Pixel intensity normalization.
          
        Returns:
            Normalized 2D float32 array of shape (height, width).
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")

        # 1. Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 2. Optional alignment
        if self.aligner is not None:
            gray = self.aligner.align(gray)

        # 3. Resize
        target_w, target_h = self.target_size
        h, w = gray.shape[:2]
        interp = cv2.INTER_AREA if (w >= target_w and h >= target_h) else cv2.INTER_CUBIC
        resized = cv2.resize(gray, (target_w, target_h), interpolation=interp)

        # 4. CLAHE
        if self.enable_clahe:
            equalized = self.clahe.apply(resized)
        else:
            equalized = cv2.equalizeHist(resized)

        # 5. Normalization
        if self.normalize_intensity:
            # Scale to [0.0, 1.0] as float32
            normalized = equalized.astype(np.float32) / 255.0
        else:
            normalized = equalized.astype(np.float32)

        return normalized

    def process_and_flatten(self, image: np.ndarray) -> np.ndarray:
        """Process image and return 1D flattened vector of size D = width * height."""
        processed = self.process(image)
        return processed.flatten()

    def batch_process(self, images: List[np.ndarray]) -> np.ndarray:
        """Process a list of images into a 2D data matrix of shape (N, D).
        
        Returns:
            NumPy array of shape (N, D).
        """
        vectors = []
        for img in images:
            vec = self.process_and_flatten(img)
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

