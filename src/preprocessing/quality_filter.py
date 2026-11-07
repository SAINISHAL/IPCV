"""Image Quality Filtering Module."""
import cv2
import numpy as np
from typing import Tuple


class QualityFilter:
    """Filters low-quality, blurry, or degenerate face images."""

    @staticmethod
    def compute_blur_laplacian_variance(image: np.ndarray) -> float:
        """Compute variance of the Laplacian as a metric of image sharpness."""
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(laplacian.var())
        return variance

    @classmethod
    def is_acceptable_quality(cls, image: np.ndarray,
                              min_size: Tuple[int, int] = (24, 24),
                              min_laplacian_var: float = 15.0) -> Tuple[bool, str]:
        """Assess if face crop has acceptable sharpness and size."""
        if image is None or image.size == 0:
            return False, "Empty or None image"
            
        h, w = image.shape[:2]
        if w < min_size[0] or h < min_size[1]:
            return False, f"Crop size {w}x{h} smaller than minimum {min_size[0]}x{min_size[1]}"
            
        blur_score = cls.compute_blur_laplacian_variance(image)
        if blur_score < min_laplacian_var:
            return False, f"Blurry face: Laplacian variance {blur_score:.2f} < threshold {min_laplacian_var:.2f}"
            
        # Check for near-solid color degenerate crop
        std_intensity = float(image.std())
        if std_intensity < 8.0:
            return False, f"Degenerate crop: Intensity std {std_intensity:.2f} too low (near flat color)"
            
        return True, "Quality acceptable"

