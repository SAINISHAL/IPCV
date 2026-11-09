"""Face Alignment and Preprocessing Module.

Applies 2D similarity transformation based on eye landmarks to normalize roll rotation,
preserves aspect ratios, prevents facial distortion, and extracts standardized face crops.
"""
from typing import Tuple, Optional, Dict, Any, List
import cv2
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("FaceAligner")

# Match the 112×112 reference crop used to train the saved Eigenfaces model.
# Reference: OpenCV modules/objdetect/src/face_recognize.cpp (Apache-2.0).
_REFERENCE_LANDMARKS = np.float32([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
])
_REFERENCE_CENTER = np.float32([56.0262, 71.9008])
_CENTERED_REFERENCE = _REFERENCE_LANDMARKS - _REFERENCE_CENTER


def align_five_point_crop(image: np.ndarray, landmarks: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Fit a rotation, uniform scale and translation to the training landmarks."""
    if landmarks is None or landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
        return None

    points = landmarks.astype(np.float32, copy=False)
    center = points.mean(axis=0)
    centered = points - center
    energy = np.sum(centered * centered, dtype=np.float64)
    if energy <= np.finfo(np.float32).eps:
        return None

    # Solve the least-squares similarity transform without reflection.
    cross = (_CENTERED_REFERENCE[:, :, None] * centered[:, None, :]).sum(axis=0, dtype=np.float64)
    a = (cross[0, 0] + cross[1, 1]) / energy
    b = (cross[1, 0] - cross[0, 1]) / energy
    if np.hypot(a, b) <= np.finfo(np.float32).eps:
        return None
    linear = np.array([[a, -b], [b, a]])
    offset = _REFERENCE_CENTER - linear @ center
    return cv2.warpAffine(image, np.column_stack((linear, offset)), (112, 112), flags=cv2.INTER_LINEAR)


class FaceAligner:
    """Aligns faces using 5 facial landmarks or aspect-ratio preserving square cropping."""

    def __init__(self, desired_face_size: Tuple[int, int] = (100, 100),
                 desired_left_eye: Tuple[float, float] = (0.32, 0.38)):
        self.desired_w, self.desired_h = desired_face_size
        self.desired_left_eye_x = desired_left_eye[0]
        self.desired_left_eye_y = desired_left_eye[1]

    def align_face(self, image: np.ndarray,
                   box: Tuple[int, int, int, int],
                   landmarks: Optional[np.ndarray] = None,
                   margin_ratio: float = 0.15) -> np.ndarray:
        """Align face using eye landmarks if provided; otherwise extract square padded crop.
        
        Args:
            image: Full original image (BGR)
            box: (x1, y1, x2, y2) bounding box
            landmarks: Optional (5, 2) array of coordinates:
                       [0]: right eye (subject's right, viewer's left)
                       [1]: left eye (subject's left, viewer's right)
                       [2]: nose
                       [3]: right mouth
                       [4]: left mouth
            margin_ratio: Margin to add around face bounding box
            
        Returns:
            Aligned face crop resized to desired_face_size without aspect ratio distortion.
        """
        img_h, img_w = image.shape[:2]
        x1, y1, x2, y2 = box

        # 1. Landmark-based Affine Alignment (Roll rotation correction)
        if landmarks is not None and len(landmarks) >= 2:
            try:
                # Landmark indices in OpenCV YuNet:
                # 0: right eye, 1: left eye (viewer's left and right)
                right_eye = landmarks[0]
                left_eye = landmarks[1]

                # Compute the angle between the eye centroids
                d_y = left_eye[1] - right_eye[1]
                d_x = left_eye[0] - right_eye[0]
                dist = np.sqrt(d_x ** 2 + d_y ** 2)

                if dist > 5.0:  # Valid eye distance
                    angle = float(np.degrees(np.arctan2(d_y, d_x)))

                    # Desired right eye position in normalized coordinates
                    desired_right_eye_x = 1.0 - self.desired_left_eye_x

                    # Determine the scale of the new resulting image
                    desired_dist = (desired_right_eye_x - self.desired_left_eye_x) * self.desired_w
                    scale = desired_dist / dist

                    # Compute center between eyes in original image
                    eyes_center = ((right_eye[0] + left_eye[0]) // 2, (right_eye[1] + left_eye[1]) // 2)

                    # Get rotation matrix
                    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

                    # Update translation component of the matrix
                    t_x = self.desired_w * 0.5
                    t_y = self.desired_h * self.desired_left_eye_y
                    M[0, 2] += (t_x - eyes_center[0])
                    M[1, 2] += (t_y - eyes_center[1])

                    # Apply affine transformation
                    aligned = cv2.warpAffine(
                        image, M, (self.desired_w, self.desired_h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REFLECT_101
                    )
                    return aligned
            except Exception as e:
                logger.debug(f"Landmark alignment fallback due to: {e}")

        # 2. Geometric Square Cropping with Symmetric Margin (Preserves 1:1 Aspect Ratio)
        bw = x2 - x1
        bh = y2 - y1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Make crop square based on the larger dimension plus margin
        side = int(max(bw, bh) * (1.0 + margin_ratio * 2.0))
        half_side = side // 2

        crop_x1 = max(0, cx - half_side)
        crop_y1 = max(0, cy - half_side)
        crop_x2 = min(img_w, cx + half_side)
        crop_y2 = min(img_h, cy + half_side)

        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

        if crop.size == 0:
            # Fallback to simple crop
            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(img_w, x2), min(img_h, y2)
            crop = image[y1_c:y2_c, x1_c:x2_c]
            if crop.size == 0:
                return np.zeros((self.desired_h, self.desired_w, 3), dtype=np.uint8)

        # Pad with border reflect if crop touches image boundaries to prevent aspect distortion
        ch, cw = crop.shape[:2]
        pad_top = max(0, half_side - (cy - crop_y1))
        pad_bottom = max(0, half_side - (crop_y2 - cy))
        pad_left = max(0, half_side - (cx - crop_x1))
        pad_right = max(0, half_side - (crop_x2 - cx))

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            crop = cv2.copyMakeBorder(
                crop, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REFLECT_101
            )

        # Standardized resize
        resized = cv2.resize(crop, (self.desired_w, self.desired_h), interpolation=cv2.INTER_AREA)
        return resized

