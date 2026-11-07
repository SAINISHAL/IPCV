"""Face alignment utility (applied when reliable landmarks or eye detectors are available)."""
import cv2
import numpy as np
from typing import Optional, Tuple


class FaceAligner:
    """Aligns faces using 2D affine transformation based on eye coordinates when available."""

    def __init__(self, eye_cascade_path: Optional[str] = None):
        if eye_cascade_path is None:
            eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    def align(self, face_crop: np.ndarray) -> np.ndarray:
        """Attempt to align face horizontally based on detected eyes.
        
        If eyes cannot be detected reliably, returns original face crop safely.
        """
        if face_crop is None or face_crop.size == 0:
            return face_crop

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
        eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))

        # Only align if exactly two eyes are found in upper half of face
        if len(eyes) == 2:
            # Sort left to right
            eyes = sorted(eyes, key=lambda e: e[0])
            (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes
            eye1_center = (x1 + w1 // 2, y1 + h1 // 2)
            eye2_center = (x2 + w2 // 2, y2 + h2 // 2)

            # Ensure both eyes are roughly in the top half
            fh = face_crop.shape[0]
            if eye1_center[1] < fh * 0.65 and eye2_center[1] < fh * 0.65:
                # Compute angle
                d_y = eye2_center[1] - eye1_center[1]
                d_x = eye2_center[0] - eye1_center[0]
                angle = np.degrees(np.arctan2(d_y, d_x))

                # Limit rotation to reasonable bounds (-30 to +30 degrees)
                if abs(angle) < 30.0:
                    center = ((eye1_center[0] + eye2_center[0]) // 2, (eye1_center[1] + eye2_center[1]) // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    aligned = cv2.warpAffine(face_crop, M, (face_crop.shape[1], face_crop.shape[0]),
                                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    return aligned

        return face_crop

