from types import SimpleNamespace

import numpy as np
import pytest

from src.webcam.enrollment import save_capture


def test_confirmed_sessions_reject_cross_split_duplicate_and_multiple_faces(tmp_path):
    rng = np.random.default_rng(9)
    frame = rng.integers(0, 256, (160, 180, 3), dtype=np.uint8)

    def face(image):
        return {"raw_box": (28, 20, 140, 132), "aligned_crop": image[20:132, 28:140]}

    detector = SimpleNamespace(detect_faces=lambda image, **kwargs: [face(image)])
    saved = save_capture(tmp_path, "train", "Student A", "one", 1, frame, detector)
    assert saved.relative_to(tmp_path).parts == ("train", "Student A", "one", "capture_001.jpg")

    with pytest.raises(ValueError, match="too similar"):
        save_capture(tmp_path, "test", "Student A", "two", 1, frame, detector)
    new_frame = rng.integers(0, 256, frame.shape, dtype=np.uint8)
    saved_test = save_capture(tmp_path, "test", "Student A", "two", 1, new_frame, detector)
    assert saved_test.is_file()

    detector.detect_faces = lambda image, **kwargs: [face(image), face(image)]
    with pytest.raises(ValueError, match="only the person"):
        save_capture(tmp_path, "train", "Student A", "three", 1, frame, detector)
