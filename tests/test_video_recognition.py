from types import SimpleNamespace

import cv2
import numpy as np

from src.webcam.video_processor import TemporalPredictionSmoother, VideoProcessor


def test_unaligned_detection_stays_unknown_with_cluster_model(monkeypatch):
    aligned_face = np.zeros((112, 112, 3), dtype=np.uint8)
    captions = []
    monkeypatch.setattr(cv2, "putText", lambda image, caption, *args: captions.append(caption))

    def detect_faces(frame, *, alignment):
        assert alignment == "sface"
        return [
            {"box": (0, 0, 40, 40), "confidence": 0.9, "aligned_crop": aligned_face},
            {"box": (45, 0, 85, 40), "confidence": 0.9, "aligned_crop": None},
        ]

    def recognize_face(crop):
        assert crop is aligned_face
        return {"identity": "Student A", "confidence": 0.8}

    processor = VideoProcessor(
        detector=SimpleNamespace(detect_faces=detect_faces),
        recognition_pipeline=SimpleNamespace(
            eigenfaces_model=SimpleNamespace(face_alignment="sface"),
            recognize_face=recognize_face,
        ),
        enable_smoothing=False,
    )
    _, meta = processor.process_frame(np.zeros((100, 100, 3), dtype=np.uint8), is_video_stream=False)
    assert [face["identity"] for face in meta["faces"]] == ["Student A", "UNKNOWN"]
    assert (meta["detected"], meta["recognized"], meta["unknown"]) == (2, 1, 1)
    assert captions == ["Student A (0.80)", "MOVE INTO FRAME"]


def test_webcam_history_never_labels_a_rejected_face_as_known():
    smoother = TemporalPredictionSmoother()
    face = {"box": (0, 0, 40, 40), "identity": "Student A"}
    for _ in range(3):
        assert smoother.smooth([face])[0]["identity"] == "Student A"

    next_frame = [{**face, "identity": "UNKNOWN"}, {**face, "box": (15, 0, 55, 40)}]
    first, second = smoother.smooth(next_frame)
    assert first["identity"] == "UNKNOWN"
    assert first["track_id"] != second["track_id"]
