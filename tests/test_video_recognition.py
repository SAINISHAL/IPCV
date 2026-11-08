from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.recognition.prediction import RecognitionPipeline
from src.recognition.svm_classifier import SVMClassifier
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
    assert captions == ["Student A (0.80)", "UNKNOWN - MOVE INTO FRAME"]


def test_webcam_history_holds_a_name_for_one_bad_frame_and_keeps_tracks_separate():
    smoother = TemporalPredictionSmoother()
    face = {"box": (0, 0, 40, 40), "identity": "Student A"}
    assert smoother.smooth([face])[0]["identity"] == "UNKNOWN"
    for _ in range(2):
        assert smoother.smooth([face])[0]["identity"] == "Student A"

    next_frame = [{**face, "identity": "Student B", "rec_conf": 0.95},
                  {**face, "box": (15, 0, 55, 40)}]
    first, second = smoother.smooth(next_frame)
    assert first["identity"] == "Student A"
    assert first["rec_conf"] is None
    assert second["identity"] == "UNKNOWN"
    assert first["track_id"] != second["track_id"]
    assert smoother.smooth(next_frame)[0]["identity"] == "UNKNOWN"
    assert smoother.smooth(next_frame)[0]["identity"] == "Student B"


@pytest.mark.parametrize("original_scores, corrected_scores, expected", [
    ([0.60, 0.30, 0.10], [0.82, 0.18, 0], "Student A"),
    ([0.60, 0.30, 0.10], [0.18, 0.82, 0], "UNKNOWN"),
    ([0.60, 0.30, 0.10], [0.68, 0.32, 0], "UNKNOWN"),
    ([0.39, 0.31, 0.30], [0.82, 0.18, 0], "UNKNOWN"),
])
def test_webcam_retry_requires_a_supported_candidate(original_scores, corrected_scores, expected):
    classifier = SVMClassifier(confidence_threshold=0.625)
    classifier.classes_ = np.array(["Student A", "Student B", "Student C"])
    classifier.model = SimpleNamespace(predict_proba=lambda x: np.array([
        corrected_scores if x[0, 0] < 115 else original_scores
    ]))
    classifier.is_trained = True
    preprocessor = SimpleNamespace(process_and_flatten=lambda crop: np.array([crop[0, 0, 0]]))
    model = SimpleNamespace(n_components=1, pca=SimpleNamespace(project=lambda vector, k: vector))
    pipeline = RecognitionPipeline(preprocessor, model)
    pipeline.set_classifier(classifier, "svm")

    crop = np.full((112, 112, 3), 120, dtype=np.uint8)
    assert pipeline.recognize_face(crop)["identity"] == "UNKNOWN"
    assert pipeline.recognize_face(crop, retry_webcam=True)["identity"] == expected
    assert classifier.confidence_threshold == 0.625
