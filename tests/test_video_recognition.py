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
            {"box": (0, 0, 40, 40), "raw_box": (5, 5, 35, 35),
             "confidence": 0.9, "aligned_crop": aligned_face},
            {"box": (45, 0, 85, 40), "confidence": 0.9, "aligned_crop": None},
        ]

    def recognize_face(crop):
        assert crop is aligned_face
        return {"identity": "Student A", "confidence": 0.8}

    processor = VideoProcessor(
        detector=SimpleNamespace(detect_faces=detect_faces),
        recognition_pipeline=SimpleNamespace(
            face_alignment="sface",
            recognize_face=recognize_face,
        ),
        enable_smoothing=False,
    )
    _, meta = processor.process_frame(np.zeros((100, 100, 3), dtype=np.uint8), is_video_stream=False)
    assert [face["identity"] for face in meta["faces"]] == ["Student A", "UNKNOWN"]
    assert meta["faces"][0]["box"] == (5, 5, 35, 35)
    assert (meta["detected"], meta["recognized"], meta["unknown"]) == (2, 1, 1)
    assert captions == ["Student A (0.80)", "UNKNOWN - MOVE INTO FRAME"]


def test_large_photo_refines_faces_without_assigning_a_duplicate_name():
    boxes = [(20 + i * 60, 20, 60 + i * 60, 70) for i in range(4)]
    detected = [{"box": box, "confidence": 0.9, "aligned_crop": np.full((112, 112, 3), i, np.uint8)}
                for i, box in enumerate(boxes)]
    refined = [np.full((112, 112, 3), i, np.uint8) for i in range(4, 7)] + [None]
    names = ["Student A", "UNKNOWN", "UNKNOWN", "Student C", "Student A", "Student B", "Student A"]
    processor = VideoProcessor(
        detector=SimpleNamespace(detect_faces=lambda frame, **kwargs: detected,
                                 align_photo_face=lambda frame, box: refined[boxes.index(box)]),
        recognition_pipeline=SimpleNamespace(
            face_alignment="sface", classifier_type="svm",
            recognize_face=lambda crop: {"identity": names[crop[0, 0, 0]], "confidence": 0.9},
        ),
        enable_smoothing=False,
    )

    _, meta = processor.process_frame(np.zeros((1800, 600, 3), dtype=np.uint8), is_video_stream=False)

    assert [f["box"] for f in meta["faces"]] == boxes
    assert [f["identity"] for f in meta["faces"]] == ["Student A", "Student B", "UNKNOWN", "Student C"]
    assert meta["faces"][1]["crop"] is refined[1]
    assert meta["faces"][2]["crop"] is detected[2]["aligned_crop"]


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


def test_confirmed_name_survives_brief_uncertainty_but_expires():
    tracker = TemporalPredictionSmoother()
    face = {"box": (5, 5, 65, 65), "identity": "Student A"}
    rejected = dict(face, identity="UNKNOWN", rec_conf=0.51)
    tracker.smooth([face], timestamp=0.0)
    assert tracker.smooth([face], timestamp=0.2)[0]["identity"] == "Student A"
    for timestamp in (0.5, 0.8, 1.3):
        tracked = tracker.smooth([rejected], timestamp=timestamp)[0]
        assert tracked["identity"] == "Student A" and tracked["rec_conf"] is None
    assert tracker.smooth([rejected], timestamp=1.5)[0]["identity"] == "UNKNOWN"


def test_short_missed_detection_keeps_track_without_drawing_a_box_or_leaking_a_name():
    tracker = TemporalPredictionSmoother()
    face = {"box": (5, 5, 65, 65), "identity": "Student A"}
    tracker.smooth([face], timestamp=0.0)
    stable = tracker.smooth([face], timestamp=0.2)[0]
    assert tracker.smooth([], timestamp=0.4) == []
    returned = tracker.smooth([face], timestamp=0.6)[0]
    assert returned["identity"] == "Student A" and returned["track_id"] == stable["track_id"]
    tracker.smooth([], timestamp=0.8)
    different = tracker.smooth([dict(face, identity="Student B")], timestamp=1.0)[0]
    assert different["identity"] == "UNKNOWN"
    tracker.smooth([], timestamp=1.2)
    new_track = tracker.smooth([face], timestamp=1.8)[0]
    assert new_track["track_id"] != stable["track_id"] and new_track["identity"] == "UNKNOWN"


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
