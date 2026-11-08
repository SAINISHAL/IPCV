"""Store manually verified webcam captures locally in separate session folders."""
from pathlib import Path

import cv2

from src.data.photo_groups import fingerprint, near_duplicate


CAPTURE_PROMPTS = (
    "Face the camera directly with even light on your face.",
    "Turn slightly to your left. Keep both eyes visible.",
    "Turn slightly to your right. Keep both eyes visible.",
    "Change the room lighting. Keep your face clearly visible.",
    "Sit slightly farther from the camera.",
    "Move a little closer and use a different natural expression.",
)


def save_capture(root: Path, split: str, identity: str, session_id: str, index: int, frame, detector):
    """Validate a confirmed, single-face capture before writing it to its assigned split."""
    if split not in {"train", "test", "unknown"} or any(
            not part or Path(part).name != part or part in {".", ".."} for part in (identity, session_id)):
        raise ValueError("Invalid capture session.")
    if frame is None or frame.size == 0:
        raise ValueError("The camera image could not be read. Take another photo.")
    faces = detector.detect_faces(frame, alignment="sface")
    if len(faces) != 1 or faces[0]["aligned_crop"] is None:
        raise ValueError("Show only the person being enrolled. Keep other people and screens out of view.")
    x1, y1, x2, y2 = faces[0]["raw_box"]
    h, w = frame.shape[:2]
    if x1 <= 2 or y1 <= 2 or x2 >= w - 2 or y2 >= h - 2 or min(x2 - x1, y2 - y1) < 48:
        raise ValueError("Keep your full face in the frame and move a little closer.")
    gray = cv2.cvtColor(faces[0]["aligned_crop"], cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 30:
        raise ValueError("The face is too blurry. Hold still and take another photo.")
    candidate = fingerprint(frame)
    for prior in root.glob("*/*/*/*.jpg"):
        image = cv2.imread(str(prior))
        if image is not None and near_duplicate(candidate, fingerprint(image)):
            raise ValueError("This is too similar to a saved photo. Change pose, lighting, or distance.")
    target = root / split / identity / session_id / f"capture_{index:03}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("The capture could not be saved. Try again.")
    with target.open("xb") as stream:
        stream.write(encoded.tobytes())
    return target
