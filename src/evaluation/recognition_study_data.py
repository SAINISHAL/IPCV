"""Load and audit recognition samples without trusting YOLO identity labels."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from src.data.photo_groups import fingerprint, group_photos, near_duplicate
from src.webcam.video_processor import TemporalPredictionSmoother


@dataclass
class FaceSample:
    name: str
    source: str
    face_id: str
    aligned: np.ndarray
    unaligned: np.ndarray | None
    group: str = ""

    def crop(self, alignment):
        return self.aligned if alignment == "sface" else self.unaligned


def align_detections(detector, frame):
    return detector.detect_faces(frame, alignment="sface")


def sample_from_detection(name, source, face_id, frame, detection):
    left, top, right, bottom = detection["raw_box"]
    return FaceSample(name, source, face_id, detection["aligned_crop"], frame[top:bottom, left:right].copy())


def load_classroom(root: Path, detector):
    clusters = json.loads((root / "data/clusters.json").read_text(encoding="utf-8"))["clusters"]
    mapping = json.loads((root / "data/identity_mapping.json").read_text(encoding="utf-8"))
    by_photo = defaultdict(list)
    named_photos = set()
    for cluster, items in clusters.items():
        name = mapping.get(cluster, "").strip()
        if cluster == "-1":
            name = "UNKNOWN"
        elif name in {"", "UNKNOWN", "(Unassigned)", "REMOVE / FALSE DETECTION"}:
            continue
        for item in items:
            source = item["original_image"]
            by_photo[source].append((name, item))
            if name != "UNKNOWN":
                named_photos.add(source)

    rows = []
    signatures = {}
    flags = defaultdict(list)
    for source, annotations in sorted(by_photo.items()):
        frame = cv2.imread(str(root / source))
        if frame is None:
            raise FileNotFoundError(f"Could not read classroom source {source}")
        signatures[source] = fingerprint(frame)
        detections = align_detections(detector, frame)
        counts = Counter(name for name, _ in annotations if name != "UNKNOWN")
        used = set()
        for name, item in sorted(annotations, key=lambda row: row[0] == "UNKNOWN"):
            face_id = item["face_id"]
            if counts[name] > 1:
                flags["same_name_on_multiple_faces"].append((face_id, name, source))
                continue
            x, y, w, h = item["box"]
            box = (x, y, x + w, y + h)
            candidates = [(TemporalPredictionSmoother.compute_iou(box, d["raw_box"]), i)
                          for i, d in enumerate(detections) if i not in used and d["aligned_crop"] is not None]
            overlap, index = max(candidates, default=(0.0, -1))
            if overlap < 0.5:
                flags["unmatched_or_unaligned"].append((face_id, name, source))
                continue
            used.add(index)
            sample = sample_from_detection(name, source, face_id, frame, detections[index])
            gray = cv2.cvtColor(sample.aligned, cv2.COLOR_BGR2GRAY)
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur < 30:
                flags["blurry"].append((face_id, name, source))
            if blur < 12:
                flags["severely_blurry"].append((face_id, name, source))
            if min(sample.unaligned.shape[:2]) < 20 or float(np.mean(gray == 0)) > 0.30:
                flags["small_or_clipped_crop"].append((face_id, name, source))
            rows.append(sample)

    named = [row for row in rows if row.name != "UNKNOWN"]
    crop_fingerprints = [fingerprint(row.aligned) for row in named]
    links = []
    for i, row in enumerate(named):
        for j, other in enumerate(named[:i]):
            if row.source != other.source and near_duplicate(crop_fingerprints[i], crop_fingerprints[j], max_difference=3):
                links.append((row.source, other.source))
                if row.name != other.name:
                    flags["conflicting_duplicate_labels"].append((row.face_id, other.face_id, row.name, other.name))
    groups = group_photos(signatures, links)
    for row in rows:
        row.group = groups[row.source]
    order = sorted(named_photos)
    np.random.RandomState(42).shuffle(order)
    train_end = round(len(order) * 0.70)
    val_end = train_end + round(len(order) * 0.15)
    group_split = {}
    for split, paths in (("test", order[val_end:]), ("validation", order[train_end:val_end]), ("train", order[:train_end])):
        for path in paths:
            group_split[groups[path]] = split  # Never let a duplicate of an old training photo into test.
    parts = {split: [] for split in ("train", "validation", "test")}
    for row in rows:
        if row.group in group_split:
            parts[group_split[row.group]].append(row)
    return parts, flags, groups, signatures


def audit_identities(rows, recognizer, flags):
    """Flag visual conflicts for human review. This never changes a person's label."""
    labeled = [row for row in rows if row.name != "UNKNOWN"]
    vectors = np.stack([recognizer.feature(row.aligned).ravel() for row in labeled])
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -1)
    names = np.array([row.name for row in labeled])
    for i, row in enumerate(labeled):
        same = names == row.name
        nearest_other = int(np.argmax(np.where(same, -1, similarities[i])))
        if similarities[i, nearest_other] >= 0.55 and similarities[i, nearest_other] > similarities[i, same].max() + 0.05:
            flags["review_possible_wrong_identity"].append((row.face_id, row.name, labeled[nearest_other].name))
    fingerprints = [fingerprint(row.aligned) for row in labeled]
    for i, row in enumerate(labeled):
        for j in np.flatnonzero(similarities[i, :i] > 0.88):
            other = labeled[j]
            if near_duplicate(fingerprints[i], fingerprints[j]):
                flags["duplicate_face_crops"].append((row.face_id, other.face_id, row.name, other.name))


def select_study_rows(parts, flags):
    conflicts = {face_id for a, b, _, _ in flags["conflicting_duplicate_labels"] for face_id in (a, b)}
    quality_flags = {item[0] for key in ("severely_blurry", "small_or_clipped_crop") for item in flags[key]}
    selected = {}
    for split, rows in parts.items():
        unique = {}
        for row in rows:
            if row.face_id not in conflicts and (split != "train" or row.face_id not in quality_flags):
                unique.setdefault((row.group, row.name if row.name != "UNKNOWN" else row.face_id), row)
        selected[split] = list(unique.values())
    return selected


def load_camera_sessions(root, detector, split, occupied_fingerprints, allowed_names):
    rows = []
    directory = root / "data" / "enrollment_sessions" / split
    for path in sorted(directory.glob("*/*/*.jpg")):
        name = path.parent.parent.name if split != "unknown" else "UNKNOWN"
        if name != "UNKNOWN" and name not in allowed_names:
            raise ValueError("A webcam capture folder has a name outside the enrolled roster.")
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Unreadable webcam capture: {path.name}")
        signature = fingerprint(frame)
        if any(near_duplicate(signature, prior) for prior in occupied_fingerprints):
            continue
        detections = align_detections(detector, frame)
        if len(detections) == 1 and detections[0]["aligned_crop"] is not None:
            sample = sample_from_detection(name, str(path), path.name, frame, detections[0])
            sample.group = str(path.parent)
            rows.append(sample)
            occupied_fingerprints.append(signature)
    return rows
