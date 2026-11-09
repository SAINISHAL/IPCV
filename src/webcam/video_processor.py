"""Unified Video and Image Processor with Visual Label Overlay and Temporal Smoothing."""
import json
from time import monotonic
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter, defaultdict, deque
import cv2
import numpy as np

from src.recognition.prediction import RecognitionPipeline
from src.utils.logger import setup_logger

logger = setup_logger("VideoProcessor")


class TemporalPredictionSmoother:
    """Stabilizes identity predictions across consecutive video frames to prevent label flickering."""

    def __init__(self, window_size: int = 5, iou_match_threshold: float = 0.35,
                 unknown_grace_seconds: float = 1.25, missed_grace_seconds: float = 0.50):
        self.window_size = max(2, window_size)
        self.iou_match_threshold = iou_match_threshold
        self.track_history = defaultdict(lambda: deque(maxlen=self.window_size))
        self.track_boxes = {}
        self.track_identities = {}
        self.confirmed_at = {}
        self.missing_since = {}
        self.unknown_grace_seconds = unknown_grace_seconds
        self.missed_grace_seconds = missed_grace_seconds
        self.next_track_id = 0

    @staticmethod
    def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        return interArea / float(boxAArea + boxBArea - interArea)

    def smooth(self, detected_faces: List[Dict[str, Any]], *, timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        now = monotonic() if timestamp is None else timestamp
        new_track_boxes = {}
        smoothed_results = []

        overlaps = []
        for index, face in enumerate(detected_faces):
            for track_id, box in self.track_boxes.items():
                if track_id in self.missing_since and now - self.missing_since[track_id] > self.missed_grace_seconds:
                    continue
                iou = self.compute_iou(face["box"], box)
                if iou >= self.iou_match_threshold:
                    overlaps.append((iou, index, track_id))

        assignments = {}
        used_tracks = set()
        for _, index, track_id in sorted(overlaps, reverse=True):
            if index not in assignments and track_id not in used_tracks:
                assignments[index] = track_id
                used_tracks.add(track_id)

        for index, face in enumerate(detected_faces):
            raw_id = face["identity"]
            track_id = assignments.get(index)
            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1

            history = self.track_history[track_id]
            stable_id = self.track_identities.get(track_id, "UNKNOWN")
            was_missing = self.missing_since.pop(track_id, None) is not None
            if was_missing and (raw_id == "UNKNOWN" or raw_id != stable_id):
                # Require new evidence if the returning face doesn't confirm the previous name.
                history.clear()
                stable_id = "UNKNOWN"
                self.confirmed_at.pop(track_id, None)
            history.append(raw_id)
            votes = Counter(history)
            agrees_twice = len(history) >= 2 and history[-2] == raw_id
            if raw_id == "UNKNOWN":
                if now - self.confirmed_at.get(track_id, float("-inf")) > self.unknown_grace_seconds:
                    stable_id = "UNKNOWN"
            elif agrees_twice and votes[raw_id] * 2 > len(history):
                stable_id = raw_id
            elif raw_id != stable_id and (now - self.confirmed_at.get(track_id, float("-inf")) > self.unknown_grace_seconds
                                         or (len(history) >= 2 and history[-2] not in {stable_id, "UNKNOWN"})):
                # A competing name needs repeated agreement and cannot extend the old label.
                stable_id = "UNKNOWN"
            if stable_id != "UNKNOWN" and raw_id == stable_id:
                self.confirmed_at[track_id] = now
            self.track_identities[track_id] = stable_id
            new_track_boxes[track_id] = face["box"]

            smoothed_face = dict(face, identity=stable_id, track_id=track_id)
            if raw_id != stable_id:
                smoothed_face.update(rec_conf=None, distance=None)
            smoothed_results.append(smoothed_face)

        for track_id in self.track_boxes.keys() - new_track_boxes.keys():
            missing = self.missing_since.setdefault(track_id, now)
            if now - missing <= self.missed_grace_seconds:
                new_track_boxes[track_id] = self.track_boxes[track_id]
            else:
                del self.track_history[track_id]
                del self.track_identities[track_id]
                self.confirmed_at.pop(track_id, None)
                del self.missing_since[track_id]
        self.track_boxes = new_track_boxes
        return smoothed_results


class VideoProcessor:
    """Processes webcam frames and uploaded images, runs detection + Eigenfaces recognition, and renders labels."""

    def __init__(self, detector: Any,
                 recognition_pipeline: RecognitionPipeline,
                 enable_smoothing: bool = True,
                 smoothing_window: int = 5, diagnostics: bool = False):
        self.detector = detector
        self.pipeline = recognition_pipeline
        self.diagnostics = diagnostics
        self.smoother = TemporalPredictionSmoother(window_size=smoothing_window) if enable_smoothing else None
        
        self.frame_count = 0
        self.start_time = None
        self.fps = 0.0

    def _recognize_photo_faces(self, frame: np.ndarray, faces: list[dict]) -> list[dict]:
        original = [self.pipeline.recognize_face(face["crop"]) for face in faces]
        refine = getattr(self.detector, "align_photo_face", None)
        if refine is None or self.pipeline.face_alignment != "sface" or self.pipeline.classifier_type != "svm":
            return original

        crops = [refine(frame, face["box"]) for face in faces]
        candidates = [self.pipeline.recognize_face(crop) if crop is not None else prior
                      for crop, prior in zip(crops, original)]
        counts = Counter(candidate["identity"] for candidate in candidates)
        original_names = {prior["identity"] for prior in original}

        results = []
        for face, crop, prior, candidate in zip(faces, crops, original, candidates):
            name = candidate["identity"]
            # A refinement must not introduce a duplicate name into a group photo.
            if name != "UNKNOWN" and name != prior["identity"] and (counts[name] > 1 or name in original_names):
                results.append(prior)
                continue
            if crop is not None:
                face["crop"] = crop
            results.append(candidate)
        return results

    def draw_face_annotations(self, image: np.ndarray, faces: List[Dict[str, Any]]) -> np.ndarray:
        """Draw high-contrast, professional visual bounding boxes and name labels directly on image."""
        annotated = image.copy()
        img_h, img_w = annotated.shape[:2]

        for f in faces:
            x1, y1, x2, y2 = f["box"]
            identity = f["identity"]
            rec_conf = f.get("rec_conf")
            dist = f.get("distance")

            is_unknown = (identity == "UNKNOWN")
            
            # Colors: Vibrant Green for enrolled students, Red for UNKNOWN
            box_color = (0, 0, 240) if is_unknown else (0, 200, 0)
            badge_color = (0, 0, 180) if is_unknown else (0, 150, 0)

            # Draw outer rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            # Build label caption
            if is_unknown:
                caption = "UNKNOWN - MOVE INTO FRAME" if f.get("at_frame_edge") else "UNKNOWN"
            else:
                if rec_conf is not None:
                    caption = f"{identity} ({rec_conf:.2f})"
                elif dist is not None:
                    caption = f"{identity} [d={dist:.1f}]"
                else:
                    caption = identity

            # Calculate text size
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = max(0.45, min(0.75, (y2 - y1) / 160.0))
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(caption, font, font_scale, thickness)

            # Position badge above bounding box (or below if too close to top)
            if y1 - th - 10 > 5:
                badge_y1 = y1 - th - 10
                badge_y2 = y1
                text_y = y1 - 5
            elif y2 + th + 10 < img_h:
                badge_y1 = y2
                badge_y2 = y2 + th + 10
                text_y = y2 + th + 5
            else:
                badge_y1 = y1
                badge_y2 = y1 + th + 10
                text_y = y1 + th + 5

            badge_x1 = max(0, min(x1, img_w - tw - 10))
            badge_x2 = min(img_w, badge_x1 + tw + 10)

            # Background solid badge for 100% readability
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), badge_color, -1)
            # Outline around badge
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), box_color, 1, cv2.LINE_AA)
            # Bold white text
            cv2.putText(annotated, caption, (badge_x1 + 5, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return annotated

    def process_frame(self, frame: np.ndarray, is_video_stream: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a single camera frame or uploaded photograph consistently.
        
        Returns:
            Tuple of (annotated_image_with_labels, metadata_dict).
        """
        if frame is None or frame.size == 0:
            return frame, {"detected": 0, "recognized": 0, "unknown": 0, "fps": 0.0, "faces": []}

        # 1. Detect faces using Unified Detector
        alignment = self.pipeline.face_alignment
        if alignment == "sface":
            detections = self.detector.detect_faces(frame, alignment=alignment)
        else:
            detections = self.detector.detect_faces(frame)
        faces_recognized = []
        eligible = []

        # 2. Recognize each detected face crop using Eigenfaces
        frame_h, frame_w = frame.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            left, top, right, bottom = det.get("raw_box", det["box"])
            at_frame_edge = left <= 2 or top <= 2 or right >= frame_w - 2 or bottom >= frame_h - 2
            
            aligned_crop = det.get("aligned_crop")
            crop = aligned_crop if aligned_crop is not None else frame[y1:y2, x1:x2]
            if alignment == "box":
                crop = frame[top:bottom, left:right]
            reason = ("alignment unavailable" if alignment == "sface" and aligned_crop is None
                      else "face crop too small" if min(crop.shape[:2]) < 15 else None)
            face = {
                "box": (left, top, right, bottom),
                "det_conf": det["confidence"],
                "identity": "UNKNOWN",
                "rec_conf": 0.0,
                "distance": None,
                "crop": crop,
                "at_frame_edge": at_frame_edge,
                "source": det.get("source", "detector")
            }
            if self.diagnostics:
                face["diagnostic"] = {"crop_dimensions": list(crop.shape), "rejection_reason": reason}
            if reason is None:
                eligible.append(face)
            faces_recognized.append(face)

        try:
            results = (self.pipeline.recognize_faces([f["crop"] for f in eligible], retry_webcam=True)
                       if is_video_stream else self._recognize_photo_faces(frame, eligible))
            for face, result in zip(eligible, results, strict=True):
                face.update(identity=result["identity"], rec_conf=result["confidence"], distance=result.get("distance"))
                if self.diagnostics:
                    details = face["diagnostic"]
                    if self.pipeline.classifier_type == "svm":
                        details.update(self.pipeline.classifier.prediction_details(result["weights"]))
                    details.update(result.get("diagnostic", {}))
                    details.update(raw_prediction=result["identity"], confidence=result["confidence"],
                                   distance=result.get("distance"),
                                   rejection_reason=details.get("rejection_reason") or
                                   ("below cutoff" if result["identity"] == "UNKNOWN" else None))
        except Exception as exc:
            logger.error(f"Recognition pipeline error: {exc}")
            for face in eligible:
                if self.diagnostics:
                    face["diagnostic"]["rejection_reason"] = "recognition error"

        # 3. Apply smoothing ONLY for continuous video streams
        if is_video_stream and self.smoother is not None:
            faces_recognized = self.smoother.smooth(faces_recognized)

        if self.diagnostics:
            for face in faces_recognized:
                details = face["diagnostic"]
                details["displayed_identity"] = face["identity"]
                if face["identity"] == "UNKNOWN" and details["rejection_reason"] is None:
                    details["rejection_reason"] = "waiting for consistent frames"
                logger.info("Face prediction %s", json.dumps(details))

        # 4. Draw bounding boxes and labels directly on faces
        annotated = self.draw_face_annotations(frame, faces_recognized)

        # 5. Compute stats
        n_detected = len(detections)
        n_unknown = sum(1 for f in faces_recognized if f["identity"] == "UNKNOWN")
        predicted_names = [f["identity"] for f in faces_recognized if f["identity"] != "UNKNOWN"]
        n_recognized = len(predicted_names)

        # FPS calculation for live video
        if is_video_stream:
            if self.start_time is None:
                self.start_time = cv2.getTickCount()
            else:
                self.frame_count += 1
                elapsed = (cv2.getTickCount() - self.start_time) / cv2.getTickFrequency()
                if elapsed > 0.5:
                    self.fps = float(self.frame_count / elapsed)
                    self.frame_count = 0
                    self.start_time = cv2.getTickCount()

            hud_text = f"FPS: {self.fps:.1f} | Detected: {n_detected} | Known: {n_recognized} | Unknown: {n_unknown}"
            cv2.putText(annotated, hud_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, hud_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        meta = {
            "detected": n_detected,
            "recognized": n_recognized,
            "unknown": n_unknown,
            "predicted_names": predicted_names,
            "fps": self.fps if is_video_stream else None,
            "faces": faces_recognized
        }
        return annotated, meta
