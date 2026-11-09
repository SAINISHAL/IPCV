"""Unified Face Detection Engine for High-Accuracy Classroom & Webcam Recognition.

Implements face-specific pretrained models (YOLOv8-Face and YuNet Deep Face Detector),
aspect-ratio preserving letterbox preprocessing, 5-point facial landmark alignment,
adaptive NMS, and comprehensive debugging diagnostics.
"""
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import time
import cv2
import numpy as np

from src.utils.logger import setup_logger
from src.detection.face_aligner import FaceAligner, align_five_point_crop

logger = setup_logger("UnifiedFaceDetector")
YUNET_IMAGE_LIMIT = 1600
YUNET_GRID_SIZE = 32


class UnifiedFaceDetector:
    """Accurate, multi-scale face detector supporting YOLOv8-Face, YuNet, and Ensembles."""

    def __init__(self,
                 engine: str = "yolo_face",
                 yolo_face_path: Optional[str or Path] = None,
                 yunet_model_path: Optional[str or Path] = None,
                 conf_thresh: float = 0.35,
                 nms_thresh: float = 0.40,
                 align_faces: bool = True):
        self.engine = engine.lower()
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.align_faces = align_faces
        self.project_root = Path(__file__).resolve().parent.parent.parent

        self.aligner = FaceAligner(desired_face_size=(100, 100)) if align_faces else None

        # 1. Initialize YOLOv8-Face Model
        self.yolo_face = None
        if yolo_face_path is None:
            yolo_face_path = self.project_root / "models" / "yolo" / "yolov8n-face.pt"
        self.yolo_path = Path(yolo_face_path)

        if self.yolo_path.exists():
            try:
                from ultralytics import YOLO
                self.yolo_face = YOLO(str(self.yolo_path))
                logger.info(f"Loaded specialized YOLOv8-Face detector from: {self.yolo_path}")
            except Exception as e:
                logger.warning(f"Could not load YOLOv8-Face model: {e}")

        # 2. Initialize YuNet ONNX Model
        self.yunet_detector = None
        if yunet_model_path is None:
            yunet_model_path = self.project_root / "models" / "pretrained" / "yunet.onnx"
        self.yunet_path = Path(yunet_model_path)

        if self.yunet_path.exists():
            try:
                self.yunet_detector = cv2.FaceDetectorYN.create(
                    str(self.yunet_path), "", (320, 320),
                    score_threshold=self.conf_thresh,
                    nms_threshold=self.nms_thresh,
                    top_k=500
                )
                logger.info(f"Loaded YuNet face detector from: {self.yunet_path}")
            except Exception as e:
                logger.warning(f"Failed loading YuNet detector: {e}")

        # Fallback Haar Cascade
        self.haar_cascade = None
        if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            try:
                cascade_p = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.haar_cascade = cv2.CascadeClassifier(cascade_p)
            except Exception as e:
                logger.warning(f"Could not initialize Haar cascade: {e}")

    @staticmethod
    def letterbox_image(image: np.ndarray, target_size: int = 1280) -> Tuple[np.ndarray, float, int, int]:
        """Resize image preserving aspect ratio with symmetric zero padding."""
        h, w = image.shape[:2]
        scale = min(target_size / h, target_size / w)
        nw, nh = int(round(w * scale)), int(round(h * scale))

        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dw = (target_size - nw) // 2
        dh = (target_size - nh) // 2

        padded = cv2.copyMakeBorder(
            resized, dh, target_size - nh - dh, dw, target_size - nw - dw,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return padded, scale, dw, dh

    def detect_faces(self, image: np.ndarray,
                      conf_thresh: Optional[float] = None,
                      min_face_size: Tuple[int, int] = (15, 15),
                      margin_ratio: float = 0.12,
                      debug: bool = False,
                      alignment: str = "eyes") -> Any:
        """Detect faces in an image with optional diagnostic debugging.
        
        Args:
            image: BGR image matrix
            conf_thresh: Override confidence threshold
            min_face_size: (min_w, min_h) in original pixels
            margin_ratio: Margin to expand around detected bounding box
            debug: If True, returns (detections, debug_info_dict)
            alignment: Saved models use "sface" to refer to the 112x112 five-point crop geometry.
            
        Returns:
            List of detection dicts (or tuple of (list, debug_info) if debug=True):
              - 'box': (x1, y1, x2, y2) in original image space
              - 'confidence': detection score float [0, 1]
              - 'landmarks': 5 facial landmark points if available
              - 'aligned_crop': training-compatible face crop, or None if it cannot be aligned
        """
        if image is None or image.size == 0:
            return ([], {}) if debug else []

        t_start = time.time()
        conf = conf_thresh if conf_thresh is not None else self.conf_thresh
        h, w = image.shape[:2]

        raw_detections: List[Dict[str, Any]] = []

        # ---------------------------------------------------------------------
        # 1. Primary: YOLOv8-Face Inference
        # ---------------------------------------------------------------------
        if (self.engine in ["yolo_face", "ensemble"] or self.yunet_detector is None) and self.yolo_face is not None:
            try:
                # Dynamic inference size based on input resolution
                max_dim = max(h, w)
                imgsz = 1280 if max_dim >= 1000 else 640

                results = self.yolo_face(image, conf=conf, imgsz=imgsz, verbose=False)

                for res in results:
                    for b in res.boxes:
                        coords = b.xyxy[0].cpu().numpy().astype(int)
                        box_conf = float(b.conf[0].cpu().numpy())
                        if box_conf < conf:
                            continue

                        x1, y1, x2, y2 = coords
                        raw_detections.append({
                            "box": (int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))),
                            "confidence": box_conf,
                            "landmarks": None,
                            "source": "yolo_face"
                        })
            except Exception as e:
                logger.error(f"YOLOv8-Face inference error: {e}")

        # ---------------------------------------------------------------------
        # 2. Secondary / YuNet Inference (with letterboxing and landmarks)
        # ---------------------------------------------------------------------
        if (self.engine in ["yunet", "ensemble"] or len(raw_detections) == 0) and self.yunet_detector is not None:
            try:
                self.yunet_detector.setScoreThreshold(conf)
                self.yunet_detector.setNMSThreshold(self.nms_thresh)

                # Letterbox image for optimal YuNet receptive field if very large
                if max(h, w) > YUNET_IMAGE_LIMIT:
                    target_sz = YUNET_IMAGE_LIMIT
                    padded_img, scale, dw, dh = self.letterbox_image(image, target_size=target_sz)
                    self.yunet_detector.setInputSize((target_sz, target_sz))
                    _, faces = self.yunet_detector.detect(padded_img)

                    if faces is not None:
                        for f in faces:
                            bx, by, bw, bh = f[:4]
                            score = float(f[-1])
                            if score < conf:
                                continue

                            # Map back from letterbox coordinates to original image coordinates
                            orig_bx = int(round((bx - dw) / scale))
                            orig_by = int(round((by - dh) / scale))
                            orig_x2 = int(round((bx + bw - dw) / scale))
                            orig_y2 = int(round((by + bh - dh) / scale))

                            # Landmarks scaling
                            landmarks = None
                            if len(f) >= 14:
                                lm = f[4:14].reshape((5, 2))
                                orig_lm = np.zeros_like(lm)
                                orig_lm[:, 0] = (lm[:, 0] - dw) / scale
                                orig_lm[:, 1] = (lm[:, 1] - dh) / scale
                                landmarks = orig_lm

                            raw_detections.append({
                                "box": (max(0, orig_bx), max(0, orig_by), min(w, orig_x2), min(h, orig_y2)),
                                "confidence": score,
                                "landmarks": landmarks,
                                "source": "yunet"
                            })
                else:
                    self.yunet_detector.setInputSize((w, h))
                    _, faces = self.yunet_detector.detect(image)
                    if faces is not None:
                        for f in faces:
                            bx, by, bw, bh = f[:4].astype(int)
                            score = float(f[-1])
                            if score < conf:
                                continue

                            landmarks = f[4:14].reshape((5, 2)) if len(f) >= 14 else None
                            raw_detections.append({
                                "box": (max(0, bx), max(0, by), min(w, bx + bw), min(h, by + bh)),
                                "confidence": score,
                                "landmarks": landmarks,
                                "source": "yunet"
                            })
            except Exception as e:
                logger.error(f"YuNet inference error: {e}")

        # ---------------------------------------------------------------------
        # 3. Final Fallback: Haar Cascade (if all deep detectors failed)
        # ---------------------------------------------------------------------
        if len(raw_detections) == 0 and self.haar_cascade is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=min_face_size)
            for (x, y, fw, fh) in faces:
                raw_detections.append({
                    "box": (x, y, min(w, x + fw), min(h, y + fh)),
                    "confidence": 0.70,
                    "landmarks": None,
                    "source": "haar"
                })

        # ---------------------------------------------------------------------
        # 4. Non-Maximum Suppression (Merge duplicate boxes across engines)
        # ---------------------------------------------------------------------
        filtered_detections = self.apply_nms(raw_detections, iou_thresh=self.nms_thresh)

        # ---------------------------------------------------------------------
        # 5. Cropping, Landmark Querying, and Alignment
        # ---------------------------------------------------------------------
        final_detections = []
        for det in filtered_detections:
            x1, y1, x2, y2 = det["box"]
            bw = x2 - x1
            bh = y2 - y1

            # Size filtering
            if bw < min_face_size[0] or bh < min_face_size[1]:
                continue

            # Query YuNet landmarks if missing
            landmarks = det.get("landmarks")
            if landmarks is None and self.yunet_detector is not None:
                landmarks = self.query_face_landmarks(image, (x1, y1, x2, y2))

            # Expand box with safe margin
            dx = int(bw * margin_ratio)
            dy = int(bh * margin_ratio)
            exp_box = (max(0, x1 - dx), max(0, y1 - dy), min(w, x2 + dx), min(h, y2 + dy))

            # Aligned face crop for Eigenfaces
            aligned_crop = None
            if alignment == "sface":
                aligned_crop = align_five_point_crop(image, landmarks)
            elif self.aligner is not None:
                aligned_crop = self.aligner.align_face(image, (x1, y1, x2, y2), landmarks=landmarks)
            else:
                crop = image[exp_box[1]:exp_box[3], exp_box[0]:exp_box[2]]
                aligned_crop = cv2.resize(crop, (100, 100)) if crop.size > 0 else None

            final_detections.append({
                "box": exp_box,
                "raw_box": (x1, y1, x2, y2),
                "confidence": det["confidence"],
                "landmarks": landmarks,
                "aligned_crop": aligned_crop,
                "source": det.get("source", "detector")
            })

        dt_ms = (time.time() - t_start) * 1000.0

        if debug:
            debug_info = {
                "original_resolution": (w, h),
                "faces_detected": len(final_detections),
                "processing_time_ms": round(dt_ms, 2),
                "engine_used": self.engine,
                "confidence_threshold": conf,
                "nms_threshold": self.nms_thresh,
                "raw_detections_count": len(raw_detections)
            }
            return final_detections, debug_info

        return final_detections

    def align_photo_face(self, image: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Align a large photo's face using landmarks at the original pixel resolution."""
        if self.yunet_detector is None or max(image.shape[:2]) <= YUNET_IMAGE_LIMIT:
            return None
        landmarks = self.query_face_landmarks(image, box, preserve_grid=True)
        return align_five_point_crop(image, landmarks)

    def query_face_landmarks(self, image: np.ndarray, box: Tuple[int, int, int, int], *,
                             preserve_grid: bool = False) -> Optional[np.ndarray]:
        """Crop face region with margin and run fast YuNet landmark detection."""
        try:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = box
            bw = x2 - x1
            bh = y2 - y1
            pad = int(max(bw, bh) * 0.3)

            if preserve_grid:
                # Keep YuNet's sampling grid aligned to the original image and leave context around the face.
                pad = max(3 * YUNET_GRID_SIZE, round(max(bw, bh) * 0.3))
            cx1, cy1, cx2, cy2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
            if preserve_grid:
                grid = YUNET_GRID_SIZE
                cx1, cy1 = cx1 // grid * grid, cy1 // grid * grid
                cx2, cy2 = (cx2 + grid - 1) // grid * grid, (cy2 + grid - 1) // grid * grid
            cx1, cy1, cx2, cy2 = max(0, cx1), max(0, cy1), min(w, cx2), min(h, cy2)
            if preserve_grid and max(cx2 - cx1, cy2 - cy1) > YUNET_IMAGE_LIMIT:
                return None
            crop = image[cy1:cy2, cx1:cx2]

            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                return None

            cw, ch = crop.shape[1], crop.shape[0]
            self.yunet_detector.setInputSize((cw, ch))
            self.yunet_detector.setScoreThreshold(0.20)
            _, faces = self.yunet_detector.detect(crop)

            if faces is not None and len(faces) > 0:
                # In a group photo the biggest face in the crop may be a neighbor.
                def box_iou(face):
                    fx, fy, fw, fh = face[:4]
                    fx += cx1
                    fy += cy1
                    overlap_w = max(0, min(x2, fx + fw) - max(x1, fx))
                    overlap_h = max(0, min(y2, fy + fh) - max(y1, fy))
                    intersection = overlap_w * overlap_h
                    return intersection / (bw * bh + fw * fh - intersection)

                best_face = max(faces, key=box_iou)
                min_overlap = 0.5 if preserve_grid else 0.3
                if len(best_face) >= 14 and box_iou(best_face) >= min_overlap:
                    lm = best_face[4:14].reshape((5, 2))
                    # Map back to full image
                    lm[:, 0] += cx1
                    lm[:, 1] += cy1
                    return lm
        except Exception:
            pass
        return None

    @staticmethod
    def apply_nms(detections: List[Dict[str, Any]], iou_thresh: float = 0.40) -> List[Dict[str, Any]]:
        """Non-Maximum Suppression to eliminate overlapping boxes."""
        if len(detections) <= 1:
            return detections

        boxes = np.array([d["box"] for d in detections], dtype=float)
        scores = np.array([d["confidence"] for d in detections], dtype=float)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]

        return [detections[idx] for idx in keep]
