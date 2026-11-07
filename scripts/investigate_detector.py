import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.unified_detector import UnifiedFaceDetector
from src.data.image_loader import RobustImageLoader
from src.utils.config import dataset_sample

detector = UnifiedFaceDetector(conf_thresh=0.50)

# Inspect several sample photos from data/raw
sample_paths = [dataset_sample(key) for key in ('medium', 'wide', 'comparison', 'mixed')]

for p in sample_paths:
    if not p.exists():
        continue
    img = RobustImageLoader.read_image_file(p)
    h, w = img.shape[:2]
    dets = detector.detect_faces(img)
    print(f"\nImage: {p.name} | Res: {w}x{h}")
    print(f"Total detected with YuNet at (w, h)=({w}, {h}) conf=0.50: {len(dets)}")
    for i, d in enumerate(dets[:5]):
        b = d["box"]
        bw = b[2] - b[0]
        bh = b[3] - b[1]
        print(f"  Face {i}: box={b}, size={bw}x{bh}, conf={d['confidence']:.3f}")

    # What happens if we test different confidence thresholds (e.g. 0.35, 0.40, 0.50, 0.60)?
    for th in [0.25, 0.35, 0.50, 0.65]:
        detector.yunet_detector.setScoreThreshold(th)
        detector.yunet_detector.setInputSize((w, h))
        _, raw_faces = detector.yunet_detector.detect(img)
        count = len(raw_faces) if raw_faces is not None else 0
        print(f"  Threshold {th:.2f} -> {count} faces")

