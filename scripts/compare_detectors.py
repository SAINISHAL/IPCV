import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.unified_detector import UnifiedFaceDetector
from src.data.image_loader import RobustImageLoader
from src.utils.config import dataset_sample
from ultralytics import YOLO

# 1. Old Detector (Baseline YuNet conf=0.50)
old_det = UnifiedFaceDetector(conf_thresh=0.50)

# 2. YOLOv8-Face Model
yolo_face = YOLO(str(PROJECT_ROOT / "models/yolo/yolov8n-face.pt"))

test_images = [dataset_sample(key) for key in ('medium', 'wide', 'comparison', 'mixed', 'angle')]

print("=" * 75)
print("COMPARISON: OLD DETECTOR vs YOLOv8-FACE ON REPRESENTATIVE CLASSROOM PHOTOS")
print("=" * 75)

for p in test_images:
    if not p.exists():
        continue
    img = RobustImageLoader.read_image_file(p)
    h, w = img.shape[:2]
    
    # Old detector
    old_faces = old_det.detect_faces(img)
    
    # YOLOv8-Face at imgsz=1280, conf=0.35
    yolo_res = yolo_face(img, conf=0.35, imgsz=1280, verbose=False)
    yolo_boxes = []
    for r in yolo_res:
        for b in r.boxes:
            c = b.xyxy[0].cpu().numpy().astype(int)
            conf = float(b.conf[0].cpu().numpy())
            yolo_boxes.append((c, conf))

    print(f"\nImage: {p.name} ({p.parent.parent.name}) | Resolution: {w}x{h}")
    print(f"  Old Detector (YuNet conf=0.50):    {len(old_faces)} faces detected")
    print(f"  New YOLOv8-Face (conf=0.35, 1280): {len(yolo_boxes)} faces detected")
    
    # Print face box sizes to inspect small faces detection
    small_old = sum(1 for f in old_faces if (f['box'][2]-f['box'][0]) < 50)
    small_yolo = sum(1 for (b, c) in yolo_boxes if (b[2]-b[0]) < 50)
    print(f"  Small Faces (<50px width): Old = {small_old} | YOLOv8-Face = {small_yolo}")

