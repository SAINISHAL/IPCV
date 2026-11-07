import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.unified_detector import UnifiedFaceDetector
from src.recognition.prediction import RecognitionPipeline
from src.webcam.video_processor import VideoProcessor
from src.data.image_loader import RobustImageLoader
from src.utils.config import dataset_sample

det = UnifiedFaceDetector(conf_thresh=0.50)
pipe = RecognitionPipeline.load_pipeline(
    PROJECT_ROOT / "models" / "eigenfaces" / "pca_model.pkl",
    PROJECT_ROOT / "models" / "classifiers" / "svm_classifier.pkl",
    "svm"
)
proc = VideoProcessor(det, pipe, False)

img = RobustImageLoader.read_image_file(dataset_sample("medium"))
_, meta = proc.process_frame(img, False)

for i, f in enumerate(meta["faces"]):
    if f["identity"] == "UNKNOWN":
        x1, y1, x2, y2 = f["box"]
        pad = int(max(x2-x1, y2-y1) * 0.4)
        h, w = img.shape[:2]
        crop_single = img[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
        annotated_single, single_meta = proc.process_frame(crop_single, False)
        preds = [s["identity"] for s in single_meta["faces"]]
        print(f"Face {i} (UNKNOWN): detected {len(preds)} -> {preds}")
        if "UNKNOWN" in preds:
            cv2.imwrite(str(PROJECT_ROOT / "reports/test_outputs/sample_unknown_portrait.jpg"), crop_single)
            print("Saved sample unknown portrait!")
            break

