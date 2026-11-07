import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.detection.unified_detector import UnifiedFaceDetector
from src.recognition.prediction import RecognitionPipeline
from src.webcam.video_processor import VideoProcessor
from src.data.image_loader import RobustImageLoader
from src.utils.config import dataset_sample

det = UnifiedFaceDetector(engine='yolo_face', conf_thresh=0.35)
pipe = RecognitionPipeline.load_pipeline(Path('models/eigenfaces/pca_model.pkl'), Path('models/classifiers/svm_classifier.pkl'), 'svm')
proc = VideoProcessor(det, pipe, False)

img = RobustImageLoader.read_image_file(dataset_sample("medium"))
_, meta = proc.process_frame(img, False)
for i, f in enumerate(meta['faces']):
    print(f"Face {i}: id={f['identity']} rec_conf={f.get('rec_conf')} det_conf={f.get('det_conf'):.2f}")
