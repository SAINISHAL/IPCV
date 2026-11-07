import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.detection.unified_detector import UnifiedFaceDetector
from src.recognition.prediction import RecognitionPipeline
from src.webcam.video_processor import VideoProcessor
from src.data.image_loader import RobustImageLoader

detector = UnifiedFaceDetector(conf_thresh=0.50)
pipeline = RecognitionPipeline.load_pipeline(
    Path('models/eigenfaces/pca_model.pkl'),
    Path('models/classifiers/svm_classifier.pkl'),
    'svm'
)
proc = VideoProcessor(detector=detector, recognition_pipeline=pipeline)

for p in list(Path('data/raw').rglob('*.*'))[:20]:
    if p.suffix.lower() in ['.jpg', '.jpeg']:
        im = RobustImageLoader.read_image_file(p)
        if im is not None:
            _, m = proc.process_frame(im, is_video_stream=False)
            knowns = [f['identity'] for f in m['faces'] if f['identity'] != 'UNKNOWN']
            if len(knowns) > 0:
                print(f"{p.name} ({p.parent.parent.name}): detected={m['detected']}, knowns={knowns}")
