"""Main Entry Point for the Classroom Face Recognition System."""
import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Classroom Face Recognition System (YOLO + Eigenfaces PCA)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--inspect", action="store_true", help="Run comprehensive dataset inspection & generate report")
    parser.add_argument("--prepare", action="store_true", help="Extract and preprocess face crops from verified archives")
    parser.add_argument("--train", action="store_true", help="Train PCA Eigenfaces and classifiers (Euclidean, KNN, SVM)")
    parser.add_argument("--evaluate", action="store_true", help="Benchmark trained models on unseen test split")
    parser.add_argument("--webcam", action="store_true", help="Launch real-time webcam face recognition")
    parser.add_argument("--device", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file fallback for webcam")
    parser.add_argument("--app", action="store_true", help="Launch interactive Streamlit dashboard")
    parser.add_argument("--all", action="store_true", help="Execute complete pipeline (inspect -> prepare -> train -> evaluate)")

    args = parser.parse_args()

    # If no flag passed, run full pipeline
    if not any([args.inspect, args.prepare, args.train, args.evaluate, args.webcam, args.app, args.all]):
        print("No specific mode specified. Running complete pipeline (--all)...")
        args.all = True

    if args.inspect or args.all:
        print("\n" + "=" * 60)
        print("PHASE 1 & 2: DATASET INSPECTION")
        print("=" * 60)
        from scripts.inspect_dataset import main as run_inspect
        run_inspect()

    if args.prepare or args.all:
        print("\n" + "=" * 60)
        print("PHASE 5 & 6: DATASET EXTRACTION & PREPROCESSING")
        print("=" * 60)
        from scripts.prepare_dataset import main as run_prepare
        run_prepare()

    if args.train or args.all:
        print("\n" + "=" * 60)
        print("PHASE 7, 8 & 9: MODEL TRAINING & VALIDATION TUNING")
        print("=" * 60)
        from scripts.train import main as run_train
        run_train()

    if args.evaluate or args.all:
        print("\n" + "=" * 60)
        print("PHASE 10 & 11: TEST-SET BENCHMARKING & EVALUATION")
        print("=" * 60)
        from scripts.evaluate import main as run_eval
        run_eval()

    if args.webcam:
        print("\n" + "=" * 60)
        print("PHASE 12: REAL-TIME WEBCAM RECOGNITION")
        print("=" * 60)
        from src.webcam.webcam_recognition import WebcamRecognitionApp
        app = WebcamRecognitionApp()
        app.initialize_models()
        app.run(device_id=args.device, video_source=args.video)

    if args.app:
        print("\n" + "=" * 60)
        print("LAUNCHING STREAMLIT DASHBOARD")
        print("=" * 60)
        app_path = PROJECT_ROOT / "app" / "app.py"
        subprocess.run(["streamlit", "run", str(app_path)], check=True)


if __name__ == "__main__":
    main()

