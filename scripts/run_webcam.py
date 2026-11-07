"""Executable script to launch real-time webcam face recognition."""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.webcam.webcam_recognition import WebcamRecognitionApp


def main():
    parser = argparse.ArgumentParser(description="Live Classroom Face Recognition via YOLO + Eigenfaces")
    parser.add_argument("--device", type=int, default=None, help="Camera device ID (default from config.yaml: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file fallback")
    args = parser.parse_args()

    app = WebcamRecognitionApp()
    app.initialize_models()
    app.run(device_id=args.device, video_source=args.video)


if __name__ == "__main__":
    main()

