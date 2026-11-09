"""Upload classroom photos and recognize enrolled students with Eigenfaces."""
import sys
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.unified_detector import UnifiedFaceDetector
from src.recognition.prediction import RecognitionPipeline
from src.utils.config import load_config
from src.webcam.video_processor import VideoProcessor


@st.cache_resource
def load_photo_recognition():
    config = load_config()
    pca_path = PROJECT_ROOT / config["paths"]["pca_model_dir"] / "pca_model.pkl"
    svm_path = PROJECT_ROOT / config["paths"]["classifier_model_dir"] / "svm_classifier.pkl"
    if not pca_path.is_file() or not svm_path.is_file():
        raise FileNotFoundError("The saved recognition model is missing. Run python scripts/train_from_clusters.py first.")

    pipeline = RecognitionPipeline.load_pipeline(pca_path, svm_path)
    detection = config["detection"]
    detector = UnifiedFaceDetector(
        engine=detection.get("engine", "ensemble"),
        conf_thresh=detection.get("confidence_threshold", 0.35),
        nms_thresh=detection.get("nms_threshold", 0.40),
    )
    if detector.yolo_face is None and detector.yunet_detector is None:
        raise RuntimeError("The pretrained face detector files are missing from the models folder.")
    if pipeline.face_alignment == "sface" and detector.yunet_detector is None:
        raise RuntimeError("The saved Eigenfaces model requires YuNet to align faces.")

    return VideoProcessor(detector, pipeline, enable_smoothing=False), Lock()


def main():
    st.set_page_config(page_title="Classroom Face Recognition", page_icon="🎓", layout="wide")
    st.title("Classroom Face Recognition")
    st.caption("Upload one or more photos to see who is in each one.")
    uploads = st.file_uploader(
        "Upload photos", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True,
    )
    if not uploads:
        return

    try:
        processor, lock = load_photo_recognition()
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        st.error(str(exc))
        return

    for index, upload in enumerate(uploads, start=1):
        st.divider()
        st.subheader(upload.name)
        file_bytes = np.frombuffer(upload.getvalue(), dtype=np.uint8)
        try:
            photo = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) if file_bytes.size else None
        except cv2.error:
            photo = None
        if photo is None:
            st.error("This image could not be opened. Try another JPG, PNG, BMP, or WEBP file.")
            continue

        try:
            with st.spinner(f"Checking photo {index} of {len(uploads)}…"), lock:
                annotated, result = processor.process_frame(photo, is_video_stream=False)
        except (ValueError, RuntimeError, cv2.error):
            st.error("This photo could not be processed. Try a different image.")
            continue

        st.image(annotated, channels="BGR", use_container_width=True)
        names = list(dict.fromkeys(result["predicted_names"]))
        if names:
            st.write("**Recognized** · " + ", ".join(names))
        elif not result["detected"]:
            st.info("No faces were detected in this photo.")
        else:
            st.info("Faces were found, but no enrolled names could be confirmed.")
        st.caption(f"{result['detected']} detected · {result['recognized']} recognized · {result['unknown']} unknown")


if __name__ == "__main__":
    main()
