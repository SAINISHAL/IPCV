"""Interactive Streamlit Presentation Dashboard for Classroom Face Recognition."""
import sys
from pathlib import Path
import json
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Setup project path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, load_dataset_profile
from src.data.identity_mapper import IdentityMapper
from src.eigenfaces.eigenfaces_model import EigenfacesModel
from src.detection.unified_detector import UnifiedFaceDetector
from src.preprocessing.image_preprocessing import ImagePreprocessor
from src.recognition.prediction import RecognitionPipeline
from src.recognition.svm_classifier import SVMClassifier
from src.recognition.knn_classifier import KNNClassifier
from src.recognition.euclidean_classifier import EuclideanClassifier
from src.webcam.video_processor import VideoProcessor

# Set page config
st.set_page_config(
    page_title="Classroom Face Recognition | YOLO + Eigenfaces",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_system_components():
    """Load configuration, models, and detection pipelines."""
    config = load_config()
    target_size = tuple(config["preprocessing"]["target_size"])
    label_path = PROJECT_ROOT / config["paths"]["label_file"]

    identity_mapper = IdentityMapper(label_path)

    # Models
    pca_path = PROJECT_ROOT / config["paths"]["pca_model_dir"] / "pca_model.pkl"
    pca_model = None
    if pca_path.exists():
        pca_model = EigenfacesModel(target_shape=target_size)
        pca_model.load(pca_path)

    classifiers = {}
    clf_dir = PROJECT_ROOT / config["paths"]["classifier_model_dir"]

    svm_p = clf_dir / "svm_classifier.pkl"
    if svm_p.exists():
        svm = SVMClassifier()
        svm.load(svm_p)
        classifiers["svm"] = svm

    knn_p = clf_dir / "knn_classifier.pkl"
    if knn_p.exists():
        knn = KNNClassifier()
        knn.load(knn_p)
        classifiers["knn"] = knn

    euc_p = clf_dir / "euclidean_classifier.pkl"
    if euc_p.exists():
        euc = EuclideanClassifier()
        euc.load(euc_p)
        classifiers["euclidean"] = euc

    # Unified Face Detector (YOLOv8-Face + YuNet Ensemble)
    yunet_path = PROJECT_ROOT / "models" / "pretrained" / "yunet.onnx"
    yolo_face_path = PROJECT_ROOT / "models" / "yolo" / "yolov8n-face.pt"
    detector = UnifiedFaceDetector(
        engine=config["detection"].get("engine", "ensemble"),
        yolo_face_path=yolo_face_path,
        yunet_model_path=yunet_path,
        conf_thresh=config["detection"].get("confidence_threshold", 0.35),
        align_faces=True
    )

    preprocessor = ImagePreprocessor(target_size=target_size, enable_clahe=True)

    return config, identity_mapper, pca_model, classifiers, detector, preprocessor


config, identity_mapper, pca_model, classifiers, detector, preprocessor = get_system_components()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["1. Project Overview",
     "2. Uploaded Image Recognition",
     "3. Detection Diagnostics & Debugging (Compulsory)",
     "4. Live Webcam Recognition",
     "5. Cluster Review & Identity Mapping",
     "6. Dataset & Image Access Audit",
     "7. Eigenfaces Visualization",
     "8. Model Evaluation & Benchmark"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**System Status**\n"
    f"- Detector: `YOLOv8-Face + YuNet ({detector.engine.upper()})`\n"
    f"- PCA Model: `{'Loaded (K=' + str(pca_model.n_components) + ')' if pca_model else 'Not Found'}`\n"
    f"- Classifiers: `{list(classifiers.keys())}`"
)

# -----------------------------------------------------------------------------
# PAGE 1: PROJECT OVERVIEW
# -----------------------------------------------------------------------------
if page == "1. Project Overview":
    st.markdown('<div class="main-header">🎓 Classroom Face Recognition System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Scalable Multi-Face Detection (YOLOv8-Face + YuNet) + Mathematical Subspace Recognition (Eigenfaces / PCA)</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Confirmed Identities", "12 Students", "identity_mapping.json")
    col2.metric("Discovered Faces", "605 Crops", "71 Raw Photos")
    col3.metric("PCA Subspace", f"K = {pca_model.n_components if pca_model else 20}", "NumPy Dual PCA")
    col4.metric("Top Model Accuracy", "67.74%", "SVM on Test Split")

    st.markdown("### 🏛️ System Architecture")
    st.markdown("""
    The classroom face recognition system decouples **spatial detection** from **subspace identity representation**:
    1. **Pretrained Face Detection Engine**: Specialized YOLOv8-Face and YuNet models detect all student faces simultaneously in high-resolution unconstrained classroom photos and live camera frames.
    2. **5-Point Affine Landmark Alignment**: Rotates faces horizontally based on eye pupils to avoid distortion when projected into PCA subspace.
    3. **Deterministic Preprocessing**: Standardized 100×100 resizing with CLAHE illumination equalization.
    4. **Mathematical Eigenfaces (NumPy Dual PCA)**: Projects the 10,000-dimensional face vectors into a compact 20-dimensional orthogonal subspace via $L = A^T A$.
    5. **Ensemble Identity Recognition**: Benchmarks Euclidean Distance, KNN, and SVM with calibrated unknown rejection.
    """)

# -----------------------------------------------------------------------------
# PAGE 2: UPLOADED IMAGE RECOGNITION
# -----------------------------------------------------------------------------
elif page == "2. Uploaded Image Recognition":
    st.markdown('<div class="main-header">🖼️ Uploaded Photograph Face Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload any classroom photograph or portrait to detect and identify multiple students simultaneously.</div>', unsafe_allow_html=True)

    c_clf, c_engine, c_conf = st.columns(3)
    with c_clf:
        selected_clf = st.selectbox("Classification Engine:", ["SVM", "Euclidean", "KNN"], index=0)
    with c_engine:
        selected_engine = st.selectbox("Detection Engine:", ["ensemble", "yolo_face", "yunet"], index=0)
    with c_conf:
        selected_conf = st.slider("Detection Confidence Threshold:", 0.20, 0.80, 0.35, 0.05)

    clf_obj = classifiers.get(selected_clf.lower())

    if clf_obj is None or pca_model is None:
        st.error("Trained models not loaded. Please run `python scripts/train_from_clusters.py` first.")
    else:
        # Reconfigure detector engine
        detector.engine = selected_engine.lower()
        detector.conf_thresh = selected_conf

        pipeline = RecognitionPipeline(
            preprocessor=preprocessor,
            eigenfaces_model=pca_model,
            classifier_type=selected_clf.lower(),
            k_components=pca_model.n_components
        )
        pipeline.set_classifier(clf_obj, selected_clf.lower())
        processor = VideoProcessor(detector=detector, recognition_pipeline=pipeline, enable_smoothing=False)

        uploaded_file = st.file_uploader(
            "Choose a classroom photograph (JPG, JPEG, PNG, BMP, WEBP)...",
            type=["jpg", "jpeg", "png", "bmp", "webp"]
        )

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if raw_img is None:
                st.error("Failed to decode uploaded image file. Please upload a valid image.")
            else:
                with st.spinner("Detecting and recognizing faces..."):
                    annotated_img, meta = processor.process_frame(raw_img, is_video_stream=False)

                st.markdown("---")
                col_orig, col_pred = st.columns(2)

                with col_orig:
                    st.subheader("📷 Original Uploaded Photograph")
                    rgb_orig = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                    st.image(rgb_orig, use_container_width=True)

                with col_pred:
                    st.subheader("🎯 Annotated Prediction with Name Labels")
                    rgb_pred = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    st.image(rgb_pred, use_container_width=True)

                st.markdown("---")
                st.markdown("### 📊 Recognition Summary & Statistics")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Faces Detected", meta["detected"])
                m2.metric("Enrolled Faces Recognized", meta["recognized"])
                m3.metric("Unknown Faces Count", meta["unknown"])
                m4.metric("Classifier Used", selected_clf)

                if meta["detected"] == 0:
                    st.warning("⚠️ No faces detected in this photograph. Try lowering the confidence threshold.")
                else:
                    st.markdown("#### Detected Faces Detail:")
                    df_data = []
                    for i, f in enumerate(meta["faces"]):
                        df_data.append({
                            "Face #": i + 1,
                            "Predicted Identity": f["identity"],
                            "Detection Conf": f"{f['det_conf']:.2f}",
                            "Recognition Score / Dist": f"{f['rec_conf']:.2f}" if f.get("rec_conf") is not None else f"{f.get('distance', 0):.2f}",
                            "Status": "✅ Recognized" if f["identity"] != "UNKNOWN" else "❓ UNKNOWN"
                        })
                    st.table(pd.DataFrame(df_data))

                    # GALLERY OF DETECTED FACE CROPS (Compulsory Section 5)
                    st.markdown("#### 🖼️ Gallery of Detected Face Crops:")
                    cols_gallery = st.columns(min(6, max(1, len(meta["faces"]))))
                    for i, f in enumerate(meta["faces"]):
                        crop_img = f.get("crop")
                        if crop_img is not None and crop_img.size > 0:
                            col_idx = i % len(cols_gallery)
                            with cols_gallery[col_idx]:
                                rgb_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                                caption = f"#{i+1}: {f['identity']}"
                                st.image(rgb_crop, caption=caption, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 3: DETECTION DIAGNOSTICS & DEBUGGING (COMPULSORY)
# -----------------------------------------------------------------------------
elif page == "3. Detection Diagnostics & Debugging (Compulsory)":
    st.markdown('<div class="main-header">🔬 Face Detection Diagnostics & Debugging</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect raw detector inputs, aspect-ratio letterboxing, bounding box proposals, and execution latency.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        dbg_engine = st.selectbox("Select Detector Engine:", ["ensemble", "yolo_face", "yunet"], index=0, key="dbg_eng")
    with c2:
        dbg_conf = st.slider("Detection Confidence:", 0.15, 0.90, 0.35, 0.05, key="dbg_cnf")
    with c3:
        dbg_nms = st.slider("NMS IoU Threshold:", 0.10, 0.70, 0.40, 0.05, key="dbg_nms")
    with c4:
        dbg_min_size = st.number_input("Minimum Face Size (px):", min_value=10, max_value=100, value=15, step=5)

    detector.engine = dbg_engine.lower()
    detector.conf_thresh = dbg_conf
    detector.nms_thresh = dbg_nms

    source_mode = st.radio("Select Image Source:", ["Sample Classroom Photo from Dataset", "Upload Custom Photo"], horizontal=True)

    test_img = None
    img_name = "test_image"

    if source_mode == "Sample Classroom Photo from Dataset":
        sample_paths = load_dataset_profile().get("samples", {})
        sample_titles = {
            "medium": "Classroom Medium",
            "wide": "Classroom Wide with Back Row",
            "angle": "Classroom Angled/Side",
            "mixed": "Classroom Mixed Lighting",
            "high_res": "High Resolution",
        }
        samples = {title: PROJECT_ROOT / sample_paths[key] for key, title in sample_titles.items()
                   if key in sample_paths}
        if samples:
            choice = st.selectbox("Choose sample photograph:", list(samples))
            sample_path = samples[choice]
            if sample_path.is_file():
                test_img = cv2.imread(str(sample_path))
                img_name = sample_path.name
        else:
            st.info("Choose Upload Custom Photo or configure local samples in data/dataset_profile.json.")
    else:
        uploaded_dbg = st.file_uploader("Upload photograph to debug...", type=["jpg", "jpeg", "png", "bmp", "webp"], key="dbg_upload")
        if uploaded_dbg is not None:
            bytes_data = np.asarray(bytearray(uploaded_dbg.read()), dtype=np.uint8)
            test_img = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
            img_name = uploaded_dbg.name

    if test_img is not None:
        raw_h, raw_w = test_img.shape[:2]
        padded_img, scale, dw, dh = detector.letterbox_image(test_img, target_size=1280)

        # Run detection with diagnostic debug info
        faces, dbg_info = detector.detect_faces(test_img, conf_thresh=dbg_conf, min_face_size=(dbg_min_size, dbg_min_size), debug=True)

        st.markdown("---")
        st.markdown("### ⏱️ Diagnostic Runtime Performance")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Original Resolution", f"{raw_w} × {raw_h}")
        d2.metric("Faces Detected", dbg_info["faces_detected"])
        d3.metric("Processing Latency", f"{dbg_info['processing_time_ms']} ms")
        d4.metric("Detector Model", dbg_engine.upper())

        st.markdown("---")
        v1, v2 = st.columns(2)

        with v1:
            st.subheader("1. Original Image Passed to Pipeline")
            st.image(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with v2:
            st.subheader("2. Letterboxed Image Passed to Detector (1280×1280)")
            st.image(cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Visual bounding boxes overlay
        overlay = test_img.copy()
        for i, f in enumerate(faces):
            x1, y1, x2, y2 = f["box"]
            conf = f["confidence"]
            src = f.get("source", "det")

            color = (0, 255, 0) if src == "yolo_face" else (255, 100, 0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            caption = f"#{i+1} ({conf:.2f}) [{src}]"
            cv2.putText(overlay, caption, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        st.markdown("---")
        st.subheader(f"3. Detected Bounding Boxes on Full Resolution Image ({len(faces)} faces)")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("---")
        st.subheader("4. Cropped and 5-Point Affine Aligned Face Gallery")
        if len(faces) > 0:
            crop_cols = st.columns(min(6, max(1, len(faces))))
            for i, f in enumerate(faces):
                with crop_cols[i % len(crop_cols)]:
                    c_img = f.get("aligned_crop")
                    if c_img is not None and c_img.size > 0:
                        st.image(cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB), caption=f"Face #{i+1} (conf={f['confidence']:.2f})", use_container_width=True)

        # Save debug case button
        st.markdown("---")
        if st.button("💾 Save Diagnostic Output for Inspection"):
            debug_dir = PROJECT_ROOT / "reports" / "detection_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            out_img_path = debug_dir / f"debug_{img_name}_{timestamp}.jpg"
            out_json_path = debug_dir / f"debug_{img_name}_{timestamp}.json"
            cv2.imwrite(str(out_img_path), overlay)
            with open(out_json_path, "w") as f:
                json.dump({
                    "image_name": img_name,
                    "resolution": (raw_w, raw_h),
                    "debug_info": dbg_info,
                    "boxes": [f["box"] for f in faces]
                }, f, indent=2)
            st.success(f"Diagnostic artifacts saved to: `{out_img_path.relative_to(PROJECT_ROOT)}`")

# -----------------------------------------------------------------------------
# PAGE 4: LIVE WEBCAM RECOGNITION
# -----------------------------------------------------------------------------
elif page == "4. Live Webcam Recognition":
    st.markdown('<div class="main-header">📹 Live Webcam Face Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Continuous real-time multi-face detection with predictions displayed directly above faces.</div>', unsafe_allow_html=True)

    selected_clf = st.selectbox("Select Classification Engine for Webcam:", ["SVM", "Euclidean", "KNN"], index=0, key="cam_clf")
    clf_obj = classifiers.get(selected_clf.lower())

    pipeline = RecognitionPipeline(
        preprocessor=preprocessor,
        eigenfaces_model=pca_model,
        classifier_type=selected_clf.lower(),
        k_components=pca_model.n_components
    )
    pipeline.set_classifier(clf_obj, selected_clf.lower())
    processor = VideoProcessor(detector=detector, recognition_pipeline=pipeline, enable_smoothing=True)

    st.info("💡 For optimal 30+ FPS low-latency performance on your laptop camera, launch the native OpenCV camera window:")
    if st.button("🚀 Launch Native OpenCV Live Webcam Window"):
        from src.webcam.webcam_recognition import WebcamRecognitionApp
        app = WebcamRecognitionApp()
        app.initialize_models()
        app.run(device_id=0)

    st.markdown("---")
    st.markdown("#### Or test using browser camera snapshot:")
    camera_img = st.camera_input("Take a snapshot with your webcam")
    if camera_img is not None:
        file_bytes = np.asarray(bytearray(camera_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated, meta = processor.process_frame(img, is_video_stream=False)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(rgb, caption=f"Webcam Result: {meta['detected']} face(s) found", use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 5: CLUSTER REVIEW & IDENTITY MAPPING
# -----------------------------------------------------------------------------
elif page == "5. Cluster Review & Identity Mapping":
    st.markdown('<div class="main-header">👥 Unsupervised Face Clusters & Identity Assignment</div>', unsafe_allow_html=True)

    clusters_json_path = PROJECT_ROOT / "data" / "clusters.json"
    mapping_json_path = PROJECT_ROOT / "data" / "identity_mapping.json"
    clusters_fig_dir = PROJECT_ROOT / "reports" / "clusters"

    if clusters_json_path.exists():
        overview_path = clusters_fig_dir / "all_clusters_overview.png"
        if overview_path.exists():
            st.image(str(overview_path), caption="Cluster Exemplars Overview", use_container_width=True)

        if mapping_json_path.exists():
            st.markdown("### 📋 Confirmed Identity Mapping (`identity_mapping.json`)")
            with open(mapping_json_path, "r", encoding="utf-8") as f:
                cur_map = json.load(f)
            df_map = pd.DataFrame([{"Cluster ID": f"Cluster {k}", "Assigned Identity": v} for k, v in cur_map.items()])
            st.table(df_map)

# -----------------------------------------------------------------------------
# PAGE 6: DATASET & IMAGE ACCESS AUDIT
# -----------------------------------------------------------------------------
elif page == "6. Dataset & Image Access Audit":
    st.markdown('<div class="main-header">📂 Image Access & Dataset Audit</div>', unsafe_allow_html=True)

    audit_md_path = PROJECT_ROOT / "reports" / "image_access_report.md"
    if audit_md_path.exists():
        with open(audit_md_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Run `python scripts/test_image_access.py` to generate image access audit.")

# -----------------------------------------------------------------------------
# PAGE 7: EIGENFACES VISUALIZATION
# -----------------------------------------------------------------------------
elif page == "7. Eigenfaces Visualization":
    st.markdown('<div class="main-header">🧠 Mathematical Eigenfaces (PCA) Subspace</div>', unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["Top Eigenfaces Basis", "Variance Curves", "Reconstruction Quality"])
    with t1:
        f6 = PROJECT_ROOT / "reports" / "figures" / "06_eigenfaces_basis.png"
        if f6.exists(): st.image(str(f6), use_container_width=True)
    with t2:
        f7 = PROJECT_ROOT / "reports" / "figures" / "07_pca_variance_curves.png"
        if f7.exists(): st.image(str(f7), use_container_width=True)
    with t3:
        f8 = PROJECT_ROOT / "reports" / "figures" / "08_face_reconstructions.png"
        if f8.exists(): st.image(str(f8), use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 8: MODEL EVALUATION & BENCHMARK
# -----------------------------------------------------------------------------
elif page == "8. Model Evaluation & Benchmark":
    st.markdown('<div class="main-header">📊 Model Benchmarking & Evaluation</div>', unsafe_allow_html=True)

    tab_recog, tab_detect = st.tabs(["Face Recognition Benchmark", "Face Detection Benchmark (NEW)"])

    with tab_recog:
        eval_md = PROJECT_ROOT / "reports" / "model_evaluation_report.md"
        if eval_md.exists():
            with open(eval_md, "r", encoding="utf-8") as f:
                st.markdown(f.read())

        st.markdown("### 🔲 Confusion Matrices across Classifiers")
        c1, c2, c3 = st.columns(3)
        f_euc = PROJECT_ROOT / "reports" / "figures" / "10_confusion_matrix_euclidean.png"
        f_knn = PROJECT_ROOT / "reports" / "figures" / "10_confusion_matrix_knn.png"
        f_svm = PROJECT_ROOT / "reports" / "figures" / "10_confusion_matrix_svm.png"
        if f_euc.exists(): c1.image(str(f_euc), caption="Euclidean Confusion Matrix", use_container_width=True)
        if f_knn.exists(): c2.image(str(f_knn), caption="KNN Confusion Matrix", use_container_width=True)
        if f_svm.exists(): c3.image(str(f_svm), caption="SVM Confusion Matrix", use_container_width=True)

    with tab_detect:
        det_md = PROJECT_ROOT / "reports" / "detection_accuracy_report.md"
        if det_md.exists():
            with open(det_md, "r", encoding="utf-8") as f:
                st.markdown(f.read())
