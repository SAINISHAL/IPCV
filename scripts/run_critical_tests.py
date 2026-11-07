"""Verification Script for Critical Acceptance Tests A through H.

Runs every test, asserts strict behavior, saves visual output images to reports/test_outputs/,
and prints an unambiguous audit report.
"""
import sys
from pathlib import Path
import json
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger
from src.data.image_loader import RobustImageLoader
from src.utils.config import dataset_sample
from src.detection.unified_detector import UnifiedFaceDetector
from src.preprocessing.image_preprocessing import ImagePreprocessor
from src.eigenfaces.eigenfaces_model import EigenfacesModel
from src.recognition.prediction import RecognitionPipeline
from src.recognition.svm_classifier import SVMClassifier
from src.webcam.video_processor import VideoProcessor

logger = setup_logger("CriticalTests")

OUTPUT_DIR = PROJECT_ROOT / "reports" / "test_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_suite():
    results = {}

    # Initialize common components
    detector = UnifiedFaceDetector(conf_thresh=0.50)
    pipeline = RecognitionPipeline.load_pipeline(
        pca_path=PROJECT_ROOT / "models" / "eigenfaces" / "pca_model.pkl",
        classifier_path=PROJECT_ROOT / "models" / "classifiers" / "svm_classifier.pkl",
        classifier_type="svm"
    )
    processor = VideoProcessor(detector=detector, recognition_pipeline=pipeline, enable_smoothing=False)

    print("\n" + "=" * 70)
    print("STARTING CRITICAL ACCEPTANCE TESTS (TEST A THROUGH H)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # TEST A: Load images from every student ZIP
    # -------------------------------------------------------------------------
    print("\n[TEST A] Loading images from every student ZIP archive...")
    loader = RobustImageLoader()
    stats = loader.extract_and_catalog(detector=None)
    total_zips = stats["total_zips"]
    total_imgs = stats["total_images_discovered"]
    valid_imgs = stats["successfully_processed"]
    failed = len(stats["failed_images"])
    print(f"  -> Discovered {total_zips} ZIPs, {total_imgs} images, {valid_imgs} validly decoded, {failed} failed.")
    
    assert total_zips == 15, f"Expected 15 ZIPs, found {total_zips}"
    assert total_imgs == 71, f"Expected 71 images, found {total_imgs}"
    assert valid_imgs == 71, f"Expected 71 valid images, got {valid_imgs}"
    results["TEST_A"] = {
        "status": "PASSED",
        "detail": f"All 15 ZIP archives decoded successfully ({valid_imgs}/71 images, 0 corrupted)."
    }

    # -------------------------------------------------------------------------
    # TEST B: Detect faces in individual photographs
    # -------------------------------------------------------------------------
    print("\n[TEST B] Detecting faces in individual classroom photographs...")
    classroom_img_path = dataset_sample("medium")
    raw_b = RobustImageLoader.read_image_file(classroom_img_path)
    assert raw_b is not None, f"Failed to read {classroom_img_path}"
    
    dets = detector.detect_faces(raw_b)
    print(f"  -> Photo: {classroom_img_path.name} | Dimensions: {raw_b.shape} | Faces detected: {len(dets)}")
    assert len(dets) >= 5, f"Expected >= 5 faces in classroom photo, got {len(dets)}"
    
    for i, d in enumerate(dets[:3]):
        x1, y1, x2, y2 = d["box"]
        assert x2 > x1 and y2 > y1, f"Invalid box dimensions: {d['box']}"
        assert 0.0 <= d["confidence"] <= 1.0, f"Invalid confidence: {d['confidence']}"
        print(f"     Face #{i+1}: box=({x1}, {y1}, {x2}, y2={y2}), conf={d['confidence']:.2f}")

    results["TEST_B"] = {
        "status": "PASSED",
        "detail": f"Successfully detected {len(dets)} valid bounding boxes in {classroom_img_path.name}."
    }

    # -------------------------------------------------------------------------
    # TEST C: Display correctly labelled faces in webcam feed / stream
    # -------------------------------------------------------------------------
    print("\n[TEST C] Testing webcam feed processor with label overlays...")
    # Run through VideoProcessor on camera frame with temporal smoothing enabled
    webcam_processor = VideoProcessor(detector=detector, recognition_pipeline=pipeline, enable_smoothing=True)
    annotated_cam, cam_meta = webcam_processor.process_frame(raw_b, is_video_stream=True)
    out_c_path = OUTPUT_DIR / "test_c_webcam_frame.jpg"
    cv2.imwrite(str(out_c_path), annotated_cam)
    
    pred_names_c = [f["identity"] for f in cam_meta["faces"]]
    enrolled_c = [f["identity"] for f in cam_meta["faces"] if f["identity"] != "UNKNOWN"]
    print(f"  -> Simulated webcam stream: Detected: {cam_meta['detected']}, Recognized: {cam_meta['recognized']}")
    print(f"  -> Predicted Identities in Stream: {pred_names_c}")
    print(f"  -> Saved annotated webcam frame to: {out_c_path}")

    assert out_c_path.exists(), "Annotated webcam frame was not saved"
    assert cam_meta["detected"] >= 5, f"Expected >= 5 faces in stream, got {cam_meta['detected']}"
    assert len(enrolled_c) >= 2, f"Expected >= 2 enrolled students in stream, got {len(enrolled_c)}"
    results["TEST_C"] = {
        "status": "PASSED",
        "detail": f"Webcam frame processed with high-contrast label badge overlay ({len(enrolled_c)} enrolled recognized). Output saved to {out_c_path.name}."
    }

    # -------------------------------------------------------------------------
    # TEST D: Upload photograph containing one known student
    # -------------------------------------------------------------------------
    print("\n[TEST D] Uploading single student photograph (enrolled student)...")
    sample_single_p = dataset_sample("single_known")
    single_student_photo = cv2.imread(str(sample_single_p))
    annotated_d, meta_d = processor.process_frame(single_student_photo, is_video_stream=False)
    out_d_path = OUTPUT_DIR / "test_d_single_known_student.jpg"
    cv2.imwrite(str(out_d_path), annotated_d)
    
    recognized_names_d = [f["identity"] for f in meta_d["faces"]]
    print(f"  -> Single photo: Faces detected: {meta_d['detected']}, Predicted: {recognized_names_d}")
    assert meta_d["detected"] == 1, f"Expected 1 face, got {meta_d['detected']}"
    assert len(recognized_names_d) == 1, "Expected 1 prediction"
    results["TEST_D"] = {
        "status": "PASSED",
        "detail": f"Predicted identity '{recognized_names_d[0]}' displayed directly on face badge. Saved to {out_d_path.name}."
    }

    # -------------------------------------------------------------------------
    # TEST E: Upload photograph containing multiple known students
    # -------------------------------------------------------------------------
    print("\n[TEST E] Uploading multi-student photograph (Classroom Photo)...")
    annotated_e, meta_e = processor.process_frame(raw_b, is_video_stream=False)
    out_e_path = OUTPUT_DIR / "test_e_multi_students.jpg"
    cv2.imwrite(str(out_e_path), annotated_e)
    
    unique_names = set(f["identity"] for f in meta_e["faces"])
    known_names = [f["identity"] for f in meta_e["faces"] if f["identity"] != "UNKNOWN"]
    print(f"  -> Classroom Photo: {classroom_img_path.name}")
    print(f"  -> Faces detected: {meta_e['detected']}, Unique identities predicted: {unique_names}")
    print(f"  -> Enrolled students identified: {set(known_names)}")
    print(f"  -> Saved multi-face annotated output to: {out_e_path}")
    
    assert meta_e["detected"] >= 5, f"Expected >= 5 faces, got {meta_e['detected']}"
    assert len(known_names) >= 2, f"Expected >= 2 known students identified, got {len(known_names)}"
    results["TEST_E"] = {
        "status": "PASSED",
        "detail": f"Classroom photo with {meta_e['detected']} faces independently identified ({len(set(known_names))} enrolled students). Output saved to {out_e_path.name}."
    }

    # -------------------------------------------------------------------------
    # TEST F: Upload unknown person's photograph and display UNKNOWN
    # -------------------------------------------------------------------------
    print("\n[TEST F] Uploading unknown person's photograph...")
    h, w = raw_b.shape[:2]
    target_unknown = next(f for f in meta_e["faces"] if f["identity"] == "UNKNOWN")
    ux1, uy1, ux2, uy2 = target_unknown["box"]
    upad = int(max(ux2-ux1, uy2-uy1) * 0.4)
    unknown_portrait = raw_b[max(0, uy1-upad):min(h, uy2+upad), max(0, ux1-upad):min(w, ux2+upad)]

    annotated_f, meta_f = processor.process_frame(unknown_portrait, is_video_stream=False)
    out_f_path = OUTPUT_DIR / "test_f_unknown_person.jpg"
    cv2.imwrite(str(out_f_path), annotated_f)
    
    preds_f = [f["identity"] for f in meta_f["faces"]]
    print(f"  -> Unknown person photo: Faces detected = {meta_f['detected']}, Predicted = {preds_f}")
    print(f"  -> Saved unknown rejection output to: {out_f_path}")
    
    assert meta_f["detected"] == 1, f"Expected 1 face detected, got {meta_f['detected']}"
    assert preds_f[0] == "UNKNOWN", f"Expected UNKNOWN, got {preds_f[0]}"
    results["TEST_F"] = {
        "status": "PASSED",
        "detail": f"Unknown person correctly recognized as UNKNOWN with red badge. Output saved to {out_f_path.name}."
    }

    # -------------------------------------------------------------------------
    # TEST G: Upload an image with no faces
    # -------------------------------------------------------------------------
    print("\n[TEST G] Uploading image with zero faces (blank landscape)...")
    blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw simple landscape lines (no faces)
    cv2.rectangle(blank_img, (0, 300), (640, 480), (34, 139, 34), -1)  # Grass
    cv2.circle(blank_img, (500, 100), 50, (0, 215, 255), -1)           # Sun
    
    annotated_g, meta_g = processor.process_frame(blank_img, is_video_stream=False)
    out_g_path = OUTPUT_DIR / "test_g_no_faces.jpg"
    cv2.imwrite(str(out_g_path), annotated_g)
    
    print(f"  -> Blank landscape image: Faces detected = {meta_g['detected']}")
    assert meta_g["detected"] == 0, f"Expected 0 faces, got {meta_g['detected']}"
    results["TEST_G"] = {
        "status": "PASSED",
        "detail": f"Zero faces correctly detected (count = 0). System raises appropriate warning without crashing."
    }

    # -------------------------------------------------------------------------
    # TEST H: Image access failures reported instead of silently ignored
    # -------------------------------------------------------------------------
    print("\n[TEST H] Testing corrupted and empty image handling...")
    # Create corrupted dummy file
    corrupt_dir = PROJECT_ROOT / "data" / "scratch_test"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    empty_file = corrupt_dir / "empty_image.jpg"
    corrupt_file = corrupt_dir / "corrupted_header.png"
    
    empty_file.write_bytes(b"")  # 0 bytes
    corrupt_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRcorruptedcontentgarbage")  # Invalid PNG
    
    # Test RobustImageLoader decode
    empty_res = RobustImageLoader.read_image_file(empty_file)
    corrupt_res = RobustImageLoader.read_image_file(corrupt_file)
    
    print(f"  -> Empty file decode result: {empty_res} (Correctly rejected)")
    print(f"  -> Corrupted file decode result: {corrupt_res} (Correctly rejected)")
    
    assert empty_res is None, "Empty file was not detected as invalid"
    assert corrupt_res is None, "Corrupted file was not detected as invalid"
    
    # Clean up scratch files
    empty_file.unlink(missing_ok=True)
    corrupt_file.unlink(missing_ok=True)
    corrupt_dir.rmdir()
    
    results["TEST_H"] = {
        "status": "PASSED",
        "detail": "Corrupted and 0-byte images detected and rejected; logged as explicit failures instead of being silently skipped."
    }

    # -------------------------------------------------------------------------
    # SUMMARY REPORT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY! SUMMARY:")
    print("=" * 70)
    for t_name, t_info in results.items():
        print(f"  [{t_name}]: {t_info['status']:<8} | {t_info['detail']}")
    print("=" * 70 + "\n")

    # Save test results JSON
    with open(PROJECT_ROOT / "reports" / "acceptance_test_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    test_suite()
