"""Script: Automatic Face Detection, Embedding Extraction, and Unsupervised Clustering."""
import sys
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.clustering.face_detector_deep import DeepFaceDetector
from src.clustering.face_clusterer import FaceClusterer

logger = setup_logger("DetectAndCluster")


def create_contact_sheet(face_images: List[np.ndarray], title: str,
                         save_path: Path, max_faces: int = 36, cols: int = 6) -> None:
    """Create a contact-sheet image displaying faces belonging to a cluster."""
    n_faces = min(len(face_images), max_faces)
    if n_faces == 0:
        return

    rows = int(np.ceil(n_faces / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i in range(n_faces):
        ax = axes[i]
        rgb = cv2.cvtColor(face_images[i], cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(f"#{i+1}", fontsize=8)
        ax.axis("off")

    for j in range(n_faces, len(axes)):
        axes[j].axis("off")

    plt.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180)
    plt.close()


def main():
    config = load_config()
    raw_dir = PROJECT_ROOT / config["paths"]["raw_data_dir"]
    output_crops_dir = PROJECT_ROOT / "data" / "detected_faces"
    output_crops_dir.mkdir(parents=True, exist_ok=True)

    clusters_report_dir = PROJECT_ROOT / "reports" / "clusters"
    clusters_report_dir.mkdir(parents=True, exist_ok=True)

    yunet_path = PROJECT_ROOT / "models" / "pretrained" / "yunet.onnx"
    sface_path = PROJECT_ROOT / "models" / "pretrained" / "sface.onnx"

    logger.info("Initializing pretrained YuNet detector and SFace embedder...")
    detector = DeepFaceDetector(yunet_path, sface_path, conf_threshold=0.55)

    # 1. Detect all faces in all raw student images
    image_extensions = {".jpg", ".jpeg", ".png"}
    all_image_paths = sorted([
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    ])
    logger.info(f"Discovered {len(all_image_paths)} images across raw archives.")

    face_metadata = []
    face_counter = 0

    logger.info("Running automatic face detection and embedding extraction across all images...")
    for img_path in all_image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        rel_path = str(img_path.relative_to(PROJECT_ROOT))
        archive_name = img_path.parent.name
        detections = detector.detect_in_image(img)

        for det in detections:
            face_counter += 1
            face_id = f"face_{face_counter:04d}"
            crop_filename = f"{face_id}.jpg"
            crop_save_path = output_crops_dir / crop_filename

            # Save aligned face crop (112x112)
            cv2.imwrite(str(crop_save_path), det["aligned"])

            meta_item = {
                "face_id": face_id,
                "crop_path": str(crop_save_path.relative_to(PROJECT_ROOT)),
                "original_image": rel_path,
                "archive_source": archive_name,
                "box": det["box"],
                "confidence": round(det["confidence"], 3),
                "embedding": det["embedding"]
            }
            face_metadata.append(meta_item)

    logger.info(f"Detected and extracted {len(face_metadata)} total face crops across all photos.")

    # Save raw detection metadata
    raw_meta_path = PROJECT_ROOT / "data" / "detected_faces_metadata.json"
    with open(raw_meta_path, "w", encoding="utf-8") as f:
        # Save without raw embedding vectors for compact inspection
        json.dump([
            {k: v for k, v in item.items() if k != "embedding"}
            for item in face_metadata
        ], f, indent=2)

    # 2. Perform Unsupervised Clustering
    logger.info("Performing unsupervised clustering (cosine distance DBSCAN)...")
    clusterer = FaceClusterer(distance_threshold=0.38, min_cluster_size=4)
    clusters = clusterer.cluster_faces(face_metadata)
    summaries = clusterer.get_cluster_summaries()

    # Save clusters JSON
    clusters_json_path = PROJECT_ROOT / "data" / "clusters.json"
    clusterer.save_clusters(clusters_json_path)

    # 3. Generate Contact Sheets
    logger.info("Generating visual contact sheets for each cluster...")
    overview_exemplars = []
    overview_titles = []

    for summary in summaries:
        cid = summary["cluster_id"]
        items = clusters[cid]
        count = summary["count"]

        # Load crops
        face_crops = []
        for it in items:
            p = PROJECT_ROOT / it["crop_path"]
            if p.exists():
                c = cv2.imread(str(p))
                if c is not None:
                    face_crops.append(c)

        if not face_crops:
            continue

        if cid == -1:
            title = f"Unassigned / Uncertain / Noise Faces ({count} faces)"
            sheet_path = clusters_report_dir / "uncertain_faces.png"
        else:
            title = f"Cluster {cid} ({count} faces, avg sim: {summary['avg_cosine_similarity']})"
            sheet_path = clusters_report_dir / f"cluster_{cid:02d}.png"
            # Keep exemplar for overview sheet
            overview_exemplars.append(face_crops[0])
            overview_titles.append(f"Cluster {cid} (N={count})")

        create_contact_sheet(face_crops, title, sheet_path)

    # Create All-Clusters Overview Sheet
    if overview_exemplars:
        create_contact_sheet(overview_exemplars, "Overview: Discovered Person Clusters (Exemplar Faces)",
                             clusters_report_dir / "all_clusters_overview.png", cols=5)

    logger.info(f"Visual review sheets saved to: {clusters_report_dir}")
    print("\n" + "=" * 60)
    print("FACE CLUSTERING COMPLETE - AWAITING USER IDENTITY ASSIGNMENT")
    print("=" * 60)
    print(f"Total Detected Faces: {len(face_metadata)}")
    print(f"Total Discovered Clusters: {len([s for s in summaries if not s['is_noise']])}")
    print(f"Uncertain / Noise Faces: {len(clusters.get(-1, []))}")
    print(f"Visual Gallery Saved: {clusters_report_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

