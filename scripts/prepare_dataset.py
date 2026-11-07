"""Dataset Preparation Script: Extracts, filters, preprocesses, and stores face crops."""
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.archive_extractor import ArchiveExtractor
from src.data.annotation_parser import AnnotationParser
from src.data.identity_mapper import IdentityMapper
from src.detection.face_extractor import FaceExtractor
from src.preprocessing.image_preprocessing import ImagePreprocessor
from src.preprocessing.quality_filter import QualityFilter

logger = setup_logger("PrepareDataset")


def main():
    config = load_config()
    raw_dir = PROJECT_ROOT / config["paths"]["raw_data_dir"]
    extracted_dir = PROJECT_ROOT / config["paths"]["extracted_faces_dir"]
    processed_dir = PROJECT_ROOT / config["paths"]["processed_faces_dir"]
    label_path = PROJECT_ROOT / config["paths"]["label_file"]

    identity_mapper = IdentityMapper(label_path)
    face_extractor = FaceExtractor(extracted_dir, min_face_size=tuple(config["detection"]["min_face_size"]))
    preprocessor = ImagePreprocessor(target_size=tuple(config["preprocessing"]["target_size"]), enable_clahe=True)

    logger.info("Extracting and preprocessing face images...")
    student_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])

    total_extracted = 0
    total_processed = 0

    for sdir in student_dirs:
        archive_name = sdir.name
        is_verified = identity_mapper.is_archive_verified(archive_name)
        pairs = ArchiveExtractor.resolve_pairs(sdir)

        for img_idx, (img_path, txt_path) in enumerate(pairs):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            if txt_path and txt_path.exists():
                boxes = AnnotationParser.parse_file(txt_path, deduplicate=True)
                for box_idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.to_pixel_coords(w, h)
                    crop = face_extractor.extract_crop(img, (x1, y1, x2, y2))
                    if crop is None:
                        continue

                    # Filter blurry/bad crops
                    ok, _ = QualityFilter.is_acceptable_quality(crop, min_size=(24, 24), min_laplacian_var=12.0)
                    if not ok:
                        continue

                    # Determine student label
                    student_label = identity_mapper.map_class_to_student(box.class_id, archive_name=archive_name)

                    crop_fname = f"{archive_name}_img{img_idx}_box{box_idx}.jpg"
                    # Save raw extracted crop
                    face_extractor.save_crop(crop, student_label, crop_fname)
                    total_extracted += 1

                    # Preprocess and save
                    proc_img = preprocessor.process(crop)
                    # Convert float32 [0, 1] to uint8 [0, 255] for saving
                    proc_uint8 = (proc_img * 255.0).astype("uint8")
                    
                    target_proc_folder = processed_dir / student_label
                    target_proc_folder.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(target_proc_folder / crop_fname), proc_uint8)
                    total_processed += 1

    logger.info(f"Dataset preparation complete. Extracted: {total_extracted}, Processed: {total_processed}")


if __name__ == "__main__":
    main()

