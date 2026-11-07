"""Robust Image Loader & Archive Ingestion Module.

Handles Windows pathing, nested folders, spaces/special characters in filenames,
and supports JPG, JPEG, PNG, BMP, WEBP via safe binary buffer decoding.
"""
import hashlib
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("ImageLoader")


class RobustImageLoader:
    """Safely extracts, verifies, decodes, and indexes images from student archives."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, data_dir: Optional[Path] = None, output_raw_dir: Optional[Path] = None):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        
        # Resolve data_dir
        if data_dir is None:
            # Try D:/IpCv/Data or relative
            candidates = [
                self.project_root.parent / "Data",
                self.project_root / "data" / "student_archives",
                Path(r"D:\IpCv\Data")
            ]
            self.data_dir = next((c for c in candidates if c.exists()), self.project_root.parent / "Data")
        else:
            self.data_dir = Path(data_dir)

        if output_raw_dir is None:
            self.output_raw_dir = self.project_root / "data" / "raw"
        else:
            self.output_raw_dir = Path(output_raw_dir)
            
        self.output_raw_dir.mkdir(parents=True, exist_ok=True)
        
        self.report_stats: Dict[str, Any] = {
            "total_zips": 0,
            "total_images_discovered": 0,
            "successfully_processed": 0,
            "failed_images": [],
            "duplicate_images": {},
            "faces_detected": 0,
            "zip_details": []
        }

    @staticmethod
    def decode_image_bytes(raw_bytes: bytes) -> Optional[np.ndarray]:
        """Safely decode image bytes using OpenCV imdecode from buffer.
        
        Prevents Windows filepath encoding / special character issues.
        """
        if not raw_bytes or len(raw_bytes) == 0:
            return None
        try:
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None and img.size > 0 and img.shape[0] > 0 and img.shape[1] > 0:
                return img
            return None
        except Exception as e:
            logger.error(f"Image decode exception: {e}")
            return None

    @classmethod
    def read_image_file(cls, filepath: Path) -> Optional[np.ndarray]:
        """Read image file safely via binary stream, supporting spaces and Unicode in paths."""
        p = Path(filepath)
        if not p.exists() or not p.is_file():
            return None
        try:
            with open(p, "rb") as f:
                data = f.read()
            return cls.decode_image_bytes(data)
        except Exception as e:
            logger.error(f"Error reading image file {p}: {e}")
            return None

    def extract_and_catalog(self, detector: Optional[Any] = None) -> Dict[str, Any]:
        """Extract all student ZIP files, catalog every image, and optionally count detected faces."""
        logger.info(f"Scanning for ZIP archives in {self.data_dir}...")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found at: {self.data_dir}")

        zip_files = sorted(list(self.data_dir.glob("*.zip")))
        # Exclude GitHub code repository zips
        student_zips = [zf for zf in zip_files if "Real_Time" not in zf.name and "eigenfaces" not in zf.name]
        self.report_stats["total_zips"] = len(student_zips)
        logger.info(f"Discovered {len(student_zips)} student ZIP archives.")

        all_hashes = {}
        total_discovered = 0
        total_success = 0
        total_faces = 0

        for zf in student_zips:
            clean_name = zf.stem.split("-2026")[0].strip()
            student_raw_dir = self.output_raw_dir / clean_name
            student_raw_dir.mkdir(parents=True, exist_ok=True)

            zip_info = {
                "archive_name": zf.name,
                "student_name": clean_name,
                "images_found": 0,
                "images_valid": 0,
                "faces_in_zip": 0
            }

            try:
                with zipfile.ZipFile(zf, "r") as z:
                    for member in z.infolist():
                        # Protect against Zip Slip
                        target_path = (student_raw_dir / member.filename).resolve()
                        if not str(target_path).startswith(str(student_raw_dir.resolve())):
                            self.report_stats["failed_images"].append({
                                "file": member.filename,
                                "archive": zf.name,
                                "reason": "Zip slip path traversal detected"
                            })
                            continue

                        if member.is_dir():
                            target_path.mkdir(parents=True, exist_ok=True)
                            continue

                        ext = Path(member.filename).suffix.lower()
                        # Extract all files including annotations
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        raw_bytes = z.read(member.filename)
                        with open(target_path, "wb") as f_out:
                            f_out.write(raw_bytes)

                        # Check if image
                        if ext in self.SUPPORTED_EXTENSIONS:
                            total_discovered += 1
                            zip_info["images_found"] += 1

                            # Verify decoding
                            img = self.decode_image_bytes(raw_bytes)
                            if img is None:
                                self.report_stats["failed_images"].append({
                                    "file": member.filename,
                                    "archive": zf.name,
                                    "reason": "Corrupted or invalid image bytes (failed to decode)"
                                })
                            else:
                                total_success += 1
                                zip_info["images_valid"] += 1

                                # Check duplicate hash
                                h = hashlib.md5(raw_bytes).hexdigest()
                                if h in all_hashes:
                                    if h not in self.report_stats["duplicate_images"]:
                                        self.report_stats["duplicate_images"][h] = [all_hashes[h]]
                                    self.report_stats["duplicate_images"][h].append(f"{zf.name}::{member.filename}")
                                else:
                                    all_hashes[h] = f"{zf.name}::{member.filename}"

                                # Count detected faces if detector is provided
                                if detector is not None:
                                    try:
                                        dets = detector.detect_faces(img)
                                        zip_info["faces_in_zip"] += len(dets)
                                        total_faces += len(dets)
                                    except Exception:
                                        pass

            except Exception as e:
                logger.error(f"Failed opening archive {zf}: {e}")
                self.report_stats["failed_images"].append({
                    "file": zf.name,
                    "archive": zf.name,
                    "reason": f"Corrupt ZIP archive: {str(e)}"
                })

            self.report_stats["zip_details"].append(zip_info)

        self.report_stats["total_images_discovered"] = total_discovered
        self.report_stats["successfully_processed"] = total_success
        self.report_stats["faces_detected"] = total_faces

        logger.info(f"Cataloging complete. Discovered: {total_discovered}, Valid: {total_success}, Failed: {len(self.report_stats['failed_images'])}")
        return self.report_stats

    def generate_markdown_report(self, output_path: Path) -> Path:
        """Generate structured markdown report of image loading verification."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stats = self.report_stats

        lines = [
            "# Image Access and Dataset Loading Audit Report",
            "",
            "## 1. Overview Summary",
            "",
            f"- **Total Student ZIP Archives Discovered**: {stats['total_zips']}",
            f"- **Total Image Files Discovered**: {stats['total_images_discovered']}",
            f"- **Successfully Decoded & Verified Images**: {stats['successfully_processed']}",
            f"- **Failed / Corrupted Images**: {len(stats['failed_images'])}",
            f"- **Identical Duplicate Image Groups**: {len(stats['duplicate_images'])}",
            f"- **Total Faces Detected**: {stats['faces_detected']}",
            "",
            "## 2. Archive-by-Archive Verification",
            "",
            "| Student ZIP Archive | Student Name | Images Found | Valid Decodable | Faces Detected | Status |",
            "|---|---|---|---|---|---|"
        ]

        for z in stats["zip_details"]:
            status = "Verified OK" if z["images_found"] == z["images_valid"] else "Contains Errors"
            lines.append(f"| {z['archive_name']} | {z['student_name']} | {z['images_found']} | {z['images_valid']} | {z['faces_in_zip']} | {status} |")

        lines.extend([
            "",
            "## 3. Failed Images Detail",
            ""
        ])

        if stats["failed_images"]:
            for f in stats["failed_images"]:
                lines.append(f"- **{f['archive']}** -> `{f['file']}`: *{f['reason']}*")
        else:
            lines.append("*(No corrupted or undecodable images found. 100% of discovered images decoded successfully.)*")

        lines.extend([
            "",
            "## 4. Cross-Archive Duplicate Images (MD5 Hash Identical)",
            ""
        ])

        if stats["duplicate_images"]:
            for h, files in stats["duplicate_images"].items():
                lines.append(f"- **Hash `{h[:8]}`**: {len(files)} identical copies across:")
                for f in files:
                    lines.append(f"  - `{f}`")
        else:
            lines.append("*(No cross-archive duplicate images detected.)*")

        with open(output_path, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(lines))

        logger.info(f"Saved image access report to {output_path}")
        return output_path

