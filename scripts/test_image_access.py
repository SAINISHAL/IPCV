"""Script to test image access on every student ZIP archive and generate audit report."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger
from src.data.image_loader import RobustImageLoader
from src.detection.unified_detector import UnifiedFaceDetector

logger = setup_logger("TestImageAccess")


def main():
    loader = RobustImageLoader()
    detector = UnifiedFaceDetector(conf_thresh=0.55)

    print("\n" + "=" * 65)
    print("TEST A & IMAGE ACCESS VERIFICATION ACROSS ALL STUDENT ZIP ARCHIVES")
    print("=" * 65)

    stats = loader.extract_and_catalog(detector=detector)
    report_file = PROJECT_ROOT / "reports" / "image_access_report.md"
    loader.generate_markdown_report(report_file)

    print("\n" + "-" * 65)
    print("AUDIT RESULTS SUMMARY:")
    print("-" * 65)
    print(f"Total ZIP Files Found:             {stats['total_zips']}")
    print(f"Total Images Discovered:           {stats['total_images_discovered']}")
    print(f"Successfully Processed & Decoded:  {stats['successfully_processed']}")
    print(f"Failed / Corrupted Images:         {len(stats['failed_images'])}")
    print(f"Duplicate Image Groups:            {len(stats['duplicate_images'])}")
    print(f"Total Faces Detected by YuNet:     {stats['faces_detected']}")
    print(f"Report Generated At:               {report_file}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

