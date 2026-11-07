"""Script to run the full dataset inspection and generate reports & figures."""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.archive_extractor import ArchiveExtractor
from src.data.dataset_inspector import DatasetInspector

logger = setup_logger("InspectDatasetScript")


def main():
    config = load_config()
    raw_dir = PROJECT_ROOT / config["paths"]["raw_data_dir"]
    archives_dir = (PROJECT_ROOT / config["paths"]["student_archives_dir"]).resolve()
    label_path = PROJECT_ROOT / config["paths"]["label_file"]
    reports_dir = PROJECT_ROOT / config["paths"]["reports_dir"]

    logger.info("Extracting raw archives if needed...")
    extractor = ArchiveExtractor(archives_dir, raw_dir)
    extractor.extract_all(overwrite=False)

    logger.info("Running dataset inspector...")
    inspector = DatasetInspector(raw_dir, label_path, reports_dir)
    inspector.run_full_inspection()
    
    logger.info("Generating visualization figures...")
    inspector.generate_visualizations()
    
    logger.info("Generating markdown report...")
    report_file = inspector.generate_markdown_report()
    logger.info(f"Dataset inspection completed successfully. Report generated at: {report_file}")


if __name__ == "__main__":
    main()

