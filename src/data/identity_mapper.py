"""Identity Mapping Module between YOLO Class IDs and Student Roll Numbers."""
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.config import load_dataset_profile


class IdentityMapper:
    """Map zero-based YOLO class IDs to student IDs from the local label file."""

    UNKNOWN_LABEL = "UNKNOWN"

    def __init__(self, label_file_path: Optional[str or Path] = None):
        self.verified_archives = {name.strip().lower() for name in load_dataset_profile().get("verified_archives", [])}
        self.class_to_student: Dict[int, str] = {}
        self.student_to_class: Dict[str, int] = {}
        if label_file_path:
            self.load_labels(label_file_path)

    def load_labels(self, label_file_path: str or Path) -> None:
        """Parse label.txt and initialize mappings."""
        p = Path(label_file_path)
        if not p.exists():
            raise FileNotFoundError(f"Label file not found: {p}")
            
        with open(p, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            
        self.class_to_student.clear()
        self.student_to_class.clear()
        for idx, line in enumerate(lines):
            # Parse '1: STUDENT_ID' or a bare student ID.
            if ":" in line:
                val = line.split(":", 1)[1].strip()
            else:
                val = line
            self.class_to_student[idx] = val
            self.student_to_class[val] = idx

    def map_class_to_student(self, class_id: int or str, archive_name: Optional[str] = None) -> str:
        """Map a class ID to a student ID. If the archive is unverified, returns UNKNOWN."""
        if archive_name is not None and not self.is_archive_verified(archive_name):
            return self.UNKNOWN_LABEL

        try:
            cid = int(class_id)
            return self.class_to_student.get(cid, self.UNKNOWN_LABEL)
        except (ValueError, TypeError):
            return self.UNKNOWN_LABEL

    def map_student_to_class(self, student_id: str) -> Optional[int]:
        """Map student ID string to numeric class index."""
        return self.student_to_class.get(student_id, None)

    def is_archive_verified(self, archive_name: str) -> bool:
        """Check if an archive is in the verified consistent list."""
        clean = archive_name.lower().split("-2026")[0].strip()
        return clean in self.verified_archives

    def get_enrolled_students(self) -> List[str]:
        """Return list of all enrolled student IDs."""
        return list(self.student_to_class.keys())

