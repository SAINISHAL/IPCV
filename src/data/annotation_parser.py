"""YOLO Annotation Parser and Validator."""
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


class BoundingBox:
    """Represents a single normalized or pixel-space bounding box."""
    def __init__(self, class_id: int or str, x_center: float, y_center: float, width: float, height: float):
        self.class_id = int(class_id) if str(class_id).isdigit() else str(class_id)
        self.x_center = float(x_center)
        self.y_center = float(y_center)
        self.width = float(width)
        self.height = float(height)

    def is_valid(self) -> bool:
        """Check if normalized coordinates are within [0, 1] and width/height > 0."""
        return (
            0.0 <= self.x_center <= 1.0 and
            0.0 <= self.y_center <= 1.0 and
            0.0 < self.width <= 1.0 and
            0.0 < self.height <= 1.0
        )

    def to_pixel_coords(self, img_width: int, img_height: int, clamp: bool = True) -> Tuple[int, int, int, int]:
        """Convert normalized YOLO box to (x1, y1, x2, y2) pixel coordinates."""
        x1 = (self.x_center - self.width / 2.0) * img_width
        y1 = (self.y_center - self.height / 2.0) * img_height
        x2 = (self.x_center + self.width / 2.0) * img_width
        y2 = (self.y_center + self.height / 2.0) * img_height

        if clamp:
            x1 = max(0, min(int(round(x1)), img_width - 1))
            y1 = max(0, min(int(round(y1)), img_height - 1))
            x2 = max(0, min(int(round(x2)), img_width))
            y2 = max(0, min(int(round(y2)), img_height))
        else:
            x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            
        return (x1, y1, x2, y2)

    def to_yolo_line(self) -> str:
        """Format back to standard YOLO txt line."""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


class AnnotationParser:
    """Parses, cleans, and validates YOLO annotation files."""

    @staticmethod
    def parse_text(content: str, deduplicate: bool = True) -> List[BoundingBox]:
        """Parse raw annotation text into list of BoundingBox objects."""
        boxes = []
        seen = set()
        for line in content.strip().splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            if deduplicate and line_str in seen:
                continue
            seen.add(line_str)
            
            parts = line_str.split()
            if len(parts) >= 5:
                try:
                    cid = parts[0]
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    box = BoundingBox(cid, xc, yc, w, h)
                    boxes.append(box)
                except ValueError:
                    continue
        return boxes

    @staticmethod
    def parse_file(file_path: str or Path, deduplicate: bool = True) -> List[BoundingBox]:
        """Parse an annotation text file from disk."""
        path = Path(file_path)
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return AnnotationParser.parse_text(f.read(), deduplicate=deduplicate)

