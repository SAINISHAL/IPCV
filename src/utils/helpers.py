"""General helper functions for filesystem, visualization, and image conversions."""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any


def ensure_dir(dir_path: str or Path) -> Path:
    """Ensure directory exists and return Path object."""
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_labels(label_file_path: str or Path) -> Dict[int, str]:
    """Read label.txt file mapping 1-based or 0-based lines to student roll numbers.
    
    Returns:
        dict mapping 0-based class ID (int) -> student ID (str)
    """
    labels = {}
    p = Path(label_file_path)
    if not p.exists():
        raise FileNotFoundError(f"Label file not found: {p}")
        
    with open(p, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            clean = line.strip()
            if not clean:
                continue
            # Handle indexed labels and bare student IDs.
            if ":" in clean:
                parts = clean.split(":")
                val = parts[1].strip()
            else:
                val = clean
            labels[idx] = val
    return labels


def draw_bounding_box(image: np.ndarray, box: Tuple[int, int, int, int], label: str,
                      confidence: float = None, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Draw a bounding box with label and optional confidence score on an image."""
    img = image.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    caption = label
    if confidence is not None:
        caption += f" {confidence:.2f}"
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(caption, font, font_scale, thickness)
    
    # Background rectangle for text
    cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
    # Text in black or white depending on color brightness
    text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    cv2.putText(img, caption, (x1 + 3, y1 - 4), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return img

