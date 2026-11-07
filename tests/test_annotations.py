"""Unit tests for YOLO annotation parsing and bounding box conversions."""
import pytest
import numpy as np
from src.data.annotation_parser import AnnotationParser, BoundingBox
from src.data.dataset_validator import DatasetValidator


def test_bounding_box_validity():
    box_valid = BoundingBox(class_id=0, x_center=0.5, y_center=0.5, width=0.2, height=0.3)
    assert box_valid.is_valid() is True

    box_invalid = BoundingBox(class_id=0, x_center=1.5, y_center=0.5, width=0.2, height=0.3)
    assert box_invalid.is_valid() is False


def test_bounding_box_pixel_conversion():
    box = BoundingBox(class_id=1, x_center=0.5, y_center=0.5, width=0.2, height=0.4)
    x1, y1, x2, y2 = box.to_pixel_coords(img_width=1000, img_height=1000)
    assert x1 == 400
    assert y1 == 300
    assert x2 == 600
    assert y2 == 700


def test_annotation_parsing_and_deduplication():
    sample_txt = """
    0 0.500000 0.500000 0.200000 0.200000
    1 0.300000 0.300000 0.100000 0.100000
    0 0.500000 0.500000 0.200000 0.200000
    """
    # Test with deduplication
    boxes_dedup = AnnotationParser.parse_text(sample_txt, deduplicate=True)
    assert len(boxes_dedup) == 2
    assert boxes_dedup[0].class_id == 0
    assert boxes_dedup[1].class_id == 1

    # Test without deduplication
    boxes_all = AnnotationParser.parse_text(sample_txt, deduplicate=False)
    assert len(boxes_all) == 3


def test_dataset_validator_box():
    box = BoundingBox(class_id=0, x_center=0.5, y_center=0.5, width=0.2, height=0.2)
    valid, err = DatasetValidator.validate_box(box, img_w=640, img_h=480)
    assert valid is True
    assert err is None

