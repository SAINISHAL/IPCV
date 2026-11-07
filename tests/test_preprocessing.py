"""Unit tests for image preprocessing and quality filtering."""
import pytest
import numpy as np
from src.preprocessing.image_preprocessing import ImagePreprocessor
from src.preprocessing.quality_filter import QualityFilter
from src.detection.face_extractor import FaceExtractor


def test_quality_filter():
    # Sharp gradient image
    sharp_img = np.zeros((100, 100, 3), dtype=np.uint8)
    sharp_img[:, :50] = 255
    var = QualityFilter.compute_blur_laplacian_variance(sharp_img)
    assert var > 50.0

    ok, msg = QualityFilter.is_acceptable_quality(sharp_img, min_size=(50, 50), min_laplacian_var=10.0)
    assert ok is True

    # Blurry image
    blurry_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    ok_b, _ = QualityFilter.is_acceptable_quality(blurry_img, min_size=(50, 50), min_laplacian_var=10.0)
    assert ok_b is False


def test_image_preprocessor_dimensions():
    preprocessor = ImagePreprocessor(target_size=(100, 100), enable_clahe=True, normalize_intensity=True)
    dummy_face = np.random.randint(0, 256, (120, 80, 3), dtype=np.uint8)

    processed_2d = preprocessor.process(dummy_face)
    assert processed_2d.shape == (100, 100)
    assert processed_2d.dtype == np.float32
    assert 0.0 <= processed_2d.min() and processed_2d.max() <= 1.0

    flattened = preprocessor.process_and_flatten(dummy_face)
    assert flattened.shape == (10000,)


def test_face_extractor():
    extractor = FaceExtractor(output_dir="tmp_test_dir", min_face_size=(20, 20))
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    crop = extractor.extract_crop(img, (50, 50, 150, 150))
    assert crop is not None
    assert crop.shape == (100, 100, 3)

    # Degenerate crop
    bad_crop = extractor.extract_crop(img, (50, 50, 55, 55))
    assert bad_crop is None

