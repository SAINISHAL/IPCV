"""Compare source photos before assigning them to dataset splits."""
import cv2
import numpy as np


def fingerprint(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    thumb = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(thumb)[:8, :8].ravel()[1:]
    return low > np.median(low), thumb


def near_duplicate(first, second, *, max_difference=6.0):
    bits_a, a = first
    bits_b, b = second
    if np.count_nonzero(bits_a != bits_b) > 7:
        return False
    # The second check keeps distinct photos of a similar scene separate.
    difference = a - b
    difference -= np.median(difference)
    return float(np.mean(np.abs(difference))) <= max_difference


def group_photos(fingerprints, links=()):
    parents = {key: key for key in fingerprints}

    def root(key):
        while key != parents[key]:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    keys = sorted(fingerprints)
    for i, key in enumerate(keys):
        for other in keys[:i]:
            if near_duplicate(fingerprints[key], fingerprints[other]):
                parents[root(key)] = root(other)
    for left, right in links:
        parents[root(left)] = root(right)
    return {key: root(key) for key in keys}
