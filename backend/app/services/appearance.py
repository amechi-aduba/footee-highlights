from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# 18*16 color bins + 12 texture bins
APPEARANCE_DESCRIPTOR_SIZE = 300


def cheap_appearance_descriptor(frame: Any, bbox: dict[str, float]) -> np.ndarray | None:
    """HSV color + gradient-texture descriptor for a player crop.

    Deliberately not neural: ImageNet embeddings on tiny upscaled crops are noise,
    while this stays meaningful down to ~24 px wide crops and costs microseconds.
    """
    height, width = frame.shape[:2]
    x1 = max(0, int(bbox["x1"]))
    y1 = max(0, int(bbox["y1"]))
    x2 = min(width, int(bbox["x2"]))
    y2 = min(height, int(bbox["y2"]))
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None
    crop = frame[y1:y2, x1:x2]

    resized = cv2.resize(crop, (32, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    color_histogram = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256]).flatten()
    color_histogram /= float(np.linalg.norm(color_histogram) + 1e-8)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)
    texture_histogram, _ = np.histogram(angle, bins=12, range=(0, 360), weights=magnitude)
    texture_histogram = texture_histogram.astype(np.float32)
    texture_histogram /= float(np.linalg.norm(texture_histogram) + 1e-8)

    descriptor = np.concatenate([color_histogram * 0.8, texture_histogram * 0.2]).astype(np.float32)
    descriptor /= float(np.linalg.norm(descriptor) + 1e-8)
    return descriptor


def cosine_similarity(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None:
        return None
    return float(np.clip(np.dot(first, second), 0.0, 1.0))
