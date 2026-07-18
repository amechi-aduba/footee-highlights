from __future__ import annotations

import cv2
import numpy as np

from app.core.config import (
    PITCH_MASK_MARGIN_FRACTION,
    PITCH_MASK_MIN_GRASS_FRACTION,
)

_GRASS_HSV_LOW = np.array([30, 40, 40], dtype=np.uint8)
_GRASS_HSV_HIGH = np.array([90, 255, 255], dtype=np.uint8)


def grass_fraction(frame_small: np.ndarray) -> float:
    """Fraction of the frame that looks like grass/turf. Cheap (HSV threshold)."""
    hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _GRASS_HSV_LOW, _GRASS_HSV_HIGH)
    return float(np.count_nonzero(mask)) / float(mask.size)


def grass_hull(frame_small: np.ndarray) -> np.ndarray | None:
    """Convex hull (in small-frame coords) of the largest grass region, or None.

    Returns None when grass covers too little of the frame — indoor footage,
    extreme zoom on crowd, etc. Callers must treat None as "filter disabled",
    never as "everything off pitch".
    """
    hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _GRASS_HSV_LOW, _GRASS_HSV_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    grass_fraction = float(np.count_nonzero(mask)) / float(mask.size)
    if grass_fraction < PITCH_MASK_MIN_GRASS_FRACTION:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < PITCH_MASK_MIN_GRASS_FRACTION * mask.size * 0.5:
        return None
    return cv2.convexHull(largest)


def feet_on_pitch(
    bbox: dict[str, float],
    hull: np.ndarray | None,
    scale: float,
    frame_height_small: int,
) -> bool:
    """A detection is on-pitch when its feet (bbox bottom-center) fall inside the
    dilated grass hull. Spectators behind the touchline fail this; players whose
    torso overlaps the crowd still pass because their feet are on grass."""
    if hull is None:
        return True
    foot_x = (bbox["x1"] + bbox["x2"]) / 2 * scale
    foot_y = bbox["y2"] * scale
    margin = PITCH_MASK_MARGIN_FRACTION * frame_height_small
    distance = cv2.pointPolygonTest(hull, (float(foot_x), float(foot_y)), True)
    return distance >= -margin
