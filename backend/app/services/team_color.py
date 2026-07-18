from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _torso_crop(frame: Any, bbox: dict[str, float]) -> np.ndarray | None:
    height, width = frame.shape[:2]
    box_width = max(0.0, bbox["x2"] - bbox["x1"])
    box_height = max(0.0, bbox["y2"] - bbox["y1"])
    x1 = max(0, int(bbox["x1"] + box_width * 0.18))
    x2 = min(width, int(bbox["x2"] - box_width * 0.18))
    y1 = max(0, int(bbox["y1"] + box_height * 0.16))
    y2 = min(height, int(bbox["y1"] + box_height * 0.58))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return frame[y1:y2, x1:x2]


def extract_jersey_descriptor(
    frame: Any,
    bbox: dict[str, float],
) -> tuple[np.ndarray | None, str | None]:
    """Describe a player's torso color while preserving white and black kits."""
    crop = _torso_crop(frame, bbox)
    if crop is None:
        return None, None

    resized = cv2.resize(crop, (24, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    visible = pixels[:, 2] >= 25
    if int(np.count_nonzero(visible)) < 12:
        return None, None

    visible_pixels = pixels[visible]
    saturated = visible_pixels[:, 1] >= 45
    hue_histogram, _ = np.histogram(
        visible_pixels[saturated, 0] if np.any(saturated) else np.array([], dtype=np.uint8),
        bins=18,
        range=(0, 180),
    )
    value_histogram, _ = np.histogram(
        visible_pixels[~saturated, 2] if np.any(~saturated) else np.array([], dtype=np.uint8),
        bins=6,
        range=(0, 256),
    )
    descriptor = np.concatenate(
        [hue_histogram.astype(np.float32), value_histogram.astype(np.float32)]
    )
    descriptor /= float(np.linalg.norm(descriptor) + 1e-8)

    representative_hsv = np.median(visible_pixels, axis=0).astype(np.uint8)
    representative_bgr = cv2.cvtColor(
        representative_hsv.reshape(1, 1, 3),
        cv2.COLOR_HSV2BGR,
    )[0, 0]
    blue, green, red = [int(value) for value in representative_bgr]
    return descriptor, f"#{red:02x}{green:02x}{blue:02x}"


def jersey_similarity(reference: np.ndarray | None, candidate: np.ndarray | None) -> float:
    if reference is None or candidate is None or reference.shape != candidate.shape:
        return 0.0
    return float(np.clip(np.dot(reference, candidate), 0.0, 1.0))


def add_team_color_groups(frame: Any, detections: list[dict[str, Any]]) -> None:
    color_entries: list[tuple[dict[str, Any], np.ndarray, str]] = []
    for detection in detections:
        if detection["role"] not in {"player", "goalkeeper"}:
            continue
        descriptor, color_hex = extract_jersey_descriptor(frame, detection["bbox"])
        if descriptor is None or color_hex is None:
            continue
        detection["jersey_descriptor"] = [round(float(value), 5) for value in descriptor]
        detection["jersey_color_hex"] = color_hex
        color_entries.append((detection, descriptor, color_hex))

    if not color_entries:
        return
    if len(color_entries) == 1:
        color_entries[0][0]["team_id"] = "team_1"
        return

    features = np.stack([entry[1] for entry in color_entries]).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    _, labels, _ = cv2.kmeans(
        features,
        2,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    for entry, label in zip(color_entries, labels.flatten(), strict=True):
        entry[0]["team_id"] = f"team_{int(label) + 1}"
