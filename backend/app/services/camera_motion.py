from __future__ import annotations

import cv2
import numpy as np

from app.core.config import (
    CAMERA_MOTION_DOWNSCALE_WIDTH,
    CAMERA_MOTION_MIN_INLIER_RATIO,
)

IDENTITY_AFFINE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def downscale_gray(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a small grayscale copy and the small->full scale factor."""
    height, width = frame.shape[:2]
    scale = min(1.0, CAMERA_MOTION_DOWNSCALE_WIDTH / max(1, width))
    small = cv2.resize(
        frame,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray, scale


def _feature_mask(
    shape: tuple[int, int],
    exclude_boxes: list[dict[str, float]],
    scale: float,
) -> np.ndarray:
    """Mask out player/person boxes so flow tracks the background, not the players."""
    mask = np.full(shape, 255, dtype=np.uint8)
    for bbox in exclude_boxes:
        x1 = max(0, int(bbox["x1"] * scale) - 4)
        y1 = max(0, int(bbox["y1"] * scale) - 4)
        x2 = min(shape[1], int(bbox["x2"] * scale) + 4)
        y2 = min(shape[0], int(bbox["y2"] * scale) + 4)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0
    return mask


def estimate_global_affine(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    scale: float,
    exclude_boxes: list[dict[str, float]] | None = None,
) -> tuple[np.ndarray, float]:
    """Estimate the FULL camera transform between two frames (BoT-SORT GMC style).

    Returns a 2x3 similarity affine in full-resolution pixel coordinates
    (translation + rotation + SCALE — the scale is what a zoom is) plus the
    RANSAC inlier ratio as a confidence signal.

    Translation-only compensation is simply wrong under zoom: every static
    pixel moves radially, so compensated distances become garbage and a
    far-away player can look "close". We already computed this matrix with
    estimateAffinePartial2D — the historic bug was discarding everything but
    the translation.

    On failure returns (identity, 0.0). Callers must treat low confidence as a
    reason to get MORE conservative (shrink gates / prefer searching), never
    less — the camera is least known exactly when tracking is most fragile.
    """
    mask = _feature_mask(previous_gray.shape[:2], exclude_boxes or [], scale)
    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=16,
        mask=mask,
        blockSize=7,
    )
    if points is None or len(points) < 12:
        return IDENTITY_AFFINE.copy(), 0.0

    moved, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
    )
    if moved is None or status is None:
        return IDENTITY_AFFINE.copy(), 0.0
    valid = status.reshape(-1).astype(bool)
    source = points.reshape(-1, 2)[valid]
    target = moved.reshape(-1, 2)[valid]
    if len(source) < 12:
        return IDENTITY_AFFINE.copy(), 0.0

    matrix, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    if matrix is None or inliers is None:
        return IDENTITY_AFFINE.copy(), 0.0
    inlier_ratio = float(np.count_nonzero(inliers)) / float(len(source))
    if inlier_ratio < CAMERA_MOTION_MIN_INLIER_RATIO:
        return IDENTITY_AFFINE.copy(), 0.0

    # Convert small-frame affine to full-resolution coordinates:
    # full = small / scale, so A_full = S^-1 · A_small · S — the linear part is
    # unchanged, only the translation rescales.
    affine = matrix.astype(np.float32)
    affine[:, 2] /= max(scale, 1e-6)
    return affine, inlier_ratio


def estimate_global_shift(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    scale: float,
    exclude_boxes: list[dict[str, float]] | None = None,
) -> tuple[np.ndarray, float]:
    """Legacy wrapper: translation component only. Prefer estimate_global_affine —
    translation-only compensation is wrong under zoom."""
    affine, confidence = estimate_global_affine(
        previous_gray, current_gray, scale, exclude_boxes
    )
    return affine[:, 2].copy(), confidence
