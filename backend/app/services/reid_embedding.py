"""OSNet person-ReID appearance embeddings (with automatic HSV fallback).

The detection cache calls `appearance_backend()` once per build to decide which
descriptor to store, then `embed_crops()` (OSNet) or the HSV descriptor per
detection. If the OSNet checkpoint is absent or fails to load, everything falls
back to the existing HSV descriptor so the app never hard-breaks.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.config import (
    REID_BATCH_SIZE,
    REID_EMBED_DIM,
    REID_ENABLED,
    REID_INPUT_HEIGHT,
    REID_INPUT_WIDTH,
    REID_MODEL_PATH,
)
from app.services.appearance import APPEARANCE_DESCRIPTOR_SIZE

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@lru_cache(maxsize=1)
def _load_reid() -> tuple[Any, Any] | None:
    """Load OSNet-x0.25 once. Returns (model, device) or None if unavailable."""
    if not REID_ENABLED:
        return None
    path = Path(REID_MODEL_PATH)
    if not path.exists():
        return None
    try:
        import torch

        from app.services.osnet import load_local_weights, osnet_x0_25

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = osnet_x0_25(num_classes=1000)
        load_local_weights(model, str(path))
        model.eval().to(device)
        return model, device
    except Exception as error:  # missing torch, bad checkpoint, etc. -> HSV fallback
        import warnings

        warnings.warn(f"OSNet ReID unavailable ({error}); falling back to HSV appearance.")
        return None


def reid_available() -> bool:
    return _load_reid() is not None


@lru_cache(maxsize=1)
def reid_key() -> str:
    """Fingerprint of the active appearance backend, for cache invalidation."""
    if not reid_available():
        return "hsv"
    path = Path(REID_MODEL_PATH)
    try:
        stat = path.stat()
        raw = f"osnet:{path.name}:{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        raw = "osnet:unknown"
    return "osnet-" + hashlib.sha1(raw.encode()).hexdigest()[:10]


def appearance_backend() -> tuple[str, int, str]:
    """(name, descriptor_dim, key) for the whole cache build. Chosen ONCE so the
    det_appearance array stays homogeneous."""
    if reid_available():
        return "osnet", REID_EMBED_DIM, reid_key()
    return "hsv", APPEARANCE_DESCRIPTOR_SIZE, "hsv"


def _preprocess(crop: np.ndarray) -> np.ndarray | None:
    """BGR crop -> normalized CHW float32 (ImageNet stats), OSNet input size."""
    if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (REID_INPUT_WIDTH, REID_INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    normalized = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.transpose(normalized, (2, 0, 1))  # CHW


def embed_crops(crops: list[np.ndarray | None]) -> list[np.ndarray | None]:
    """Batched OSNet embeddings, L2-normalized (512-d). None per unusable crop.

    Returns all-None (same length) when OSNet is unavailable — callers then use
    the HSV path. Preserves input order so results map 1:1 to detections.
    """
    loaded = _load_reid()
    if loaded is None:
        return [None] * len(crops)
    import torch

    model, device = loaded
    results: list[np.ndarray | None] = [None] * len(crops)
    valid_idx: list[int] = []
    tensors: list[np.ndarray] = []
    for i, crop in enumerate(crops):
        pre = _preprocess(crop)
        if pre is not None:
            valid_idx.append(i)
            tensors.append(pre)
    if not tensors:
        return results

    with torch.inference_mode():
        for start in range(0, len(tensors), REID_BATCH_SIZE):
            chunk = tensors[start:start + REID_BATCH_SIZE]
            batch = torch.from_numpy(np.stack(chunk)).to(device)
            features = model(batch).cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
            features = features / norms
            for offset, feature in enumerate(features):
                results[valid_idx[start + offset]] = feature
    return results
