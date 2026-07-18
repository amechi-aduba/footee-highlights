from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.core.config import TRACKING_MODEL_PATH, YOLO_MODEL_PATH


def _load_yolo(model_path: str) -> Any:
    try:
        from ultralytics import YOLO

        return YOLO(model_path)
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load YOLO model '{model_path}': {error}",
        ) from error


@lru_cache(maxsize=1)
def get_detection_model() -> Any:
    """High-quality detector (yolo11m) for click-frame detection. Loaded once per process."""
    return _load_yolo(YOLO_MODEL_PATH)


@lru_cache(maxsize=1)
def get_tracking_model() -> Any:
    """Fast detector (yolo11n when promoted) for the detection-cache pass. Loaded once per process."""
    return _load_yolo(TRACKING_MODEL_PATH)


def model_cache_key(model_path: str = TRACKING_MODEL_PATH) -> str:
    """Short fingerprint of the model file, used to invalidate detection caches."""
    path = Path(model_path)
    try:
        stat = path.stat()
        raw = f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        raw = path.name
    return hashlib.sha1(raw.encode()).hexdigest()[:10]
