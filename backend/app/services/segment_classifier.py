"""Classify segments as gameplay vs cutaway (intros, title cards, celebrations).

Two cheap signals per segment:
  1. Grass fraction on a few evenly sampled downscaled frames — intro graphics,
     title cards, and close-up celebrations show little or no turf.
  2. Player count from one YOLO pass on the middle sample — gameplay shows
     several players; graphics show none.

Classification NEVER deletes anything: segments are labeled and the UI tucks
non-gameplay clips into a collapsed section the user can still open.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from app.core.config import (
    SEGMENT_CLASSIFY_SAMPLES,
    SEGMENT_FILTER_ENABLED,
    SEGMENT_FILTER_USE_PLAYER_MODEL,
    SEGMENT_GAMEPLAY_MIN_PLAYERS,
    SEGMENT_GRASS_FRACTION_MIN,
    TRACKING_CONFIDENCE_THRESHOLD,
)
from app.services.model_registry import get_tracking_model
from app.services.object_detection import _normalize_class_name
from app.services.pitch_mask import grass_fraction
from app.services.scene_detection import SceneSegment


def _sample_frame_numbers(segment: SceneSegment) -> list[int]:
    length = max(1, segment.end_frame - segment.start_frame)
    count = max(1, SEGMENT_CLASSIFY_SAMPLES)
    # Evenly spaced, away from the exact boundaries (which may hold transition residue).
    return [
        segment.start_frame + int(length * (index + 1) / (count + 1))
        for index in range(count)
    ]


def _count_players(frame: np.ndarray) -> int:
    model = get_tracking_model()
    try:
        result = model.predict(
            frame, conf=TRACKING_CONFIDENCE_THRESHOLD, imgsz=416, verbose=False
        )[0]
    except Exception:
        return 0
    count = 0
    for box in result.boxes:
        role = _normalize_class_name(str(result.names[int(box.cls.item())]))
        if role in {"player", "goalkeeper"}:
            count += 1
    return count


def classify_segments(
    video_path: Path,
    segments: list[SceneSegment],
    progress_callback: Callable[[float, str, int | None, int | None], None] | None = None,
) -> list[dict[str, Any]]:
    """One entry per segment: {"kind": "gameplay"|"cutaway", "gameplay_score": float}."""
    total_segments = len(segments)
    if progress_callback:
        progress_callback(
            0.01,
            f"Preparing to filter {total_segments} clip{'s' if total_segments != 1 else ''}",
            0,
            total_segments,
        )
    if not SEGMENT_FILTER_ENABLED:
        if progress_callback:
            progress_callback(
                1.0,
                "Cutaway filtering disabled",
                total_segments,
                total_segments,
            )
        return [{"kind": "gameplay", "gameplay_score": 1.0} for _ in segments]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        if progress_callback:
            progress_callback(
                1.0,
                "Cutaway filtering unavailable; keeping every clip",
                total_segments,
                total_segments,
            )
        return [{"kind": "gameplay", "gameplay_score": 1.0} for _ in segments]

    classifications: list[dict[str, Any]] = []
    try:
        for segment_index, segment in enumerate(segments, start=1):
            sample_numbers = _sample_frame_numbers(segment)
            grassy_samples = 0
            total_samples = 0
            middle_frame: np.ndarray | None = None
            for position, frame_number in enumerate(sample_numbers):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                success, frame = capture.read()
                if not success:
                    continue
                total_samples += 1
                height, width = frame.shape[:2]
                scale = min(1.0, 480 / max(1, width))
                small = cv2.resize(
                    frame, (max(1, int(width * scale)), max(1, int(height * scale)))
                )
                if grass_fraction(small) >= SEGMENT_GRASS_FRACTION_MIN:
                    grassy_samples += 1
                if (
                    SEGMENT_FILTER_USE_PLAYER_MODEL
                    and position == len(sample_numbers) // 2
                ):
                    middle_frame = frame

            if total_samples == 0:
                classifications.append({"kind": "gameplay", "gameplay_score": 1.0})
                if progress_callback:
                    progress_callback(
                        segment_index / max(1, total_segments),
                        f"Filtering clip {segment_index} of {total_segments}",
                        segment_index,
                        total_segments,
                    )
                continue

            grass_score = grassy_samples / total_samples
            player_count: int | None = None
            if SEGMENT_FILTER_USE_PLAYER_MODEL and middle_frame is not None:
                player_count = _count_players(middle_frame)
                player_score = min(
                    1.0, player_count / max(1, SEGMENT_GAMEPLAY_MIN_PLAYERS)
                )
                gameplay_score = 0.6 * grass_score + 0.4 * player_score
            else:
                gameplay_score = grass_score

            classifications.append(
                {
                    "kind": "gameplay" if gameplay_score >= 0.5 else "cutaway",
                    "gameplay_score": round(gameplay_score, 3),
                    "grass_score": round(grass_score, 3),
                    "player_count": player_count,
                }
            )
            if progress_callback:
                progress_callback(
                    segment_index / max(1, total_segments),
                    f"Filtering clip {segment_index} of {total_segments}",
                    segment_index,
                    total_segments,
                )
    finally:
        capture.release()
    if total_segments == 0 and progress_callback:
        progress_callback(1.0, "No clips needed filtering", 0, 0)
    return classifications
