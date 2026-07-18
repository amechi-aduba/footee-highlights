from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException

from app.core.config import (
    BALL_MIN_CONFIDENCE,
    DETECTION_SAMPLE_EVERY_N_FRAMES,
    PITCH_MASK_ENABLED,
    SCENE_OBJECT_LAYOUT_GRID_COLUMNS,
    SCENE_OBJECT_LAYOUT_GRID_ROWS,
    SCENE_OBJECT_LAYOUT_MIN_DETECTIONS,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_MODEL_IS_FOOTBALL_SPECIFIC,
    YOLO_MODEL_PATH,
)
from app.services.model_registry import get_detection_model
from app.services.team_color import add_team_color_groups


SUPPORTED_ROLES = ("player", "ball", "goalkeeper", "referee")
LAYOUT_ROLES = ("player", "goalkeeper", "referee", "ball")
LAYOUT_ROLE_WEIGHTS = {
    "player": 1.0,
    "goalkeeper": 1.0,
    "referee": 0.85,
    "ball": 0.5,
}


@dataclass(frozen=True)
class DetectionWindow:
    start_frame: int
    end_frame: int


def _normalize_class_name(class_name: str) -> str | None:
    normalized = class_name.lower().replace("-", "_").replace(" ", "_")
    if normalized in {"person", "player", "soccer_player", "football_player"}:
        return "player"
    if normalized in {"sports_ball", "ball", "soccer_ball", "football"}:
        return "ball"
    if normalized in {"goalkeeper", "goalie", "keeper"}:
        return "goalkeeper"
    if normalized in {"referee", "official", "match_official"}:
        return "referee"
    return None


def _filter_off_pitch(frame: Any, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop person-class detections whose feet are off the grass hull (spectators).

    Auto-disables (keeps everything) when the frame has too little grass —
    the filter must never misfire on indoor or heavily zoomed footage."""
    if not PITCH_MASK_ENABLED or not detections:
        return detections
    from app.services.camera_motion import downscale_gray
    from app.services.pitch_mask import feet_on_pitch, grass_hull

    gray, scale = downscale_gray(frame)
    small = cv2.resize(frame, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
    hull = grass_hull(small)
    if hull is None:
        return detections
    return [
        detection
        for detection in detections
        if detection["role"] == "ball"
        or feet_on_pitch(detection["bbox"], hull, scale, gray.shape[0])
    ]


def _filter_balls(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """At most ONE ball on a single frame, and only above the ball confidence
    floor. Turf dots/line marks fire the ball class; without temporal context
    the best single-frame policy is: highest confidence wins, weak ones drop."""
    balls = [
        detection
        for detection in detections
        if detection["role"] == "ball" and detection["confidence"] >= BALL_MIN_CONFIDENCE
    ]
    best_ball = max(balls, key=lambda detection: detection["confidence"], default=None)
    return [
        detection
        for detection in detections
        if detection["role"] != "ball" or detection is best_ball
    ]


def _detect_frame(frame: Any, frame_number: int, timestamp_seconds: float) -> list[dict[str, Any]]:
    model = get_detection_model()
    try:
        result = model.predict(frame, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)[0]
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"YOLO inference failed: {error}") from error

    detections: list[dict[str, Any]] = []
    for index, box in enumerate(result.boxes):
        model_class = str(result.names[int(box.cls.item())])
        role = _normalize_class_name(model_class)
        if role is None:
            continue
        x1, y1, x2, y2 = [round(float(value), 2) for value in box.xyxy[0].tolist()]
        detections.append(
            {
                "detection_id": f"frame_{frame_number:08d}_det_{index:03d}",
                "role": role,
                "model_class": model_class,
                "confidence": round(float(box.conf.item()), 4),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
        )
    add_team_color_groups(frame, detections)
    return detections


def object_layout_signature(
    frame: Any,
    frame_number: int,
    timestamp_seconds: float,
) -> np.ndarray | None:
    """Return a compact YOLO object-layout signature for scene-cut detection.

    This is intentionally tracker-free. ByteTrack can later replace this with
    track-aware identities, but a grid signature is enough to catch many
    same-camera cuts where object positions change sharply.
    """
    detections = _detect_frame(frame, frame_number, timestamp_seconds)
    if len(detections) < SCENE_OBJECT_LAYOUT_MIN_DETECTIONS:
        return None

    height, width = frame.shape[:2]
    role_count = len(LAYOUT_ROLES)
    spatial_size = SCENE_OBJECT_LAYOUT_GRID_ROWS * SCENE_OBJECT_LAYOUT_GRID_COLUMNS
    signature = np.zeros(role_count * spatial_size, dtype=np.float32)

    for detection in detections:
        role = detection["role"]
        if role not in LAYOUT_ROLES:
            continue
        bbox = detection["bbox"]
        center_x = (bbox["x1"] + bbox["x2"]) / 2
        center_y = (bbox["y1"] + bbox["y2"]) / 2
        column = min(
            SCENE_OBJECT_LAYOUT_GRID_COLUMNS - 1,
            max(0, int(center_x / width * SCENE_OBJECT_LAYOUT_GRID_COLUMNS)),
        )
        row = min(
            SCENE_OBJECT_LAYOUT_GRID_ROWS - 1,
            max(0, int(center_y / height * SCENE_OBJECT_LAYOUT_GRID_ROWS)),
        )
        role_index = LAYOUT_ROLES.index(role)
        cell_index = row * SCENE_OBJECT_LAYOUT_GRID_COLUMNS + column
        signature[role_index * spatial_size + cell_index] += (
            detection["confidence"] * LAYOUT_ROLE_WEIGHTS[role]
        )

    total_weight = float(np.sum(signature))
    if total_weight <= 0:
        return None

    normalized_layout = signature / total_weight
    detection_density = min(len(detections) / 22, 1.0)
    return np.concatenate([normalized_layout, np.array([detection_density], dtype=np.float32)])


def object_layout_difference(
    previous_signature: np.ndarray | None,
    current_signature: np.ndarray | None,
) -> float:
    if previous_signature is None or current_signature is None:
        return 0.0
    spatial_difference = float(np.sum(np.abs(previous_signature[:-1] - current_signature[:-1])) / 2)
    count_difference = float(abs(previous_signature[-1] - current_signature[-1]))
    return (0.8 * spatial_difference) + (0.2 * count_difference)


def detect_objects_at_timestamp(video_path: Path, timestamp_seconds: float) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise HTTPException(status_code=422, detail="Could not open uploaded video")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_number = max(0, round(timestamp_seconds * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = capture.read()
        if not success:
            raise HTTPException(status_code=422, detail="Could not read selected video frame")
        detections = _filter_balls(
            _filter_off_pitch(frame, _detect_frame(frame, frame_number, timestamp_seconds))
        )
        return {
            "timestamp_seconds": round(timestamp_seconds, 3),
            "frame_number": frame_number,
            "frame_width": int(frame.shape[1]),
            "frame_height": int(frame.shape[0]),
            "detections": detections,
        }
    finally:
        capture.release()


def detect_objects_in_window(
    video_path: Path,
    fps: float,
    window: DetectionWindow,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise HTTPException(status_code=422, detail="Could not open uploaded video")

    counts: Counter[str] = Counter()
    sampled_frames = 0
    try:
        for frame_number in range(
            window.start_frame,
            window.end_frame,
            DETECTION_SAMPLE_EVERY_N_FRAMES,
        ):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = capture.read()
            if not success:
                continue
            sampled_frames += 1
            frame_roles = {detection["role"] for detection in _detect_frame(frame, frame_number, frame_number / fps)}
            counts.update(frame_roles)
    finally:
        capture.release()

    return {
        "status": "completed",
        "model_path": YOLO_MODEL_PATH,
        "start_time": round(window.start_frame / fps, 3),
        "sample_every_n_frames": DETECTION_SAMPLE_EVERY_N_FRAMES,
        "sampled_frames": sampled_frames,
        "counts": {role: counts[role] for role in SUPPORTED_ROLES},
        "counting_note": "Counts are sampled frames containing each role, not unique tracked objects.",
        "role_note": (
            "Football-specific player, goalkeeper, referee, and ball classes are active."
            if YOLO_MODEL_IS_FOOTBALL_SPECIFIC
            else "YOLO11m fallback maps COCO person to player and sports ball to ball. "
            "Train and promote the reviewed football dataset to activate goalkeeper and referee classes."
        ),
    }
