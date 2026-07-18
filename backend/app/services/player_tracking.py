from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException

from app.core.config import (
    TRACKER_CONFIG,
    TRACKING_APPEARANCE_WEIGHT,
    TRACKING_BOX_SMOOTHING_ALPHA,
    TRACKING_CONFIDENCE_THRESHOLD,
    TRACKING_FRAME_STRIDE,
    TRACKING_IMAGE_SIZE,
    TRACKING_IOU_WEIGHT,
    TRACKING_MAX_MISSING_FRAMES,
    TRACKING_MAX_PREDICTION_FRAMES,
    TRACKING_MAX_REASSOCIATION_FRAME_FRACTION,
    TRACKING_LONG_GAP_APPEARANCE,
    TRACKING_LONG_GAP_FRAMES,
    TRACKING_MIN_APPEARANCE_SIMILARITY,
    TRACKING_POSITION_WEIGHT,
    TRACKING_REASSOCIATION_THRESHOLD,
    TRACKING_REASSOCIATION_CONFIRM_FRAMES,
    TRACKING_REASSOCIATION_MARGIN,
    TRACKING_REID_VALIDATE_EVERY_N_FRAMES,
    TRACKING_REID_MAX_CANDIDATES,
    TRACKING_REFERENCE_UPDATE_RATE,
    TRACKING_SIZE_WEIGHT,
    TRACKING_TEAM_COLOR_WEIGHT,
    TRACKING_USE_NEURAL_REID,
    TRACKING_ENGINE,
)
from app.services.model_registry import get_tracking_model
from app.services.object_detection import _normalize_class_name
from app.services.person_reidentification import neural_appearance_embedding
from app.services.team_color import extract_jersey_descriptor, jersey_similarity


def track_selected_player(
    video_id: str,
    segment_id: str,
    video_path: Path,
    fps: float,
    segment_start_time: float,
    segment_end_time: float,
    selection: dict[str, Any],
    anchors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Dispatch to the configured tracking engine.

    "tracklet" (default): detection cache + conservative tracklets + offline
    stitching through every user-confirmed anchor (bidirectional,
    camera-compensated, honest gaps).
    "greedy": legacy per-frame ByteTrack-follow path, kept for A/B comparison
    (single anchor only).
    """
    if TRACKING_ENGINE == "tracklet":
        from app.services.tracklets import track_selected_player_tracklets

        return track_selected_player_tracklets(
            video_id,
            segment_id,
            video_path,
            fps,
            segment_start_time,
            segment_end_time,
            selection,
            anchors=anchors,
        )
    return _track_selected_player_greedy(
        video_path, fps, segment_start_time, segment_end_time, selection
    )


def _intersection_over_union(first: dict[str, float], second: dict[str, float]) -> float:
    x1 = max(first["x1"], second["x1"])
    y1 = max(first["y1"], second["y1"])
    x2 = min(first["x2"], second["x2"])
    y2 = min(first["y2"], second["y2"])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first["x2"] - first["x1"]) * max(0.0, first["y2"] - first["y1"])
    second_area = max(0.0, second["x2"] - second["x1"]) * max(0.0, second["y2"] - second["y1"])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_center(bbox: dict[str, float]) -> np.ndarray:
    return np.array(
        [(bbox["x1"] + bbox["x2"]) / 2, (bbox["y1"] + bbox["y2"]) / 2],
        dtype=np.float32,
    )


def _bbox_area(bbox: dict[str, float]) -> float:
    return max(0.0, bbox["x2"] - bbox["x1"]) * max(0.0, bbox["y2"] - bbox["y1"])


def _shift_bbox(bbox: dict[str, float], displacement: np.ndarray) -> dict[str, float]:
    dx, dy = float(displacement[0]), float(displacement[1])
    return {
        "x1": bbox["x1"] + dx,
        "y1": bbox["y1"] + dy,
        "x2": bbox["x2"] + dx,
        "y2": bbox["y2"] + dy,
    }


def _clamp_bbox(bbox: dict[str, float], width: int, height: int) -> dict[str, float]:
    return {
        "x1": min(max(0.0, bbox["x1"]), float(width)),
        "y1": min(max(0.0, bbox["y1"]), float(height)),
        "x2": min(max(0.0, bbox["x2"]), float(width)),
        "y2": min(max(0.0, bbox["y2"]), float(height)),
    }


def _smooth_bbox(previous: dict[str, float], current: dict[str, float]) -> dict[str, float]:
    alpha = TRACKING_BOX_SMOOTHING_ALPHA
    return {
        key: (1 - alpha) * previous[key] + alpha * current[key]
        for key in ("x1", "y1", "x2", "y2")
    }


def _player_crop(frame: Any, bbox: dict[str, float]) -> np.ndarray | None:
    height, width = frame.shape[:2]
    box_width = bbox["x2"] - bbox["x1"]
    box_height = bbox["y2"] - bbox["y1"]
    x_padding = box_width * 0.05
    y_padding = box_height * 0.03
    x1 = max(0, int(bbox["x1"] - x_padding))
    y1 = max(0, int(bbox["y1"] - y_padding))
    x2 = min(width, int(bbox["x2"] + x_padding))
    y2 = min(height, int(bbox["y2"] + y_padding))
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2]


def _appearance_descriptor(frame: Any, bbox: dict[str, float]) -> np.ndarray | None:
    """Use neural appearance features, with color/texture as an offline fallback."""
    crop = _player_crop(frame, bbox)
    if crop is None:
        return None
    if TRACKING_USE_NEURAL_REID:
        neural_descriptor = neural_appearance_embedding(crop)
        if neural_descriptor is not None:
            return neural_descriptor
    resized = cv2.resize(crop, (32, 64), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    color_histogram = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256]).flatten()
    color_histogram /= float(np.linalg.norm(color_histogram) + 1e-8)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)
    texture_histogram, _ = np.histogram(
        angle,
        bins=12,
        range=(0, 360),
        weights=magnitude,
    )
    texture_histogram = texture_histogram.astype(np.float32)
    texture_histogram /= float(np.linalg.norm(texture_histogram) + 1e-8)

    descriptor = np.concatenate([color_histogram * 0.8, texture_histogram * 0.2]).astype(np.float32)
    descriptor /= float(np.linalg.norm(descriptor) + 1e-8)
    return descriptor


def _appearance_similarity(reference: np.ndarray | None, candidate: np.ndarray | None) -> float:
    if reference is None or candidate is None:
        return 0.0
    return float(np.clip(np.dot(reference, candidate), 0.0, 1.0))


def _position_similarity(predicted_bbox: dict[str, float], candidate_bbox: dict[str, float]) -> float:
    distance = float(np.linalg.norm(_bbox_center(predicted_bbox) - _bbox_center(candidate_bbox)))
    predicted_width = max(1.0, predicted_bbox["x2"] - predicted_bbox["x1"])
    predicted_height = max(1.0, predicted_bbox["y2"] - predicted_bbox["y1"])
    scale = max(predicted_width, predicted_height) * 2.0
    return float(np.exp(-distance / scale))


def _size_similarity(reference_bbox: dict[str, float], candidate_bbox: dict[str, float]) -> float:
    first_area = _bbox_area(reference_bbox)
    second_area = _bbox_area(candidate_bbox)
    if first_area <= 0 or second_area <= 0:
        return 0.0
    return min(first_area, second_area) / max(first_area, second_area)


def _candidate_match_score(
    candidate: dict[str, Any],
    reference_descriptor: np.ndarray | None,
    reference_jersey_descriptor: np.ndarray | None,
    predicted_bbox: dict[str, float],
    frame: Any,
) -> tuple[float, float, float, float, np.ndarray | None]:
    descriptor = _appearance_descriptor(frame, candidate["bbox"])
    appearance = _appearance_similarity(reference_descriptor, descriptor)
    position = _position_similarity(predicted_bbox, candidate["bbox"])
    overlap = _intersection_over_union(predicted_bbox, candidate["bbox"])
    size = _size_similarity(predicted_bbox, candidate["bbox"])
    base_score = (
        TRACKING_APPEARANCE_WEIGHT * appearance
        + TRACKING_POSITION_WEIGHT * position
        + TRACKING_IOU_WEIGHT * overlap
        + TRACKING_SIZE_WEIGHT * size
    )
    candidate_jersey_descriptor, _ = extract_jersey_descriptor(frame, candidate["bbox"])
    team_color = jersey_similarity(reference_jersey_descriptor, candidate_jersey_descriptor)
    score = base_score
    if reference_jersey_descriptor is not None and candidate_jersey_descriptor is not None:
        score = (1 - TRACKING_TEAM_COLOR_WEIGHT) * base_score + TRACKING_TEAM_COLOR_WEIGHT * team_color
    return score, appearance, team_color, position, descriptor


def _candidate_is_plausible(
    score: float,
    appearance: float,
    position: float,
    missing_frames: int,
) -> bool:
    # Identity evidence must become stronger, not weaker, as an occlusion grows.
    adaptive_threshold = min(
        TRACKING_REASSOCIATION_THRESHOLD + min(missing_frames, 15) * 0.006,
        0.64,
    )
    minimum_appearance = TRACKING_MIN_APPEARANCE_SIMILARITY
    if missing_frames >= TRACKING_LONG_GAP_FRAMES:
        minimum_appearance = max(minimum_appearance, TRACKING_LONG_GAP_APPEARANCE)
    return (
        score >= adaptive_threshold
        and appearance >= minimum_appearance
        and (position >= 0.12 or appearance >= 0.65)
    )


def _within_reassociation_radius(
    candidate_bbox: dict[str, float],
    last_observed_bbox: dict[str, float],
    missing_source_frames: int,
    frame_width: int,
    frame_height: int,
) -> bool:
    distance = float(
        np.linalg.norm(_bbox_center(candidate_bbox) - _bbox_center(last_observed_bbox))
    )
    diagonal = float(np.hypot(frame_width, frame_height))
    observed_height = max(1.0, last_observed_bbox["y2"] - last_observed_bbox["y1"])
    base_radius = max(observed_height * 3.0, diagonal * 0.05)
    growth = min(missing_source_frames, 45) * max(1.0, observed_height * 0.08)
    radius = min(
        base_radius + growth,
        diagonal * TRACKING_MAX_REASSOCIATION_FRAME_FRACTION,
    )
    return distance <= radius


def _trackable_class_ids(model: Any) -> list[int]:
    return [
        class_id
        for class_id, class_name in model.names.items()
        if _normalize_class_name(str(class_name)) in {"player", "goalkeeper"}
    ]


def _shortlist_candidates(
    candidates: list[dict[str, Any]],
    predicted_bbox: dict[str, float],
) -> list[dict[str, Any]]:
    """Avoid expensive neural embeddings for obviously distant candidates."""
    return sorted(
        candidates,
        key=lambda candidate: (
            0.65 * _position_similarity(predicted_bbox, candidate["bbox"])
            + 0.25 * _intersection_over_union(predicted_bbox, candidate["bbox"])
            + 0.10 * _size_similarity(predicted_bbox, candidate["bbox"])
        ),
        reverse=True,
    )[:TRACKING_REID_MAX_CANDIDATES]


def _track_selected_player_greedy(
    video_path: Path,
    fps: float,
    segment_start_time: float,
    segment_end_time: float,
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Legacy greedy engine: follow one ByteTrack ID frame-by-frame.

    Known failure modes (why the tracklet engine replaced it as default):
    identity switches at crossings, permanent loss after MAX_MISSING_FRAMES,
    and predicted boxes drifting off-target during camera pans."""
    processing_started = perf_counter()
    model = get_tracking_model()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise HTTPException(status_code=422, detail="Could not open uploaded video")

    start_frame = round(selection["selected_at_time"] * fps)
    end_frame = round(segment_end_time * fps)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    selected_bbox = selection["bbox"]
    selected_frame = capture.read()[1]
    if selected_frame is None:
        raise HTTPException(status_code=422, detail="Could not read selected player frame")
    reference_descriptor = _appearance_descriptor(selected_frame, selected_bbox)
    selected_jersey_values = selection.get("jersey_descriptor")
    reference_jersey_descriptor = (
        np.asarray(selected_jersey_values, dtype=np.float32)
        if selected_jersey_values
        else extract_jersey_descriptor(selected_frame, selected_bbox)[0]
    )
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    focused_track_id: int | None = None
    last_focused_bbox = selected_bbox
    last_observed_bbox = selected_bbox
    previous_focused_center: np.ndarray | None = None
    previous_focused_frame: int | None = None
    velocity = np.zeros(2, dtype=np.float32)
    missing_frames = 0
    missing_source_frames = 0
    pending_track_id: int | None = None
    pending_track_frames = 0
    samples: list[dict[str, Any]] = []
    source_frames = 0
    inference_frames = 0
    class_ids = _trackable_class_ids(model)

    try:
        for frame_number in range(start_frame, end_frame):
            success, frame = capture.read()
            if not success:
                break
            source_frames += 1

            should_run_inference = (
                focused_track_id is None
                or (frame_number - start_frame) % TRACKING_FRAME_STRIDE == 0
            )
            if not should_run_inference:
                displacement = velocity if missing_source_frames < TRACKING_MAX_PREDICTION_FRAMES else np.zeros(2)
                predicted_bbox = _clamp_bbox(
                    _shift_bbox(last_focused_bbox, displacement),
                    frame_width,
                    frame_height,
                )
                last_focused_bbox = predicted_bbox
                missing_source_frames += 1
                timestamp = frame_number / fps
                samples.append(
                    {
                        "frame_number": frame_number,
                        "timestamp_seconds": round(timestamp, 3),
                        "clip_time_seconds": round(timestamp - segment_start_time, 3),
                        "bbox": {key: round(value, 2) for key, value in predicted_bbox.items()},
                        "confidence": 0.0,
                        "predicted": True,
                        "state": "interpolated",
                    }
                )
                continue

            inference_frames += 1
            try:
                result = model.track(
                    frame,
                    persist=True,
                    tracker=TRACKER_CONFIG,
                    conf=TRACKING_CONFIDENCE_THRESHOLD,
                    imgsz=TRACKING_IMAGE_SIZE,
                    classes=class_ids or None,
                    verbose=False,
                )[0]
            except Exception as error:
                raise HTTPException(status_code=503, detail=f"ByteTrack inference failed: {error}") from error

            candidates = []
            if result.boxes.id is not None:
                for box in result.boxes:
                    bbox_values = [float(value) for value in box.xyxy[0].tolist()]
                    candidates.append(
                        {
                            "track_id": int(box.id.item()),
                            "bbox": dict(zip(("x1", "y1", "x2", "y2"), bbox_values)),
                            "confidence": float(box.conf.item()),
                        }
                    )

            if focused_track_id is None and candidates:
                initial_candidates = []
                for candidate in _shortlist_candidates(candidates, selected_bbox):
                    score, appearance, _, position, _ = _candidate_match_score(
                        candidate,
                        reference_descriptor,
                        reference_jersey_descriptor,
                        selected_bbox,
                        frame,
                    )
                    overlap = _intersection_over_union(selected_bbox, candidate["bbox"])
                    initial_candidates.append(
                        (score + 0.35 * overlap, appearance, position, candidate)
                    )
                initial_score, initial_appearance, initial_position, best_candidate = max(
                    initial_candidates,
                    key=lambda scored: scored[0],
                )
                if (
                    _intersection_over_union(selected_bbox, best_candidate["bbox"]) > 0.05
                    or _candidate_is_plausible(
                        initial_score,
                        initial_appearance,
                        initial_position,
                        0,
                    )
                ):
                    focused_track_id = best_candidate["track_id"]

            focused = next(
                (candidate for candidate in candidates if candidate["track_id"] == focused_track_id),
                None,
            )
            predicted_bbox = _shift_bbox(last_focused_bbox, velocity)
            focused_descriptor: np.ndarray | None = None
            if focused is not None:
                should_validate_appearance = (
                    frame_number - start_frame
                ) % TRACKING_REID_VALIDATE_EVERY_N_FRAMES == 0
                if should_validate_appearance:
                    direct_score, direct_appearance, _, direct_position, direct_descriptor = _candidate_match_score(
                        focused,
                        reference_descriptor,
                        reference_jersey_descriptor,
                        predicted_bbox,
                        frame,
                    )
                    if not _candidate_is_plausible(
                        direct_score,
                        direct_appearance,
                        direct_position,
                        missing_source_frames,
                    ):
                        focused = None
                    else:
                        focused_descriptor = direct_descriptor

            if focused is None and candidates:
                scored_candidates = []
                for candidate in _shortlist_candidates(candidates, predicted_bbox):
                    score, appearance, team_color, position, descriptor = _candidate_match_score(
                        candidate,
                        reference_descriptor,
                        reference_jersey_descriptor,
                        predicted_bbox,
                        frame,
                    )
                    scored_candidates.append(
                        (score, appearance, team_color, position, descriptor, candidate)
                    )
                scored_candidates.sort(key=lambda scored: scored[0], reverse=True)
                score, appearance, _, position, descriptor, reassociated = scored_candidates[0]
                runner_up_score = scored_candidates[1][0] if len(scored_candidates) > 1 else 0.0
                has_clear_winner = score - runner_up_score >= TRACKING_REASSOCIATION_MARGIN
                is_spatially_plausible = _within_reassociation_radius(
                    reassociated["bbox"],
                    last_observed_bbox,
                    missing_source_frames,
                    frame_width,
                    frame_height,
                )
                if (
                    _candidate_is_plausible(
                        score,
                        appearance,
                        position,
                        missing_source_frames,
                    )
                    and has_clear_winner
                    and is_spatially_plausible
                ):
                    if pending_track_id == reassociated["track_id"]:
                        pending_track_frames += 1
                    else:
                        pending_track_id = reassociated["track_id"]
                        pending_track_frames = 1
                    if (
                        reassociated["track_id"] == focused_track_id
                        or pending_track_frames >= TRACKING_REASSOCIATION_CONFIRM_FRAMES
                    ):
                        focused_track_id = reassociated["track_id"]
                        focused = reassociated
                        focused_descriptor = descriptor
                        pending_track_id = None
                        pending_track_frames = 0
                else:
                    pending_track_id = None
                    pending_track_frames = 0
            if focused is None:
                missing_frames += 1
                missing_source_frames += 1
                if focused_track_id is not None and missing_frames <= TRACKING_MAX_MISSING_FRAMES:
                    displacement = (
                        velocity
                        if missing_source_frames <= TRACKING_MAX_PREDICTION_FRAMES
                        else np.zeros(2)
                    )
                    predicted_bbox = _clamp_bbox(
                        _shift_bbox(last_focused_bbox, displacement),
                        frame_width,
                        frame_height,
                    )
                    last_focused_bbox = predicted_bbox
                    timestamp = frame_number / fps
                    samples.append(
                        {
                            "frame_number": frame_number,
                            "timestamp_seconds": round(timestamp, 3),
                            "clip_time_seconds": round(timestamp - segment_start_time, 3),
                            "bbox": {key: round(value, 2) for key, value in predicted_bbox.items()},
                            "confidence": 0.0,
                            "predicted": True,
                            "state": "interpolated",
                        }
                    )
                if focused_track_id is not None and missing_frames > TRACKING_MAX_MISSING_FRAMES:
                    break
                continue

            missing_frames = 0
            missing_source_frames = 0
            pending_track_id = None
            pending_track_frames = 0
            current_center = _bbox_center(focused["bbox"])
            if previous_focused_center is not None and previous_focused_frame is not None:
                elapsed_frames = max(1, frame_number - previous_focused_frame)
                measured_velocity = (current_center - previous_focused_center) / elapsed_frames
                velocity = (0.7 * velocity) + (0.3 * measured_velocity)
            previous_focused_center = current_center
            previous_focused_frame = frame_number
            focused["bbox"] = _smooth_bbox(last_focused_bbox, focused["bbox"])
            last_focused_bbox = focused["bbox"]
            last_observed_bbox = focused["bbox"]
            if focused_descriptor is not None and reference_descriptor is not None:
                reference_descriptor = (
                    (1 - TRACKING_REFERENCE_UPDATE_RATE) * reference_descriptor
                    + TRACKING_REFERENCE_UPDATE_RATE * focused_descriptor
                )
                reference_descriptor /= float(np.linalg.norm(reference_descriptor) + 1e-8)
            timestamp = frame_number / fps
            samples.append(
                {
                    "frame_number": frame_number,
                    "timestamp_seconds": round(timestamp, 3),
                    "clip_time_seconds": round(timestamp - segment_start_time, 3),
                    "bbox": {key: round(value, 2) for key, value in focused["bbox"].items()},
                    "confidence": round(focused["confidence"], 4),
                    "predicted": False,
                    "state": "tracked",
                }
            )
    finally:
        capture.release()

    if focused_track_id is None or not samples:
        raise HTTPException(status_code=422, detail="Could not associate the selected player with a ByteTrack ID")

    return {
        "status": "completed",
        "tracker": "bytetrack",
        "engine": "greedy",
        "track_id": focused_track_id,
        "start_time": samples[0]["timestamp_seconds"],
        "end_time": samples[-1]["timestamp_seconds"],
        "frame_width": frame_width,
        "frame_height": frame_height,
        "samples": samples,
        "source_frames": source_frames,
        "inference_frames": inference_frames,
        "frame_stride": TRACKING_FRAME_STRIDE,
        "processing_seconds": round(perf_counter() - processing_started, 3),
    }
