from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException

from app.core.config import (
    CAMERA_MOTION_ENABLED,
    DETECTION_CACHE_ENABLED,
    DETECTION_CACHE_STRIDE,
    DETECTIONS_DIR,
    PITCH_MASK_ENABLED,
    TRACKING_BATCH_SIZE,
    TRACKING_CONFIDENCE_THRESHOLD,
    TRACKING_IMAGE_SIZE,
    TRACKING_MODEL_PATH,
)
from app.services.appearance import cheap_appearance_descriptor
from app.services.camera_motion import IDENTITY_AFFINE, downscale_gray, estimate_global_affine
from app.services.model_registry import get_tracking_model, model_cache_key
from app.services.object_detection import _normalize_class_name
from app.services.pitch_mask import feet_on_pitch, grass_hull
from app.services.reid_embedding import appearance_backend, embed_crops
from app.services.team_color import extract_jersey_descriptor

ROLES = ("player", "goalkeeper", "referee", "ball")
ROLE_INDEX = {role: index for index, role in enumerate(ROLES)}
JERSEY_DESCRIPTOR_SIZE = 24  # 18 hue bins + 6 value bins
# v3: det_appearance may hold OSNet ReID embeddings (512-d) instead of HSV (300-d).
# v4: camera model upgraded from translation-only (camera_shift (F,2)) to full
#     cumulative 2x3 affines (camera_affine (F,2,3)) + per-step confidence.
#     Translation-only compensation is wrong under zoom — the drift bug.
CACHE_FORMAT_VERSION = 4


def _as_3x3(affine: np.ndarray) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :] = affine
    return matrix


@dataclass
class DetectionCache:
    """Flat per-detection arrays plus per-sampled-frame camera transforms.

    frames[i] is the source frame number of sampled frame i.
    camera_affine[i] is the cumulative 2x3 similarity mapping frame-0 screen
    coordinates to frame-i screen coordinates (full resolution px).
    camera_confidence[i] is the RANSAC inlier ratio of the step i-1 -> i
    (1.0 for frame 0; 0.0 = registration failed, treat the span as untrusted).
    Detection arrays are parallel; det_frame_index[j] indexes into frames.
    """

    header: dict[str, Any]
    frames: np.ndarray            # (F,) int32 source frame numbers
    camera_affine: np.ndarray     # (F, 2, 3) float32 cumulative frame0 -> frame i
    camera_confidence: np.ndarray  # (F,) float32 per-step registration confidence
    det_frame_index: np.ndarray   # (N,) int32 -> index into frames
    det_bbox: np.ndarray          # (N, 4) float32 x1 y1 x2 y2
    det_confidence: np.ndarray    # (N,) float32
    det_role: np.ndarray          # (N,) int8 index into ROLES
    det_on_pitch: np.ndarray      # (N,) bool
    det_appearance: np.ndarray    # (N, D) float16; D per header appearance_dim (OSNet 512 / HSV 300); NaN row = unavailable
    det_jersey: np.ndarray        # (N, 24) float16, NaN row = unavailable

    @property
    def stride(self) -> int:
        return int(self.header["stride"])

    def frame_position(self, frame_number: int) -> int:
        """Index of the nearest sampled frame."""
        position = int(np.searchsorted(self.frames, frame_number))
        if position >= len(self.frames):
            return len(self.frames) - 1
        if position > 0 and abs(int(self.frames[position - 1]) - frame_number) <= abs(
            int(self.frames[position]) - frame_number
        ):
            return position - 1
        return position

    # ---- Full-affine camera geometry (BoT-SORT GMC style) ----

    def transform_between(self, from_frame: int, to_frame: int) -> np.ndarray:
        """2x3 affine mapping from_frame screen coords -> to_frame screen coords.

        A static background point at p in from_frame appears at M·p in
        to_frame. This is the correct compensation under pan AND zoom;
        translation-only differencing is wrong the moment scale != 1.
        """
        source = _as_3x3(self.camera_affine[self.frame_position(from_frame)])
        target = _as_3x3(self.camera_affine[self.frame_position(to_frame)])
        return (target @ np.linalg.inv(source))[:2, :].astype(np.float32)

    def transform_point(self, point: np.ndarray, from_frame: int, to_frame: int) -> np.ndarray:
        matrix = self.transform_between(from_frame, to_frame)
        return (matrix[:, :2] @ np.asarray(point, dtype=np.float32) + matrix[:, 2]).astype(np.float32)

    def transform_box(self, box: np.ndarray, from_frame: int, to_frame: int) -> np.ndarray:
        """Warp a box between frames: center maps through the affine, width and
        height scale with the camera zoom (rotation is negligible for handheld
        footage and ignored for the box shape)."""
        matrix = self.transform_between(from_frame, to_frame)
        scale = float(np.sqrt(abs(np.linalg.det(matrix[:, :2])))) or 1.0
        center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)
        new_center = matrix[:, :2] @ center + matrix[:, 2]
        half = np.array([(box[2] - box[0]) / 2, (box[3] - box[1]) / 2], dtype=np.float32) * scale
        return np.array(
            [new_center[0] - half[0], new_center[1] - half[1], new_center[0] + half[0], new_center[1] + half[1]],
            dtype=np.float32,
        )

    def scale_between(self, from_frame: int, to_frame: int) -> float:
        matrix = self.transform_between(from_frame, to_frame)
        return float(np.sqrt(abs(np.linalg.det(matrix[:, :2])))) or 1.0

    def world_point(self, point: np.ndarray, frame_position: int) -> np.ndarray:
        """Map a screen point at sampled index frame_position into frame-0
        ("world") coordinates — pan/zoom-invariant, for static-object tests."""
        matrix = np.linalg.inv(_as_3x3(self.camera_affine[frame_position]))
        return (matrix[:2, :2] @ np.asarray(point, dtype=np.float64) + matrix[:2, 2]).astype(np.float32)

    def min_camera_confidence(self, from_frame: int, to_frame: int) -> float:
        """Weakest registration step inside a span. Low = the camera model is
        guesswork there; associations across it must tighten, not loosen."""
        low = min(self.frame_position(from_frame), self.frame_position(to_frame))
        high = max(self.frame_position(from_frame), self.frame_position(to_frame))
        if high <= low:
            return float(self.camera_confidence[low])
        return float(np.min(self.camera_confidence[low + 1 : high + 1]))

    def detections_at(self, frame_position: int) -> np.ndarray:
        """Indices of detections belonging to sampled frame at frame_position."""
        return np.nonzero(self.det_frame_index == frame_position)[0]


def cache_path(video_id: str, segment_id: str) -> Path:
    key = model_cache_key(TRACKING_MODEL_PATH)
    return DETECTIONS_DIR / video_id / f"{segment_id}_{key}.npz"


def _header(fps: float, start_frame: int, end_frame: int, width: int, height: int) -> dict[str, Any]:
    backend_name, descriptor_dim, backend_key = appearance_backend()
    return {
        "version": CACHE_FORMAT_VERSION,
        "model": Path(TRACKING_MODEL_PATH).name,
        "model_key": model_cache_key(TRACKING_MODEL_PATH),
        "imgsz": TRACKING_IMAGE_SIZE,
        "conf": TRACKING_CONFIDENCE_THRESHOLD,
        "stride": DETECTION_CACHE_STRIDE,
        "fps": fps,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_width": width,
        "frame_height": height,
        "camera_motion": CAMERA_MOTION_ENABLED,
        "pitch_mask": PITCH_MASK_ENABLED,
        "appearance_backend": backend_name,
        "appearance_dim": descriptor_dim,
        "reid_key": backend_key,
    }


def _camera_arrays(data: Any, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Read v4 affine arrays, or synthesize them from a legacy v<=3 translation
    cache (identity linear part) so read-only consumers keep working."""
    if "camera_affine" in getattr(data, "files", []):
        return data["camera_affine"], data["camera_confidence"]
    affines = np.tile(IDENTITY_AFFINE, (frame_count, 1, 1)).astype(np.float32)
    if "camera_shift" in getattr(data, "files", []):
        affines[:, :, 2] = data["camera_shift"].astype(np.float32)
    return affines, np.ones(frame_count, dtype=np.float32)


def load_cache_unchecked(path: Path) -> DetectionCache | None:
    """Load a cache without header validation — for read-only consumers
    (analysis, debug overlays) that should use whatever detections exist."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            frames = data["frames"]
            camera_affine, camera_confidence = _camera_arrays(data, len(frames))
            return DetectionCache(
                header=json.loads(str(data["header"])),
                frames=frames,
                camera_affine=camera_affine,
                camera_confidence=camera_confidence,
                det_frame_index=data["det_frame_index"],
                det_bbox=data["det_bbox"],
                det_confidence=data["det_confidence"],
                det_role=data["det_role"],
                det_on_pitch=data["det_on_pitch"],
                det_appearance=data["det_appearance"],
                det_jersey=data["det_jersey"],
            )
    except Exception:
        return None


def load_cache(path: Path, expected_header: dict[str, Any]) -> DetectionCache | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            header = json.loads(str(data["header"]))
            invariants = (
                "version", "model_key", "imgsz", "conf", "stride",
                "start_frame", "end_frame", "reid_key",
            )
            if any(header.get(key) != expected_header.get(key) for key in invariants):
                return None
            return DetectionCache(
                header=header,
                frames=data["frames"],
                camera_affine=data["camera_affine"],
                camera_confidence=data["camera_confidence"],
                det_frame_index=data["det_frame_index"],
                det_bbox=data["det_bbox"],
                det_confidence=data["det_confidence"],
                det_role=data["det_role"],
                det_on_pitch=data["det_on_pitch"],
                det_appearance=data["det_appearance"],
                det_jersey=data["det_jersey"],
            )
    except Exception:
        return None  # corrupt cache: rebuild


def _save_cache(path: Path, cache: DetectionCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        header=np.array(json.dumps(cache.header)),
        frames=cache.frames,
        camera_affine=cache.camera_affine,
        camera_confidence=cache.camera_confidence,
        det_frame_index=cache.det_frame_index,
        det_bbox=cache.det_bbox,
        det_confidence=cache.det_confidence,
        det_role=cache.det_role,
        det_on_pitch=cache.det_on_pitch,
        det_appearance=cache.det_appearance,
        det_jersey=cache.det_jersey,
    )


def _detect_batch(model: Any, frames: list[np.ndarray], role_lookup: dict[int, str | None]) -> list[list[dict[str, Any]]]:
    try:
        results = model.predict(
            frames,
            conf=TRACKING_CONFIDENCE_THRESHOLD,
            imgsz=TRACKING_IMAGE_SIZE,
            verbose=False,
        )
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"YOLO inference failed: {error}") from error

    batch_detections: list[list[dict[str, Any]]] = []
    for result in results:
        detections = []
        for box in result.boxes:
            role = role_lookup.get(int(box.cls.item()))
            if role is None:
                continue
            values = [float(value) for value in box.xyxy[0].tolist()]
            detections.append(
                {
                    "bbox": dict(zip(("x1", "y1", "x2", "y2"), values)),
                    "confidence": float(box.conf.item()),
                    "role": role,
                }
            )
        batch_detections.append(detections)
    return batch_detections


def build_cache(
    video_path: Path,
    fps: float,
    start_frame: int,
    end_frame: int,
) -> DetectionCache:
    """Single sequential pass: batched YOLO + camera shift + pitch mask + descriptors.

    Sequential capture.read() with skipping — never capture.set() per frame,
    which forces keyframe seeks on most codecs.
    """
    model = get_tracking_model()
    role_lookup = {
        class_id: _normalize_class_name(str(class_name))
        for class_id, class_name in model.names.items()
    }

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise HTTPException(status_code=422, detail="Could not open uploaded video")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    header = _header(fps, start_frame, end_frame, frame_width, frame_height)
    stride = header["stride"]
    # Appearance backend chosen ONCE per build so det_appearance stays homogeneous.
    backend_name, descriptor_size, _ = appearance_backend()
    use_osnet = backend_name == "osnet"

    frames_list: list[int] = []
    affine_list: list[np.ndarray] = []
    confidence_list: list[float] = []
    det_rows: list[dict[str, Any]] = []

    pending_frames: list[np.ndarray] = []
    pending_numbers: list[int] = []
    previous_gray: np.ndarray | None = None
    previous_boxes: list[dict[str, float]] = []
    cumulative_affine = np.eye(3, dtype=np.float64)  # frame0 -> current, composed 3x3

    def flush_batch() -> None:
        nonlocal previous_gray, previous_boxes, cumulative_affine
        if not pending_frames:
            return
        osnet_jobs: list[tuple[int, np.ndarray]] = []
        batch_detections = _detect_batch(model, pending_frames, role_lookup)
        for frame, frame_number, detections in zip(pending_frames, pending_numbers, batch_detections):
            gray, scale = downscale_gray(frame)
            step_confidence = 1.0
            if CAMERA_MOTION_ENABLED and previous_gray is not None:
                step_affine, step_confidence = estimate_global_affine(
                    previous_gray, gray, scale, previous_boxes
                )
                cumulative_affine = _as_3x3(step_affine) @ cumulative_affine
            hull = None
            if PITCH_MASK_ENABLED:
                small = cv2.resize(
                    frame,
                    (gray.shape[1], gray.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
                hull = grass_hull(small)

            frame_position = len(frames_list)
            frames_list.append(frame_number)
            affine_list.append(cumulative_affine[:2, :].astype(np.float32).copy())
            confidence_list.append(float(step_confidence))

            for detection in detections:
                bbox = detection["bbox"]
                role = detection["role"]
                on_pitch = True
                if PITCH_MASK_ENABLED and role in {"player", "goalkeeper", "referee"}:
                    on_pitch = feet_on_pitch(bbox, hull, scale, gray.shape[0])
                appearance = None
                jersey = None
                if role in {"player", "goalkeeper"}:
                    if bbox["x2"] - bbox["x1"] >= 12:
                        jersey = extract_jersey_descriptor(frame, bbox)[0]
                    if use_osnet:
                        # Deferred to one batched OSNet forward pass per flush.
                        x1 = max(0, int(bbox["x1"]))
                        y1 = max(0, int(bbox["y1"]))
                        x2 = max(x1 + 1, int(bbox["x2"]))
                        y2 = max(y1 + 1, int(bbox["y2"]))
                        osnet_jobs.append((len(det_rows), frame[y1:y2, x1:x2].copy()))
                    else:
                        appearance = cheap_appearance_descriptor(frame, bbox)
                det_rows.append(
                    {
                        "frame_index": frame_position,
                        "bbox": [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]],
                        "confidence": detection["confidence"],
                        "role": ROLE_INDEX[role],
                        "on_pitch": on_pitch,
                        "appearance": appearance,
                        "jersey": jersey,
                    }
                )

            previous_gray = gray
            previous_boxes = [
                detection["bbox"]
                for detection in detections
                if detection["role"] in {"player", "goalkeeper", "referee"}
            ]
        if use_osnet and osnet_jobs:
            embeddings = embed_crops([crop for _, crop in osnet_jobs])
            for (row_index, _), embedding in zip(osnet_jobs, embeddings):
                det_rows[row_index]["appearance"] = embedding
        pending_frames.clear()
        pending_numbers.clear()

    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_number in range(start_frame, end_frame):
            success, frame = capture.read()
            if not success:
                break
            if (frame_number - start_frame) % stride != 0:
                continue
            pending_frames.append(frame)
            pending_numbers.append(frame_number)
            if len(pending_frames) >= TRACKING_BATCH_SIZE:
                flush_batch()
        flush_batch()
    finally:
        capture.release()

    if not frames_list:
        raise HTTPException(status_code=422, detail="No frames could be read for this segment")

    count = len(det_rows)
    det_appearance = np.full((count, descriptor_size), np.nan, dtype=np.float16)
    det_jersey = np.full((count, JERSEY_DESCRIPTOR_SIZE), np.nan, dtype=np.float16)
    for row_index, row in enumerate(det_rows):
        if row["appearance"] is not None and len(row["appearance"]) == descriptor_size:
            det_appearance[row_index] = row["appearance"].astype(np.float16)
        if row["jersey"] is not None and len(row["jersey"]) == JERSEY_DESCRIPTOR_SIZE:
            det_jersey[row_index] = row["jersey"].astype(np.float16)

    cache = DetectionCache(
        header=header,
        frames=np.array(frames_list, dtype=np.int32),
        camera_affine=np.stack(affine_list).astype(np.float32),
        camera_confidence=np.array(confidence_list, dtype=np.float32),
        det_frame_index=np.array([row["frame_index"] for row in det_rows], dtype=np.int32),
        det_bbox=(
            np.array([row["bbox"] for row in det_rows], dtype=np.float32)
            if det_rows
            else np.zeros((0, 4), dtype=np.float32)
        ),
        det_confidence=np.array([row["confidence"] for row in det_rows], dtype=np.float32),
        det_role=np.array([row["role"] for row in det_rows], dtype=np.int8),
        det_on_pitch=np.array([row["on_pitch"] for row in det_rows], dtype=bool),
        det_appearance=det_appearance,
        det_jersey=det_jersey,
    )
    return cache


def get_or_build_cache(
    video_id: str,
    segment_id: str,
    video_path: Path,
    fps: float,
    start_frame: int,
    end_frame: int,
) -> tuple[DetectionCache, bool]:
    """Returns (cache, was_cache_hit). Detection runs at most once per segment+model."""
    path = cache_path(video_id, segment_id)
    frame_width_height_unknown_header = _header(fps, start_frame, end_frame, 0, 0)
    if DETECTION_CACHE_ENABLED:
        cached = load_cache(path, frame_width_height_unknown_header)
        if cached is not None:
            return cached, True
    cache = build_cache(video_path, fps, start_frame, end_frame)
    if DETECTION_CACHE_ENABLED:
        _save_cache(path, cache)
    return cache, False
