from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from app.core.config import (
    MIN_SEGMENT_SECONDS,
    SEGMENT_START_TRIM_SECONDS,
    SCENE_MOTION_CUT_MAX_INLIER_RATIO,
    SCENE_MOTION_CUT_MIN_HSV_DIFF,
    SCENE_SECOND_PASS_DOWNSCALE_WIDTH,
    SCENE_SECOND_PASS_ENABLED,
    SCENE_COMBINED_DIFF_THRESHOLD,
    SCENE_DETECTION_METHOD,
    SCENE_DIFF_THRESHOLD,
    SCENE_EDGE_DIFF_THRESHOLD,
    SCENE_EDGE_DIFF_WEIGHT,
    SCENE_HISTOGRAM_GRID_COLUMNS,
    SCENE_HISTOGRAM_GRID_ROWS,
    SCENE_HSV_DIFF_WEIGHT,
    SCENE_OBJECT_LAYOUT_DIFF_THRESHOLD,
    SCENE_OBJECT_LAYOUT_ENABLED,
    SCENE_OBJECT_LAYOUT_MIN_SEGMENT_SECONDS,
    SCENE_OBJECT_LAYOUT_SAMPLE_EVERY_N_FRAMES,
    SCENE_REFINE_WINDOW_FRAMES,
    SCENE_SAMPLE_EVERY_N_FRAMES,
    SCENE_STRONG_CUT_MIN_EDGE_DIFF,
    SCENE_STRONG_HSV_DIFF_THRESHOLD,
)
from app.services.object_detection import object_layout_difference, object_layout_signature
from app.services.transnetv2_detection import detect_transnetv2_cut_frames


SceneProgressCallback = Callable[[float, str, int | None, int | None], None]


@dataclass(frozen=True)
class SceneSegment:
    start_frame: int
    end_frame: int
    # Real presentation timestamps (seconds) when the decoder provides reliable
    # ones — immune to variable-frame-rate drift that frame/fps conversion has.
    start_time_seconds: float | None = None
    end_time_seconds: float | None = None


def _spatial_hsv_histograms(frame: np.ndarray) -> list[np.ndarray]:
    """Describe color distribution separately in each region of the frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    histograms: list[np.ndarray] = []

    for row in range(SCENE_HISTOGRAM_GRID_ROWS):
        y1 = row * height // SCENE_HISTOGRAM_GRID_ROWS
        y2 = (row + 1) * height // SCENE_HISTOGRAM_GRID_ROWS
        for column in range(SCENE_HISTOGRAM_GRID_COLUMNS):
            x1 = column * width // SCENE_HISTOGRAM_GRID_COLUMNS
            x2 = (column + 1) * width // SCENE_HISTOGRAM_GRID_COLUMNS
            region = hsv[y1:y2, x1:x2]
            histogram = cv2.calcHist([region], [0, 1], None, [30, 32], [0, 180, 0, 256])
            histograms.append(cv2.normalize(histogram, histogram).flatten())

    return histograms


def _spatial_histogram_difference(
    previous_histograms: list[np.ndarray],
    current_histograms: list[np.ndarray],
) -> float:
    region_differences = [
        cv2.compareHist(previous, current, cv2.HISTCMP_BHATTACHARYYA)
        for previous, current in zip(previous_histograms, current_histograms)
    ]
    return float(np.mean(region_differences))


def _edge_grid_signature(frame: np.ndarray) -> np.ndarray:
    """Summarize edge density by region to catch structural scene changes."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 80, 160)
    height, width = edges.shape[:2]
    densities = []

    for row in range(SCENE_HISTOGRAM_GRID_ROWS):
        y1 = row * height // SCENE_HISTOGRAM_GRID_ROWS
        y2 = (row + 1) * height // SCENE_HISTOGRAM_GRID_ROWS
        for column in range(SCENE_HISTOGRAM_GRID_COLUMNS):
            x1 = column * width // SCENE_HISTOGRAM_GRID_COLUMNS
            x2 = (column + 1) * width // SCENE_HISTOGRAM_GRID_COLUMNS
            region = edges[y1:y2, x1:x2]
            densities.append(float(np.count_nonzero(region)) / region.size)

    return np.array(densities, dtype=np.float32)


def _edge_signature_difference(
    previous_signature: np.ndarray,
    current_signature: np.ndarray,
) -> float:
    return float(np.mean(np.abs(previous_signature - current_signature)))


def _scene_difference(previous_frame: np.ndarray, current_frame: np.ndarray) -> tuple[float, float, float]:
    hsv_difference = _spatial_histogram_difference(
        _spatial_hsv_histograms(previous_frame),
        _spatial_hsv_histograms(current_frame),
    )
    edge_difference = _edge_signature_difference(
        _edge_grid_signature(previous_frame),
        _edge_grid_signature(current_frame),
    )
    combined_difference = (
        SCENE_HSV_DIFF_WEIGHT * hsv_difference
        + SCENE_EDGE_DIFF_WEIGHT * edge_difference
    )
    return hsv_difference, edge_difference, combined_difference


def _is_likely_cut(hsv_difference: float, edge_difference: float, combined_difference: float) -> bool:
    strong_color_cut = (
        hsv_difference >= SCENE_STRONG_HSV_DIFF_THRESHOLD
        and edge_difference >= SCENE_STRONG_CUT_MIN_EDGE_DIFF
    )
    supported_cut = (
        hsv_difference >= SCENE_DIFF_THRESHOLD
        and edge_difference >= SCENE_EDGE_DIFF_THRESHOLD
        and combined_difference >= SCENE_COMBINED_DIFF_THRESHOLD
    )
    return strong_color_cut or supported_cut


def _read_frame(capture: cv2.VideoCapture, frame_number: int) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = capture.read()
    return frame if success else None


def _should_sample_object_layout(frame_number: int) -> bool:
    interval = max(SCENE_SAMPLE_EVERY_N_FRAMES, SCENE_OBJECT_LAYOUT_SAMPLE_EVERY_N_FRAMES)
    return frame_number % interval == 0


def _refine_cut_frame(
    capture: cv2.VideoCapture,
    candidate_frame: int,
    frame_count: int,
) -> int:
    """Move a sampled candidate to the strongest nearby frame-to-frame transition."""
    start_frame = max(1, candidate_frame - SCENE_REFINE_WINDOW_FRAMES)
    end_frame = min(frame_count - 1, candidate_frame + SCENE_REFINE_WINDOW_FRAMES)
    best_frame = candidate_frame
    best_difference = -1.0

    previous_frame = _read_frame(capture, start_frame - 1)
    if previous_frame is None:
        return candidate_frame

    for frame_number in range(start_frame, end_frame + 1):
        current_frame = _read_frame(capture, frame_number)
        if current_frame is None:
            continue
        hsv_difference, edge_difference, combined_difference = _scene_difference(
            previous_frame,
            current_frame,
        )
        if _is_likely_cut(hsv_difference, edge_difference, combined_difference):
            score = combined_difference
        else:
            score = combined_difference * 0.5
        if score > best_difference:
            best_difference = score
            best_frame = frame_number
        previous_frame = current_frame

    return best_frame


def _segments_from_cut_frames(
    cut_frames: list[int],
    frame_count: int,
    fps: float,
    timestamps: np.ndarray | None = None,
) -> list[SceneSegment]:
    minimum_frames = max(1, round(MIN_SEGMENT_SECONDS * fps))
    trim_frames = max(0, round(SEGMENT_START_TRIM_SECONDS * fps))
    accepted_cuts = [0]
    for frame_number in sorted(set(cut_frames)):
        if frame_number <= 0 or frame_number >= frame_count:
            continue
        if frame_number - accepted_cuts[-1] >= minimum_frames:
            accepted_cuts.append(frame_number)
    if len(accepted_cuts) > 1 and frame_count - accepted_cuts[-1] < minimum_frames:
        accepted_cuts.pop()

    def real_time(frame_number: int) -> float | None:
        if timestamps is None or len(timestamps) == 0:
            return None
        return float(timestamps[min(frame_number, len(timestamps) - 1)])

    segments: list[SceneSegment] = []
    for index, (start, end) in enumerate(zip(accepted_cuts, accepted_cuts[1:] + [frame_count])):
        # Safety trim: swallow any transition residue left after the cut.
        # Never trim the very first segment or below the minimum length.
        if index > 0 and trim_frames and (end - start) > minimum_frames + trim_frames:
            start += trim_frames
        segments.append(
            SceneSegment(
                start_frame=start,
                end_frame=end,
                start_time_seconds=real_time(start),
                end_time_seconds=real_time(end),
            )
        )
    return segments


def _detect_hybrid_scene_segments(
    video_path: str,
    fps: float,
    frame_count: int,
    progress_callback: SceneProgressCallback | None = None,
) -> list[SceneSegment]:
    """Fallback detector using spatial HSV, edges, object layout, and refinement."""
    if frame_count <= 0:
        return [SceneSegment(start_frame=0, end_frame=0)]

    capture = cv2.VideoCapture(video_path)
    minimum_frames = max(1, round(MIN_SEGMENT_SECONDS * fps))
    minimum_object_layout_frames = max(1, round(SCENE_OBJECT_LAYOUT_MIN_SEGMENT_SECONDS * fps))
    cut_frames = [0]
    previous_frame: np.ndarray | None = None
    previous_object_layout: np.ndarray | None = None

    for frame_number in range(0, frame_count, SCENE_SAMPLE_EVERY_N_FRAMES):
        frame = _read_frame(capture, frame_number)
        if frame is None:
            continue

        visual_cut_detected = False
        if previous_frame is not None:
            hsv_difference, edge_difference, combined_difference = _scene_difference(
                previous_frame,
                frame,
            )
            if _is_likely_cut(hsv_difference, edge_difference, combined_difference):
                refined_frame = _refine_cut_frame(capture, frame_number, frame_count)
                if refined_frame - cut_frames[-1] >= minimum_frames:
                    cut_frames.append(refined_frame)
                    visual_cut_detected = True

        if (
            SCENE_OBJECT_LAYOUT_ENABLED
            and not visual_cut_detected
            and _should_sample_object_layout(frame_number)
        ):
            current_object_layout = object_layout_signature(frame, frame_number, frame_number / fps)
            layout_difference = object_layout_difference(previous_object_layout, current_object_layout)
            if (
                layout_difference >= SCENE_OBJECT_LAYOUT_DIFF_THRESHOLD
                and frame_number - cut_frames[-1] >= minimum_object_layout_frames
            ):
                refined_frame = _refine_cut_frame(capture, frame_number, frame_count)
                if refined_frame - cut_frames[-1] >= minimum_object_layout_frames:
                    cut_frames.append(refined_frame)
            if current_object_layout is not None:
                previous_object_layout = current_object_layout
        previous_frame = frame
        if progress_callback:
            progress_callback(
                min(0.99, (frame_number + 1) / frame_count),
                f"Scanning frame {min(frame_number + 1, frame_count)} of {frame_count}",
                min(frame_number + 1, frame_count),
                frame_count,
            )

    capture.release()

    if len(cut_frames) > 1 and frame_count - cut_frames[-1] < minimum_frames:
        cut_frames.pop()

    segments = [
        SceneSegment(start_frame=start, end_frame=end)
        for start, end in zip(cut_frames, cut_frames[1:] + [frame_count])
    ]
    return segments or [SceneSegment(start_frame=0, end_frame=frame_count)]


def _second_pass_cut_candidates(
    video_path: str,
    frame_count: int,
    progress_callback: SceneProgressCallback | None = None,
) -> list[int]:
    """Sequential scan for hard cuts TransNetV2 missed, using two signals:

    1. HSV/edge spike — fast jump cuts between visually different plays.
    2. Camera-registration failure — two plays minutes apart at the SAME camera
       angle fool color/edge histograms (identity-blind), but optical-flow
       registration between the frames fails at a real cut because the
       backgrounds don't correspond. Low inlier ratio + a mild color floor
       (so plain motion blur doesn't fire) = cut candidate.

    Sequential decode (no seeks), downscaled frames, refined afterwards.
    """
    if not SCENE_SECOND_PASS_ENABLED:
        if progress_callback:
            progress_callback(1.0, "Visual cut verification skipped", frame_count, frame_count)
        return []
    from app.services.camera_motion import estimate_global_affine

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return []

    candidates: list[int] = []
    previous_small: np.ndarray | None = None
    previous_gray: np.ndarray | None = None
    frame_number = -1
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_number += 1
            if frame_number % SCENE_SAMPLE_EVERY_N_FRAMES:
                continue
            height, width = frame.shape[:2]
            scale = min(1.0, SCENE_SECOND_PASS_DOWNSCALE_WIDTH / max(1, width))
            small = cv2.resize(
                frame, (max(1, int(width * scale)), max(1, int(height * scale)))
            )
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if previous_small is not None and previous_gray is not None:
                hsv_difference, edge_difference, combined_difference = _scene_difference(
                    previous_small, small
                )
                if _is_likely_cut(hsv_difference, edge_difference, combined_difference):
                    candidates.append(frame_number)
                elif hsv_difference >= SCENE_MOTION_CUT_MIN_HSV_DIFF:
                    # Same-angle seamless cut check: does the background register?
                    _, inlier_ratio = estimate_global_affine(previous_gray, gray, 1.0)
                    if inlier_ratio < SCENE_MOTION_CUT_MAX_INLIER_RATIO:
                        candidates.append(frame_number)
            previous_small = small
            previous_gray = gray
            if progress_callback:
                progress_callback(
                    min(1.0, (frame_number + 1) / max(1, frame_count)),
                    f"Verifying scene cuts {min(frame_number + 1, frame_count)} of {frame_count} frames",
                    min(frame_number + 1, frame_count),
                    frame_count,
                )
    finally:
        capture.release()

    if not candidates:
        return []
    refine_capture = cv2.VideoCapture(video_path)
    try:
        return [
            _refine_cut_frame(refine_capture, candidate, frame_count)
            for candidate in candidates
        ]
    finally:
        refine_capture.release()


def detect_scene_segments(
    video_path: str,
    fps: float,
    frame_count: int,
    progress_callback: SceneProgressCallback | None = None,
) -> list[SceneSegment]:
    """TransNetV2 + second-pass visual scan, falling back to the hybrid detector."""
    if frame_count <= 0:
        return [SceneSegment(start_frame=0, end_frame=0)]

    if SCENE_DETECTION_METHOD == "transnetv2":
        try:
            cut_frames, timestamps = detect_transnetv2_cut_frames(
                Path(video_path),
                (
                    lambda fraction, message, completed, total: progress_callback(
                        fraction * 0.70, message, completed, total
                    )
                    if progress_callback
                    else None
                ),
            )
            extra_cuts = _second_pass_cut_candidates(
                video_path,
                frame_count,
                (
                    lambda fraction, message, completed, total: progress_callback(
                        0.70 + fraction * 0.29, message, completed, total
                    )
                    if progress_callback
                    else None
                ),
            )
            merged = sorted(set(cut_frames) | set(extra_cuts))
            segments = _segments_from_cut_frames(merged, frame_count, fps, timestamps)
            if progress_callback:
                progress_callback(
                    1.0,
                    f"Found {len(segments)} scene{'s' if len(segments) != 1 else ''}",
                    frame_count,
                    frame_count,
                )
            return segments
        except Exception as error:
            print(f"TransNetV2 failed; using hybrid scene detection: {error}")

    segments = _detect_hybrid_scene_segments(
        video_path, fps, frame_count, progress_callback
    )
    if progress_callback:
        progress_callback(
            1.0,
            f"Found {len(segments)} scene{'s' if len(segments) != 1 else ''}",
            frame_count,
            frame_count,
        )
    return segments
