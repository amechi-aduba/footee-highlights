from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

from app.core.config import TRANSNETV2_DEVICE, TRANSNETV2_THRESHOLD


def _device() -> torch.device:
    if TRANSNETV2_DEVICE != "auto":
        return torch.device(TRANSNETV2_DEVICE)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def _load_model() -> torch.nn.Module:
    try:
        from transnetv2pt.inference import model
    except Exception as error:
        raise RuntimeError(f"TransNetV2 is unavailable: {error}") from error

    model.to(_device())
    model.eval()
    return model


def _read_resized_rgb_frames(video_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Decode every frame for TransNetV2 and record each frame's REAL timestamp.

    Highlight videos are often variable-frame-rate exports, so frame/nominal-fps
    times drift from what the browser's <video> element plays. Since we decode
    every frame here anyway, capturing presentation timestamps is free and lets
    segment boundaries land where the viewer actually sees them.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open video for TransNetV2")

    frames: list[np.ndarray] = []
    timestamps_seconds: list[float] = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            # POS_MSEC after read() = timestamp of the NEXT frame on some
            # backends; treated uniformly it still yields consistent boundaries.
            timestamps_seconds.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
            resized = cv2.resize(frame, (48, 27), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise RuntimeError("Video contains no readable frames")

    timestamps = np.asarray(timestamps_seconds, dtype=np.float64)
    # Some codecs/backends return zeros or garbage for POS_MSEC — only trust
    # timestamps that are non-trivial and monotonically non-decreasing.
    if len(timestamps) < 2 or timestamps[-1] <= 0 or np.any(np.diff(timestamps) < -1e-3):
        return np.asarray(frames, dtype=np.uint8), None
    return np.asarray(frames, dtype=np.uint8), timestamps


def _input_windows(frames: np.ndarray):
    start_padding = 25
    end_padding = 25 + 50 - (len(frames) % 50 or 50)
    padded = np.concatenate(
        [np.repeat(frames[:1], start_padding, axis=0), frames, np.repeat(frames[-1:], end_padding, axis=0)]
    )
    for start in range(0, len(padded) - 99, 50):
        yield padded[start : start + 100][np.newaxis]


def _predict_cut_probabilities(frames: np.ndarray) -> np.ndarray:
    model = _load_model()
    device = _device()
    predictions: list[np.ndarray] = []

    with torch.inference_mode():
        for window in _input_windows(frames):
            tensor = torch.from_numpy(window).to(device)
            single_frame_logits, _ = model(tensor)
            probabilities = torch.sigmoid(single_frame_logits)[0, 25:75, 0]
            predictions.append(probabilities.cpu().numpy())

    return np.concatenate(predictions)[: len(frames)]


def _cut_frames_from_probabilities(probabilities: np.ndarray) -> list[int]:
    """Place each cut at the END of its transition run.

    Highlight edits use dissolves/wipes: the above-threshold run spans the whole
    transition. Cutting at the peak leaves the tail of the PREVIOUS clip (and
    blended frames) at the start of the next segment — cutting at the run end
    starts each segment on the first clean frame of the new shot. For hard cuts
    (run length 1) both choices are identical.
    """
    above_threshold = probabilities >= TRANSNETV2_THRESHOLD
    cut_frames: list[int] = []
    run_start: int | None = None

    for index, is_cut in enumerate(above_threshold):
        if is_cut and run_start is None:
            run_start = index
        elif not is_cut and run_start is not None:
            cut_frames.append(index)  # first frame below threshold = new shot
            run_start = None

    if run_start is not None:
        cut_frames.append(len(probabilities))
    return cut_frames


def detect_transnetv2_cut_frames(video_path: Path) -> tuple[list[int], np.ndarray | None]:
    """Returns (cut_frames, per-frame timestamps in seconds or None)."""
    frames, timestamps = _read_resized_rgb_frames(video_path)
    return _cut_frames_from_probabilities(_predict_cut_probabilities(frames)), timestamps
