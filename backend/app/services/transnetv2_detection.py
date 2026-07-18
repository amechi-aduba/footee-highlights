from __future__ import annotations

import ctypes
import gc
import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from app.core.config import LOW_MEMORY_MODE, TRANSNETV2_DEVICE, TRANSNETV2_THRESHOLD


SceneProgressCallback = Callable[[float, str, int | None, int | None], None]

# Standard TransNet inference uses 100 frames and emits the middle 50. On a
# 512 MB instance, the 3D-convolution workspace for that window can cross the
# memory limit. A 50-frame window with 12/13 frames of context emits 25 frames
# and keeps the same overlap pattern with a much smaller activation peak.
_WINDOW_SIZE = 50 if LOW_MEMORY_MODE else 100
_START_PADDING = 12 if LOW_MEMORY_MODE else 25
_OUTPUT_FRAMES = 25 if LOW_MEMORY_MODE else 50
_OUTPUT_SLICE = slice(_START_PADDING, _START_PADDING + _OUTPUT_FRAMES)


def _device() -> torch.device:
    if TRANSNETV2_DEVICE != "auto":
        return torch.device(TRANSNETV2_DEVICE)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def _load_model() -> torch.nn.Module:
    """Load TransNet without importing its module-level singleton.

    ``transnetv2pt.inference`` constructs a global model at import time. That
    module is also imported by ``transnetv2pt.__init__``, so even a seemingly
    harmless ``import transnetv2pt`` creates one hidden model before this app
    can construct its own. Load the architecture file directly, without
    executing the package initializer, to keep exactly one model resident.
    """
    try:
        package_spec = importlib.util.find_spec("transnetv2pt")
        if package_spec is None or package_spec.submodule_search_locations is None:
            raise ImportError("Could not locate the transnetv2pt package")
        package_directory = Path(next(iter(package_spec.submodule_search_locations)))
        architecture_path = package_directory / "transnetv2_pytorch.py"
        architecture_spec = importlib.util.spec_from_file_location(
            "_footee_transnetv2_architecture",
            architecture_path,
        )
        if architecture_spec is None or architecture_spec.loader is None:
            raise ImportError("Could not load the TransNetV2 architecture")
        architecture_module = importlib.util.module_from_spec(architecture_spec)
        architecture_spec.loader.exec_module(architecture_module)

        # A normal constructor first allocates randomly initialized parameters,
        # then load_state_dict replaces them with the checkpoint. Constructing
        # on the meta device avoids holding both parameter sets at once.
        with torch.device("meta"):
            model = architecture_module.TransNetV2()
        weights_path = package_directory / "transnetv2-pytorch-weights.pth"
        state_dict = torch.load(
            weights_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        # Assign the memory-mapped tensors instead of copying the complete
        # checkpoint into an already-initialized parameter set.
        model.load_state_dict(state_dict, assign=True)
        del state_dict
    except Exception as error:
        raise RuntimeError(f"TransNetV2 is unavailable: {error}") from error

    if LOW_MEMORY_MODE:
        # Thread pools allocate per-thread workspaces. One CPU thread is both
        # safer on a 512 MB instance and appropriate for Render's shared CPU.
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch only permits setting this once per process.
            pass

    model.to(_device())
    model.eval()
    return model


def release_transnetv2_model() -> None:
    """Release the scene model before another PyTorch model is loaded."""
    _load_model.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Render's native Python runtime uses glibc. Returning freed allocator pages
    # here prevents the process RSS from remaining at the TransNet peak before
    # the user later starts YOLO detection/tracking.
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError, TypeError):
        pass


def _read_resized_rgb_frame(
    capture: cv2.VideoCapture,
    timestamps_seconds: list[float],
) -> np.ndarray | None:
    success, frame = capture.read()
    if not success:
        return None
    # POS_MSEC after read() points at the next frame on some backends, but used
    # consistently it still preserves variable-frame-rate segment boundaries.
    timestamps_seconds.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
    resized = cv2.resize(frame, (48, 27), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def _predict_window(
    model: torch.nn.Module,
    device: torch.device,
    frames: list[np.ndarray],
) -> np.ndarray:
    window = np.asarray(frames, dtype=np.uint8)[np.newaxis]
    with torch.inference_mode():
        tensor = torch.from_numpy(window).to(device)
        single_frame_logits, _ = model(tensor)
        probabilities = torch.sigmoid(single_frame_logits)[0, _OUTPUT_SLICE, 0]
        result = probabilities.cpu().numpy()
        del probabilities, single_frame_logits, tensor
        return result


def _stream_cut_probabilities(
    capture: cv2.VideoCapture,
    estimated_frame_count: int,
    progress_callback: SceneProgressCallback | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run overlapping TransNet windows while keeping at most 100 frames.

    The previous implementation decoded the entire reel into a NumPy array
    before inference. Standard mode preserves TransNet's 25/50/25 layout;
    low-memory mode uses the proportional 12/25/13 layout. Both keep memory
    bounded independently of reel length.
    """
    if progress_callback:
        progress_callback(0.01, "Loading TransNetV2 scene detector", 0, estimated_frame_count)

    model = _load_model()
    device = _device()
    timestamps_seconds: list[float] = []
    predictions: list[np.ndarray] = []
    buffer: list[np.ndarray] = []
    reached_end = False

    # First window: repeated frame zero followed by enough real frames to fill
    # the initial context and output region.
    initial_frame_count = _WINDOW_SIZE - _START_PADDING
    while len(buffer) < initial_frame_count:
        frame = _read_resized_rgb_frame(capture, timestamps_seconds)
        if frame is None:
            reached_end = True
            break
        buffer.append(frame)

    if not buffer:
        raise RuntimeError("Video contains no readable frames")

    first_window_tail = list(buffer)
    while len(first_window_tail) < initial_frame_count:
        first_window_tail.append(first_window_tail[-1])
    predictions.append(
        _predict_window(
            model,
            device,
            [buffer[0]] * _START_PADDING + first_window_tail,
        )
    )
    predicted_frames = _OUTPUT_FRAMES
    buffer = buffer[_OUTPUT_FRAMES - _START_PADDING :]

    while True:
        while len(buffer) < _WINDOW_SIZE and not reached_end:
            frame = _read_resized_rgb_frame(capture, timestamps_seconds)
            if frame is None:
                reached_end = True
                break
            buffer.append(frame)

        if reached_end and predicted_frames >= len(timestamps_seconds):
            break
        if not buffer:
            break

        padded_window = list(buffer)
        while len(padded_window) < _WINDOW_SIZE:
            padded_window.append(padded_window[-1])
        predictions.append(_predict_window(model, device, padded_window))
        predicted_frames += _OUTPUT_FRAMES
        buffer = buffer[_OUTPUT_FRAMES:]

        if progress_callback:
            total = max(estimated_frame_count, len(timestamps_seconds), 1)
            completed = min(predicted_frames, total)
            progress_callback(
                min(0.99, completed / total),
                f"Detecting scene cuts {completed} of {total} frames",
                completed,
                total,
            )

    probabilities = np.concatenate(predictions)[: len(timestamps_seconds)]
    timestamps = np.asarray(timestamps_seconds, dtype=np.float64)
    if (
        len(timestamps) < 2
        or timestamps[-1] <= 0
        or np.any(np.diff(timestamps) < -1e-3)
    ):
        timestamps = None

    if progress_callback:
        total_frames = len(probabilities)
        progress_callback(
            1.0,
            f"Scene scan complete for {total_frames} frames",
            total_frames,
            total_frames,
        )
    return probabilities, timestamps


def _cut_frames_from_probabilities(probabilities: np.ndarray) -> list[int]:
    """Place each cut at the end of its transition run."""
    above_threshold = probabilities >= TRANSNETV2_THRESHOLD
    cut_frames: list[int] = []
    run_start: int | None = None

    for index, is_cut in enumerate(above_threshold):
        if is_cut and run_start is None:
            run_start = index
        elif not is_cut and run_start is not None:
            cut_frames.append(index)
            run_start = None

    if run_start is not None:
        cut_frames.append(len(probabilities))
    return cut_frames


def detect_transnetv2_cut_frames(
    video_path: Path,
    progress_callback: SceneProgressCallback | None = None,
) -> tuple[list[int], np.ndarray | None]:
    """Return cut frames and optional per-frame presentation timestamps."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open video for TransNetV2")

    estimated_frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    try:
        probabilities, timestamps = _stream_cut_probabilities(
            capture,
            estimated_frame_count,
            progress_callback,
        )
        return _cut_frames_from_probabilities(probabilities), timestamps
    finally:
        capture.release()
        release_transnetv2_model()
