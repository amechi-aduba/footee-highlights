"""Thread-safe, process-local progress for the temporary video pipeline."""

from copy import deepcopy
from threading import Lock


_STAGES = (
    ("scene_cuts", "Scene cuts (TransNetV2)"),
    ("cutaway_filtering", "Cutaway filtering"),
    ("thumbnails", "Thumbnails"),
)
_STAGE_RANGES = {
    "scene_cuts": (1.0, 60.0),
    "cutaway_filtering": (60.0, 85.0),
    "thumbnails": (85.0, 99.0),
}
_lock = Lock()
_progress_by_video: dict[str, dict] = {}


def _new_progress() -> dict:
    return {
        "status": "idle",
        "progress_percent": 0.0,
        "current_stage": None,
        "message": "Ready to process",
        "stages": [
            {
                "key": key,
                "label": label,
                "status": "pending",
                "progress_percent": 0.0,
                "completed_items": 0,
                "total_items": None,
            }
            for key, label in _STAGES
        ],
    }


def start_processing_progress(video_id: str) -> dict:
    progress = _new_progress()
    progress.update(
        status="processing",
        progress_percent=1.0,
        current_stage="scene_cuts",
        message="Reading video metadata",
    )
    progress["stages"][0]["status"] = "active"
    with _lock:
        _progress_by_video[video_id] = progress
        return deepcopy(progress)


def update_processing_progress(
    video_id: str,
    stage_key: str,
    stage_fraction: float,
    message: str,
    completed_items: int | None = None,
    total_items: int | None = None,
) -> dict:
    if stage_key not in _STAGE_RANGES:
        raise ValueError(f"Unknown processing stage: {stage_key}")

    fraction = max(0.0, min(1.0, stage_fraction))
    stage_index = next(index for index, stage in enumerate(_STAGES) if stage[0] == stage_key)
    range_start, range_end = _STAGE_RANGES[stage_key]
    calculated_percent = range_start + (range_end - range_start) * fraction

    with _lock:
        progress = _progress_by_video.setdefault(video_id, _new_progress())
        progress["status"] = "processing"
        progress["current_stage"] = stage_key
        progress["message"] = message
        # A detector fallback may restart a stage. Never make the visible bar
        # jump backwards when that happens.
        progress["progress_percent"] = round(
            max(float(progress["progress_percent"]), calculated_percent), 1
        )

        for index, stage in enumerate(progress["stages"]):
            if index < stage_index:
                stage["status"] = "complete"
                stage["progress_percent"] = 100.0
            elif index == stage_index:
                stage["status"] = "complete" if fraction >= 1.0 else "active"
                stage["progress_percent"] = round(
                    max(float(stage["progress_percent"]), fraction * 100.0), 1
                )
                if completed_items is not None:
                    stage["completed_items"] = completed_items
                if total_items is not None:
                    stage["total_items"] = total_items
            elif stage["status"] != "complete":
                stage["status"] = "pending"

        return deepcopy(progress)


def complete_processing_progress(video_id: str) -> dict:
    with _lock:
        progress = _progress_by_video.setdefault(video_id, _new_progress())
        progress.update(
            status="complete",
            progress_percent=100.0,
            current_stage=None,
            message="Reel split complete",
        )
        for stage in progress["stages"]:
            stage["status"] = "complete"
            stage["progress_percent"] = 100.0
            if stage["total_items"] is not None:
                stage["completed_items"] = stage["total_items"]
        return deepcopy(progress)


def fail_processing_progress(video_id: str, message: str) -> dict:
    with _lock:
        progress = _progress_by_video.setdefault(video_id, _new_progress())
        progress["status"] = "failed"
        progress["message"] = message
        for stage in progress["stages"]:
            if stage["status"] == "active":
                stage["status"] = "failed"
        return deepcopy(progress)


def get_processing_progress(video_id: str) -> dict:
    with _lock:
        return deepcopy(_progress_by_video.get(video_id, _new_progress()))


def discard_processing_progress(video_id: str) -> None:
    with _lock:
        _progress_by_video.pop(video_id, None)
