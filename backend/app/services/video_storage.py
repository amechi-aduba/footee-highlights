import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, BinaryIO

from fastapi import HTTPException

from app.core.config import (
    ALLOWED_VIDEO_EXTENSIONS,
    DETECTIONS_DIR,
    MAX_UPLOAD_BYTES,
    RAW_VIDEOS_DIR,
    RESULTS_DIR,
    SEGMENTS_DIR,
    THUMBNAILS_DIR,
    UPLOAD_RETENTION_SECONDS,
    ensure_storage_directories,
)

_VIDEO_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def validate_video_id(video_id: str) -> str:
    if not _VIDEO_ID_PATTERN.fullmatch(video_id):
        raise HTTPException(status_code=400, detail="Invalid video ID")
    return video_id


def _video_paths(video_id: str) -> list[Path]:
    validate_video_id(video_id)
    return [
        *RAW_VIDEOS_DIR.glob(f"{video_id}.*"),
        RESULTS_DIR / f"{video_id}.json",
        THUMBNAILS_DIR / video_id,
        DETECTIONS_DIR / video_id,
        SEGMENTS_DIR / video_id,
    ]


def save_uploaded_video(video_id: str, filename: str, file_object: BinaryIO) -> str:
    """Temporarily store one upload under its private, unguessable ID.

    Cross-user content deduplication is deliberately avoided: reusing an
    existing ID for an identical upload could expose another session's saved
    selections or analysis when the service is public.
    """
    ensure_storage_directories()
    validate_video_id(video_id)
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type. Allowed extensions: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
        )

    temporary = RAW_VIDEOS_DIR / f".upload_{video_id}{suffix}"
    destination = RAW_VIDEOS_DIR / f"{video_id}{suffix}"
    bytes_written = 0
    try:
        with temporary.open("wb") as output:
            while chunk := file_object.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video is too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                    )
                output.write(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return video_id


def find_video_path(video_id: str) -> Path:
    ensure_storage_directories()
    validate_video_id(video_id)
    matches = list(RAW_VIDEOS_DIR.glob(f"{video_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Uploaded video not found")
    matches[0].touch(exist_ok=True)
    return matches[0]


def save_analysis_result(video_id: str, result: dict[str, Any]) -> Path:
    ensure_storage_directories()
    validate_video_id(video_id)
    destination = RESULTS_DIR / f"{video_id}.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return destination


def load_analysis_result(video_id: str) -> dict[str, Any]:
    ensure_storage_directories()
    validate_video_id(video_id)
    result_path = RESULTS_DIR / f"{video_id}.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Analysis result not found")
    result_path.touch(exist_ok=True)
    return json.loads(result_path.read_text(encoding="utf-8"))


def delete_video_data(video_id: str) -> dict[str, int]:
    """Delete one upload and every known artifact derived from it."""
    ensure_storage_directories()
    paths = _video_paths(video_id)
    files_deleted = 0
    bytes_deleted = 0

    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
            files_deleted += len(files)
            bytes_deleted += sum(candidate.stat().st_size for candidate in files)
            shutil.rmtree(path)
        else:
            files_deleted += 1
            bytes_deleted += path.stat().st_size
            path.unlink(missing_ok=True)

    return {"files_deleted": files_deleted, "bytes_deleted": bytes_deleted}


def purge_expired_video_data(
    retention_seconds: int = UPLOAD_RETENTION_SECONDS,
) -> dict[str, int]:
    """Remove abandoned sessions older than the configured retention window."""
    ensure_storage_directories()
    candidate_ids: set[str] = set()

    for path in RAW_VIDEOS_DIR.iterdir():
        if path.is_file() and _VIDEO_ID_PATTERN.fullmatch(path.stem):
            candidate_ids.add(path.stem)
    for path in RESULTS_DIR.glob("*.json"):
        if _VIDEO_ID_PATTERN.fullmatch(path.stem):
            candidate_ids.add(path.stem)
    for root in (THUMBNAILS_DIR, DETECTIONS_DIR, SEGMENTS_DIR):
        for path in root.iterdir():
            if path.is_dir() and _VIDEO_ID_PATTERN.fullmatch(path.name):
                candidate_ids.add(path.name)

    cutoff = time.time() - retention_seconds
    videos_deleted = 0
    files_deleted = 0
    bytes_deleted = 0

    # A process crash can leave a partially streamed upload behind before it
    # receives its final UUID filename. Expire those files too.
    for temporary in RAW_VIDEOS_DIR.glob(".upload_*"):
        if temporary.is_file() and temporary.stat().st_mtime < cutoff:
            files_deleted += 1
            bytes_deleted += temporary.stat().st_size
            temporary.unlink(missing_ok=True)

    for video_id in candidate_ids:
        existing_paths = [path for path in _video_paths(video_id) if path.exists()]
        if not existing_paths:
            continue
        last_activity = max(path.stat().st_mtime for path in existing_paths)
        if last_activity >= cutoff:
            continue
        deleted = delete_video_data(video_id)
        videos_deleted += 1
        files_deleted += deleted["files_deleted"]
        bytes_deleted += deleted["bytes_deleted"]

    return {
        "videos_deleted": videos_deleted,
        "files_deleted": files_deleted,
        "bytes_deleted": bytes_deleted,
    }
