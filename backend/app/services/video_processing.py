from pathlib import Path

import cv2
from fastapi import HTTPException

from app.core.config import THUMBNAILS_DIR, ensure_storage_directories
from app.models.schemas import VideoMetadata
from app.services.analysis_builder import build_analysis_result
from app.services.scene_detection import SceneSegment, detect_scene_segments


def _extract_metadata(capture: cv2.VideoCapture) -> VideoMetadata:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        raise HTTPException(status_code=422, detail="Could not read video metadata")

    return VideoMetadata(
        fps=round(fps, 3),
        frame_count=frame_count,
        duration_seconds=round(frame_count / fps, 3),
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )


def _save_segment_thumbnails(
    capture: cv2.VideoCapture,
    video_id: str,
    segments: list[SceneSegment],
) -> None:
    thumbnail_directory = THUMBNAILS_DIR / video_id
    thumbnail_directory.mkdir(parents=True, exist_ok=True)

    for index, segment in enumerate(segments, start=1):
        segment_id = f"seg_{index:03d}"
        midpoint = segment.start_frame + max(0, segment.end_frame - segment.start_frame) // 2
        capture.set(cv2.CAP_PROP_POS_FRAMES, midpoint)
        success, frame = capture.read()
        if not success:
            raise HTTPException(status_code=422, detail=f"Could not extract thumbnail for {segment_id}")
        cv2.imwrite(str(thumbnail_directory / f"{segment_id}.jpg"), frame)


def process_video(video_id: str, video_path: Path) -> dict:
    """Run local MVP processing. Azure Container Apps can host this worker later."""
    ensure_storage_directories()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise HTTPException(status_code=422, detail="Could not open uploaded video")

    try:
        metadata = _extract_metadata(capture)
        segments = detect_scene_segments(str(video_path), metadata.fps, metadata.frame_count)
        _save_segment_thumbnails(capture, video_id, segments)
    finally:
        capture.release()

    from app.services.segment_classifier import classify_segments

    classifications = classify_segments(video_path, segments)
    return build_analysis_result(video_id, metadata, segments, classifications)
