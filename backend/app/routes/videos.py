from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import THUMBNAILS_DIR
from app.models.schemas import (
    FocusedPlayerSelectionRequest,
    FrameDetectionResponse,
    PlayerInfoRequest,
    VideoProcessingProgressResponse,
    SegmentDetectionRequest,
    VideoAnalysisResult,
    VideoUploadResponse,
)
from app.services.player_analysis import FIELD_POSITIONS, FOOTEDNESS, build_player_profile
from app.services.object_detection import (
    DetectionWindow,
    detect_objects_at_timestamp,
    detect_objects_in_window,
)
from app.services.player_tracking import track_selected_player
from app.services.processing_progress import (
    complete_processing_progress,
    discard_processing_progress,
    fail_processing_progress,
    get_processing_progress,
    start_processing_progress,
    update_processing_progress,
)
from app.services.video_processing import process_video
from app.services.video_storage import (
    delete_video_data,
    find_video_path,
    load_analysis_result,
    purge_expired_video_data,
    save_analysis_result,
    save_uploaded_video,
    validate_video_id,
)

router = APIRouter(prefix="/api/videos", tags=["videos"])


def _validate_segment_id(segment_id: str) -> None:
    if not segment_id.startswith("seg_") or not segment_id[4:].isdigit():
        raise HTTPException(status_code=400, detail="Invalid segment ID")


def _find_segment(result: dict, segment_id: str) -> dict:
    _validate_segment_id(segment_id)
    for segment in result["segments"]:
        if segment["segment_id"] == segment_id:
            return segment
    raise HTTPException(status_code=404, detail="Segment not found")


def _absolute_segment_time(segment: dict, clip_time_seconds: float) -> float:
    duration = segment["end_time"] - segment["start_time"]
    if clip_time_seconds < 0 or clip_time_seconds > duration:
        raise HTTPException(status_code=400, detail="Selected time is outside this segment")
    return segment["start_time"] + clip_time_seconds


@router.post("/upload", response_model=VideoUploadResponse)
def upload_video(video: UploadFile = File(...)) -> VideoUploadResponse:
    if not video.filename:
        raise HTTPException(status_code=400, detail="A filename is required")

    video_id = uuid4().hex
    purge_expired_video_data()
    save_uploaded_video(video_id, video.filename, video.file)
    return VideoUploadResponse(
        video_id=video_id,
        filename=video.filename,
        message="Video uploaded temporarily and scheduled for automatic deletion",
    )


@router.post("/{video_id}/process", response_model=VideoAnalysisResult)
def process_uploaded_video(video_id: str) -> VideoAnalysisResult:
    video_path = find_video_path(video_id)
    start_processing_progress(video_id)
    try:
        result = process_video(video_id, video_path, update_processing_progress)
        save_analysis_result(video_id, result)
    except Exception as error:
        detail = error.detail if isinstance(error, HTTPException) else "Processing failed. Please retry."
        fail_processing_progress(video_id, str(detail))
        raise
    complete_processing_progress(video_id)
    return VideoAnalysisResult.model_validate(result)


@router.get(
    "/{video_id}/processing-progress",
    response_model=VideoProcessingProgressResponse,
)
def get_uploaded_video_processing_progress(video_id: str) -> VideoProcessingProgressResponse:
    validate_video_id(video_id)
    return VideoProcessingProgressResponse.model_validate(get_processing_progress(video_id))


@router.get("/{video_id}/result", response_model=VideoAnalysisResult)
def get_video_result(video_id: str) -> VideoAnalysisResult:
    return VideoAnalysisResult.model_validate(load_analysis_result(video_id))


@router.get("/{video_id}/video")
def get_uploaded_video(video_id: str) -> FileResponse:
    video_path = find_video_path(video_id)
    return FileResponse(
        Path(video_path),
        filename=video_path.name,
        content_disposition_type="inline",
    )


@router.get("/{video_id}/thumbnail/{segment_id}")
def get_segment_thumbnail(video_id: str, segment_id: str) -> FileResponse:
    validate_video_id(video_id)
    _validate_segment_id(segment_id)
    thumbnail_path = THUMBNAILS_DIR / video_id / f"{segment_id}.jpg"
    if not thumbnail_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(Path(thumbnail_path), media_type="image/jpeg")


@router.delete("/{video_id}")
def delete_uploaded_video_data(video_id: str) -> dict:
    deleted = delete_video_data(video_id)
    discard_processing_progress(video_id)
    return {"status": "deleted", **deleted}


@router.post("/{video_id}/cleanup")
def clean_up_video_session(video_id: str) -> dict:
    """Beacon-friendly cleanup used when the browser page is closing."""
    deleted = delete_video_data(video_id)
    discard_processing_progress(video_id)
    return {"status": "deleted", **deleted}


@router.post("/{video_id}/segment/{segment_id}/detect")
def detect_segment_objects(
    video_id: str,
    segment_id: str,
    request: SegmentDetectionRequest,
) -> dict:
    result = load_analysis_result(video_id)
    segment = _find_segment(result, segment_id)
    absolute_start_time = _absolute_segment_time(segment, request.clip_time_seconds)
    fps = result["metadata"]["fps"]
    summary = detect_objects_in_window(
        find_video_path(video_id),
        fps,
        DetectionWindow(
            start_frame=round(absolute_start_time * fps),
            end_frame=round(segment["end_time"] * fps),
        ),
    )
    segment["object_detection_summary"] = summary
    save_analysis_result(video_id, result)
    return summary


@router.post(
    "/{video_id}/segment/{segment_id}/detect-frame",
    response_model=FrameDetectionResponse,
)
def detect_segment_frame(
    video_id: str,
    segment_id: str,
    request: SegmentDetectionRequest,
) -> FrameDetectionResponse:
    result = load_analysis_result(video_id)
    segment = _find_segment(result, segment_id)
    absolute_time = _absolute_segment_time(segment, request.clip_time_seconds)
    return FrameDetectionResponse.model_validate(
        detect_objects_at_timestamp(find_video_path(video_id), absolute_time)
    )


@router.post("/{video_id}/player-info")
def save_player_info(video_id: str, request: PlayerInfoRequest) -> dict:
    positions = [position for position in request.positions if position in FIELD_POSITIONS]
    if not positions:
        raise HTTPException(status_code=400, detail="Select at least one valid position")
    if request.footedness not in FOOTEDNESS:
        raise HTTPException(status_code=400, detail="Footedness must be right, left, or both")
    result = load_analysis_result(video_id)
    result["player_info"] = {"positions": positions, "footedness": request.footedness}
    # Player info changed — any existing profile is stale.
    result.pop("player_profile", None)
    save_analysis_result(video_id, result)
    return {"status": "saved", "player_info": result["player_info"]}


@router.post("/{video_id}/player-profile")
def generate_player_profile(video_id: str) -> dict:
    result = load_analysis_result(video_id)
    if not result.get("player_info"):
        raise HTTPException(
            status_code=400,
            detail="Set the player's positions and footedness before generating a profile",
        )
    profile = build_player_profile(video_id, result)
    if profile.get("status") == "insufficient_data":
        raise HTTPException(status_code=400, detail=profile["message"])
    result["player_profile"] = profile
    save_analysis_result(video_id, result)
    return profile


@router.post("/{video_id}/segment/{segment_id}/focused-player")
def select_focused_player(
    video_id: str,
    segment_id: str,
    request: FocusedPlayerSelectionRequest,
) -> dict:
    result = load_analysis_result(video_id)
    segment = _find_segment(result, segment_id)
    absolute_time = _absolute_segment_time(segment, request.clip_time_seconds)
    selection = {
        "detection_id": request.detection_id,
        "selected_at_time": round(absolute_time, 3),
        "bbox": request.bbox.model_dump(),
        "confidence": request.confidence,
        "team_id": request.team_id,
        "jersey_color_hex": request.jersey_color_hex,
        "jersey_descriptor": request.jersey_descriptor,
    }
    segment["focused_player_status"] = "selected"
    if request.additive and segment.get("focused_player_anchors"):
        # Extra anchor: the user re-clicked the SAME player where tracking was
        # lost. Identity gets pinned at every anchor; stitching bridges between.
        segment["focused_player_anchors"].append(selection)
    else:
        segment["focused_player_anchors"] = [selection]
        segment["focused_player_selection"] = selection
    save_analysis_result(video_id, result)
    return segment


@router.delete("/{video_id}/segment/{segment_id}/focused-player")
def reset_focused_player(video_id: str, segment_id: str) -> dict:
    """Undo a wrong selection: clears the selection, all anchors, and the track
    so the next click starts completely fresh."""
    result = load_analysis_result(video_id)
    segment = _find_segment(result, segment_id)
    segment["focused_player_status"] = "not_selected"
    segment["focused_player_selection"] = None
    segment["focused_player_anchors"] = None
    segment["focused_player_track"] = None
    save_analysis_result(video_id, result)
    return segment


@router.post("/{video_id}/segment/{segment_id}/track-focused-player")
def track_focused_player(video_id: str, segment_id: str) -> dict:
    result = load_analysis_result(video_id)
    segment = _find_segment(result, segment_id)
    selection = segment.get("focused_player_selection")
    if selection is None:
        raise HTTPException(status_code=400, detail="Select a player before starting tracking")

    track = track_selected_player(
        video_id,
        segment_id,
        find_video_path(video_id),
        result["metadata"]["fps"],
        segment["start_time"],
        segment["end_time"],
        selection,
        anchors=segment.get("focused_player_anchors") or None,
    )
    segment["focused_player_status"] = "tracked"
    segment["focused_player_track"] = track
    # Real per-clip stats (replaces the old always-"Pending" placeholders).
    try:
        from app.services.player_analysis import segment_features

        segment["features"] = segment_features(video_id, segment, result["metadata"]["fps"])
    except Exception:
        segment["features"] = None
    save_analysis_result(video_id, result)
    # Ship the stats with the response so the UI can show them without a refetch.
    return {**track, "clip_features": segment["features"]}
