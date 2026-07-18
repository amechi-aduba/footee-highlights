from typing import Any

from app.models.schemas import VideoMetadata
from app.services.scene_detection import SceneSegment


def track_focused_player_placeholder() -> dict[str, Any]:
    """Future ByteTrack integration point for focused-player trajectories."""
    return {"status": "not_selected"}


def extract_player_features_placeholder() -> dict[str, float | None]:
    """Future feature engineering point for tactical metrics."""
    return {
        "wide_positioning_score": None,
        "central_positioning_score": None,
        "final_third_score": None,
        "ball_proximity_score": None,
    }


def classify_player_role_placeholder() -> dict[str, str]:
    """Future Azure ML or local classifier integration point."""
    return {
        "predicted_role": "unknown",
        "confidence": "low",
        "summary": "Player role analysis will be generated after focused-player tracking is implemented.",
    }


def compare_to_player_profiles_placeholder() -> list[Any]:
    """Future comparison point for profiles stored in Azure PostgreSQL."""
    return []


def build_analysis_result(
    video_id: str,
    metadata: VideoMetadata,
    segments: list[SceneSegment],
    classifications: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    track_focused_player_placeholder()
    compare_to_player_profiles_placeholder()

    segment_results = []
    for index, segment in enumerate(segments, start=1):
        segment_id = f"seg_{index:03d}"
        classification = (
            classifications[index - 1]
            if classifications and index - 1 < len(classifications)
            else {}
        )
        # Prefer real decoder timestamps (VFR-safe) over frame/nominal-fps math.
        start_time = (
            segment.start_time_seconds
            if segment.start_time_seconds is not None
            else segment.start_frame / metadata.fps
        )
        end_time = (
            segment.end_time_seconds
            if segment.end_time_seconds is not None
            else segment.end_frame / metadata.fps
        )
        segment_results.append(
            {
                "segment_id": segment_id,
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "kind": classification.get("kind", "gameplay"),
                "gameplay_score": classification.get("gameplay_score"),
                "thumbnail_path": f"/api/videos/{video_id}/thumbnail/{segment_id}",
                "focused_player_status": "not_selected",
                "focused_player_selection": None,
                "focused_player_track": None,
                "object_detection_summary": None,
                "detected_actions_placeholder": [],
                "features_placeholder": extract_player_features_placeholder(),
            }
        )

    return {
        "video_id": video_id,
        "status": "completed",
        "metadata": metadata.model_dump(),
        "segments": segment_results,
        "final_profile_placeholder": classify_player_role_placeholder(),
    }
