"""Precompute one player-tracking showcase for each static demo reel.

The generated tracks are embedded in ``frontend/public/samples/*/result.json``.
Visitors can therefore see the real tracking overlay and clip statistics without
waking the production API or spending Azure CPU.

Run from the repository root after ``build_demo_samples.py``:

    backend/.venv/Scripts/python.exe backend/scripts/build_demo_tracks.py

The target points below identify the featured player on a reviewed frame. They
are normalized so rebuilding remains independent of the video's pixel size.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIRECTORY = REPOSITORY_ROOT / "backend"
PUBLIC_SAMPLES_DIRECTORY = REPOSITORY_ROOT / "frontend" / "public" / "samples"

# Keep this one-time local build bounded. Tracks are sampled every third frame,
# and the compact cache is deleted after validation rather than deployed.
os.environ.setdefault("FOOTEE_STORAGE_DIR", "storage/demo-tracks")
os.environ.setdefault("LOW_MEMORY_MODE", "true")
local_football_model = BACKEND_DIRECTORY / "models" / "football-yolo11m.pt"
if local_football_model.is_file():
    os.environ.setdefault("YOLO_MODEL_PATH", str(local_football_model))
    os.environ.setdefault("TRACKING_MODEL_PATH", str(local_football_model))
os.environ.setdefault("YOLO_CONFIDENCE_THRESHOLD", "0.15")
os.environ.setdefault("TRACKING_ENGINE", "tracklet")
os.environ.setdefault("TRACKING_BATCH_SIZE", "1")
os.environ.setdefault("DETECTION_CACHE_STRIDE", "3")
os.environ.setdefault("TRACKING_USE_NEURAL_REID", "false")
os.environ.setdefault("CAMERA_MOTION_DOWNSCALE_WIDTH", "320")

sys.path.insert(0, str(BACKEND_DIRECTORY))

from app.services.object_detection import detect_objects_at_timestamp  # noqa: E402
from app.services.player_analysis import segment_features  # noqa: E402
from app.services.player_tracking import track_selected_player  # noqa: E402


SHOWCASES: tuple[dict[str, Any], ...] = (
    {
        "sample_id": "alistair-johnston",
        "segment_id": "seg_007",
        "timestamp_seconds": 81.200,
        "target_point": (0.4950, 0.5590),
    },
    {
        "sample_id": "jack-harrison",
        "segment_id": "seg_013",
        "timestamp_seconds": 339.500,
        "target_point": (0.6031, 0.7611),
        # The featured number 11's dribble occupies this reviewed portion of a
        # longer scene; defenders converge immediately after this window.
        "tracking_window": (338.300, 340.300),
        "restore_windows": {"seg_002": (8.976, 21.922)},
    },
    {
        "sample_id": "ousseni-bouda",
        "segment_id": "seg_004",
        "timestamp_seconds": 12.081,
        "target_point": (0.4797, 0.5583),
    },
)


def _find_segment(result: dict[str, Any], segment_id: str) -> dict[str, Any]:
    for segment in result["segments"]:
        if segment["segment_id"] == segment_id:
            return segment
    raise RuntimeError(f"Result does not contain {segment_id}")


def _write_result(result_path: Path, result: dict[str, Any]) -> None:
    temporary_path = result_path.with_suffix(".json.tmp")
    # Explicit LF output keeps generated JSON stable on Windows and Linux.
    with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    temporary_path.replace(result_path)


def _select_nearest_player(
    frame_result: dict[str, Any], target_point: tuple[float, float]
) -> dict[str, Any]:
    target_x = target_point[0] * frame_result["frame_width"]
    target_y = target_point[1] * frame_result["frame_height"]
    eligible = [
        detection
        for detection in frame_result["detections"]
        if detection["role"] in {"player", "goalkeeper"}
    ]
    if not eligible:
        raise RuntimeError("No selectable player was detected on the showcase frame")

    def squared_distance(detection: dict[str, Any]) -> float:
        bbox = detection["bbox"]
        center_x = (bbox["x1"] + bbox["x2"]) / 2
        center_y = (bbox["y1"] + bbox["y2"]) / 2
        return (center_x - target_x) ** 2 + (center_y - target_y) ** 2

    selected = min(eligible, key=squared_distance)
    distance_fraction = (
        squared_distance(selected)
        / (frame_result["frame_width"] ** 2 + frame_result["frame_height"] ** 2)
    ) ** 0.5
    if distance_fraction > 0.04:
        raise RuntimeError(
            "The closest detection moved too far from the reviewed target point "
            f"({distance_fraction:.3f} of the frame diagonal)"
        )
    return selected


def _build_showcase(showcase: dict[str, Any], force: bool) -> None:
    sample_id = showcase["sample_id"]
    sample_directory = PUBLIC_SAMPLES_DIRECTORY / sample_id
    result_path = sample_directory / "result.json"
    video_path = sample_directory / "video.mp4"
    if not result_path.is_file() or not video_path.is_file():
        raise FileNotFoundError(
            f"Build the {sample_id} static sample before generating its track"
        )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    segment = _find_segment(result, showcase["segment_id"])
    if segment.get("focused_player_track") and not force:
        result["demo"]["tracking_showcase_segment_id"] = segment["segment_id"]
        _write_result(result_path, result)
        print(f"Skipping {sample_id}: {showcase['segment_id']} already has a track")
        return

    # A demo has one intentional showcase. Remove a superseded local showcase
    # if this configuration moves to a clearer reviewed clip.
    for other_segment in result["segments"]:
        if other_segment["segment_id"] == segment["segment_id"]:
            continue
        if other_segment.get("focused_player_track"):
            other_segment["focused_player_status"] = "demo"
            other_segment["focused_player_selection"] = None
            other_segment["focused_player_anchors"] = None
            other_segment["focused_player_track"] = None
            other_segment["features"] = None

    for segment_id, original_window in showcase.get("restore_windows", {}).items():
        restored_segment = _find_segment(result, segment_id)
        restored_segment["start_time"], restored_segment["end_time"] = original_window

    tracking_window = showcase.get("tracking_window")
    if tracking_window:
        segment["start_time"], segment["end_time"] = tracking_window

    anchor_specs = showcase.get("anchors") or (
        {
            "timestamp_seconds": showcase["timestamp_seconds"],
            "target_point": showcase["target_point"],
        },
    )
    selections = []
    for anchor_spec in anchor_specs:
        selected_at_time = anchor_spec.get("timestamp_seconds")
        if selected_at_time is None:
            selected_at_time = segment["start_time"] + anchor_spec["clip_time_seconds"]
        frame_result = detect_objects_at_timestamp(video_path, selected_at_time)
        detection = _select_nearest_player(frame_result, anchor_spec["target_point"])
        selections.append(
            {
                "detection_id": detection["detection_id"],
                "selected_at_time": round(selected_at_time, 3),
                "bbox": detection["bbox"],
                "confidence": detection["confidence"],
                "team_id": detection.get("team_id"),
                "jersey_color_hex": detection.get("jersey_color_hex"),
                "jersey_descriptor": detection.get("jersey_descriptor"),
            }
        )
    selection = selections[0]
    print(
        f"Tracking {sample_id} {segment['segment_id']} from "
        f"{len(selections)} reviewed anchor{'s' if len(selections) != 1 else ''}",
        flush=True,
    )
    track = track_selected_player(
        result["video_id"],
        segment["segment_id"],
        video_path,
        result["metadata"]["fps"],
        segment["start_time"],
        segment["end_time"],
        selection,
        anchors=selections,
    )

    segment["focused_player_status"] = "tracked"
    segment["focused_player_selection"] = selection
    segment["focused_player_anchors"] = selections
    segment["focused_player_track"] = track
    segment["features"] = segment_features(
        result["video_id"], segment, result["metadata"]["fps"]
    )
    result["demo"]["tracking_showcase_segment_id"] = segment["segment_id"]

    _write_result(result_path, result)
    coverage = track.get("metrics", {}).get("coverage")
    coverage_label = f", {coverage:.0%} coverage" if isinstance(coverage, float) else ""
    print(
        f"Embedded {len(track['samples'])} samples in {result_path.name}{coverage_label}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--sample",
        action="append",
        choices=[showcase["sample_id"] for showcase in SHOWCASES],
        help="Build only the selected sample ID. May be supplied more than once.",
    )
    arguments = parser.parse_args()
    selected_ids = set(arguments.sample or [])
    for showcase in SHOWCASES:
        if selected_ids and showcase["sample_id"] not in selected_ids:
            continue
        _build_showcase(showcase, arguments.force)


if __name__ == "__main__":
    main()
