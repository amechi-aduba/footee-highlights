"""Render a debug MP4 for a tracked segment (debug tool, not a product feature).

Shows every layer of the pipeline so one glance tells you which stage failed:
  gray thin boxes   all cached detections (players/goalkeepers)
  red X boxes       detections filtered as off-pitch (crowd)
  green thick box   tracked (observed) focused-player box
  cyan box          recovered (re-attached by the recovery pass)
  yellow dashed-ish yellow thin box: interpolated
  amber circle      searching state (no box drawn on purpose)
  white arrow       cumulative camera shift direction

Usage (from backend/):
    python scripts/render_debug_overlay.py <video_id> <segment_id> [--out overlay.mp4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import DETECTIONS_DIR, RAW_VIDEOS_DIR, RESULTS_DIR  # noqa: E402
from app.services.detection_cache import ROLE_INDEX  # noqa: E402


def _find_video(video_id: str) -> Path:
    matches = list(RAW_VIDEOS_DIR.glob(f"{video_id}.*"))
    if not matches:
        raise SystemExit(f"No raw video found for video_id={video_id}")
    return matches[0]


def _load_cache_arrays(video_id: str, segment_id: str) -> dict | None:
    matches = sorted((DETECTIONS_DIR / video_id).glob(f"{segment_id}_*.npz"))
    if not matches:
        return None
    with np.load(matches[-1], allow_pickle=False) as data:
        return {key: data[key] for key in data.files if key != "header"} | {
            "header": json.loads(str(data["header"]))
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_id")
    parser.add_argument("segment_id")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results_path = RESULTS_DIR / f"{args.video_id}.json"
    if not results_path.exists():
        raise SystemExit(f"No analysis result at {results_path}")
    result = json.loads(results_path.read_text())
    segment = next(
        (seg for seg in result["segments"] if seg["segment_id"] == args.segment_id), None
    )
    if segment is None:
        raise SystemExit(f"Segment {args.segment_id} not found")
    track = segment.get("focused_player_track")
    if not track:
        raise SystemExit("Segment has no focused_player_track. Run tracking first.")

    fps = result["metadata"]["fps"]
    samples_by_frame = {sample["frame_number"]: sample for sample in track["samples"]}
    cache = _load_cache_arrays(args.video_id, args.segment_id)
    detections_by_frame: dict[int, list[tuple[np.ndarray, bool, int]]] = {}
    shift_by_frame: dict[int, np.ndarray] = {}
    if cache is not None:
        frames = cache["frames"]
        for position, frame_number in enumerate(frames):
            if "camera_affine" in cache:  # v4: translation part of the cumulative affine
                shift_by_frame[int(frame_number)] = cache["camera_affine"][position][:, 2]
            elif "camera_shift" in cache:  # legacy v<=3 caches
                shift_by_frame[int(frame_number)] = cache["camera_shift"][position]
        person_roles = {ROLE_INDEX["player"], ROLE_INDEX["goalkeeper"], ROLE_INDEX["referee"]}
        for det_index in range(len(cache["det_frame_index"])):
            role = int(cache["det_role"][det_index])
            if role not in person_roles:
                continue
            frame_number = int(frames[int(cache["det_frame_index"][det_index])])
            detections_by_frame.setdefault(frame_number, []).append(
                (cache["det_bbox"][det_index], bool(cache["det_on_pitch"][det_index]), role)
            )

    video_path = _find_video(args.video_id)
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = round(segment["start_time"] * fps)
    end_frame = round(segment["end_time"] * fps)
    out_path = Path(args.out or f"debug_{args.video_id}_{args.segment_id}.mp4")
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    state_colors = {
        "tracked": (80, 220, 80),
        "recovered": (230, 200, 40),
        "interpolated": (60, 220, 230),
    }
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_number in range(start_frame, end_frame):
        success, frame = capture.read()
        if not success:
            break
        for bbox, on_pitch, _role in detections_by_frame.get(frame_number, []):
            x1, y1, x2, y2 = (int(value) for value in bbox)
            if on_pitch:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 220), 1)
                cv2.line(frame, (x1, y1), (x2, y2), (60, 60, 220), 1)
                cv2.line(frame, (x1, y2), (x2, y1), (60, 60, 220), 1)

        sample = samples_by_frame.get(frame_number)
        if sample:
            state = sample.get("state", "tracked")
            if sample.get("bbox"):
                bbox = sample["bbox"]
                color = state_colors.get(state, (80, 220, 80))
                thickness = 3 if state in ("tracked", "recovered") else 1
                cv2.rectangle(
                    frame,
                    (int(bbox["x1"]), int(bbox["y1"])),
                    (int(bbox["x2"]), int(bbox["y2"])),
                    color,
                    thickness,
                )
                label = state
                if sample.get("tracklet_id") is not None:
                    label += f" t{sample['tracklet_id']}"
                cv2.putText(
                    frame, label, (int(bbox["x1"]), max(12, int(bbox["y1"]) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                )
            elif state == "searching" and sample.get("search_center"):
                cx, cy = (int(value) for value in sample["search_center"])
                cv2.circle(frame, (cx, cy), 18, (40, 190, 250), 2)
                cv2.putText(
                    frame, "searching", (cx + 22, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 190, 250), 1,
                )

        shift = shift_by_frame.get(frame_number)
        if shift is not None and np.linalg.norm(shift) > 1:
            origin = (60, height - 60)
            tip = (int(origin[0] + np.clip(shift[0], -40, 40)), int(origin[1] + np.clip(shift[1], -40, 40)))
            cv2.arrowedLine(frame, origin, tip, (255, 255, 255), 2)
        cv2.putText(
            frame, f"frame {frame_number}", (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )
        writer.write(frame)

    capture.release()
    writer.release()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
