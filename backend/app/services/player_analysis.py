"""Player archetype analysis (v2: movement + ball events).

Measurable with current signals: involvement (ball proximity), work rate
(distance in player-height units — no pitch calibration needed), explosiveness
(sprints), wideness (position relative to detected teammates), and — new —
BALL EVENTS estimated from ball trajectory + possession windows:

  time with ball = seconds the ball sits within a feet-radius of the player
                   (replaces discrete "touches", which fragmented on tiny-ball
                   footage; this degrades gracefully under sparse detection)
  pass           = a release that travels ≥2 player-heights and is received by
                   another same-team player
  shot           = a fast release that converges on a detected GOALKEEPER
                   (high confidence — the one goal-directed signal we can see)
  shot attempt   = any other fast release nobody on our team receives
                   (low confidence; also catches clearances/crosses)

These are ESTIMATES. The goal is not visible, ball detection on tiny fast balls
is imperfect, and one-touch play can undercount. Everything is gated on ball
data availability and reported with that honesty in the profile evidence.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.core.config import (
    BALL_EVENTS_ENABLED,
    BALL_MIN_CONFIDENCE,
    BALL_STATIC_FRACTION,
    BALL_STATIC_RADIUS_FACTOR,
    PASS_MIN_TRAVEL_HEIGHTS,
    PASS_RECEIVE_WINDOW_SECONDS,
    PASS_TEAMMATE_MIN_JERSEY,
    POSSESSION_MIN_SAMPLES,
    POSSESSION_RADIUS_HEIGHTS,
    SHOT_ATTEMPT_MIN_TRAVEL_HEIGHTS,
    SHOT_SPEED_HEIGHTS_PER_SEC,
    TOUCH_RADIUS_HEIGHTS,
)
from app.services.detection_cache import ROLE_INDEX, DetectionCache, cache_path, load_cache_unchecked

FIELD_POSITIONS = ("st", "lw", "rw", "cm", "cdm", "10", "cb", "rb", "lb")
FOOTEDNESS = ("right", "left", "both")

ARCHETYPES: dict[str, dict[str, str]] = {
    "ball_playing_defender": {
        "label": "Ball-Playing Defender",
        "group": "DEF",
        "description": "Comfortable bringing the ball out from the back and breaking lines with passes.",
    },
    "stopper": {
        "label": "Stopper / Traditional Center-Back",
        "group": "DEF",
        "description": "Strong and aggressive — duels, tackles, blocks; defending first.",
    },
    "wing_back": {
        "label": "Attacking / Inverted Wing-Back",
        "group": "DEF",
        "description": "Pushes high to join the attack, overlapping or tucking into midfield.",
    },
    "holding_six": {
        "label": "Holding Midfielder (The 6)",
        "group": "MID",
        "description": "Shields the back line, breaks up attacks, recycles possession simply.",
    },
    "box_to_box": {
        "label": "Box-to-Box Midfielder (The 8)",
        "group": "MID",
        "description": "Elite engine — defends deep and arrives late in the opponent's box.",
    },
    "deep_lying_playmaker": {
        "label": "Deep-Lying Playmaker",
        "group": "MID",
        "description": "Dictates tempo from deep with constant involvement and a wide passing range.",
    },
    "attacking_playmaker": {
        "label": "Attacking Playmaker (The 10)",
        "group": "MID",
        "description": "The creative hub — vision and dribbling to unlock low blocks.",
    },
    "target_man": {
        "label": "Target Man",
        "group": "FWD",
        "description": "Physical reference point — wins headers, holds the ball up, links play.",
    },
    "poacher": {
        "label": "Poacher",
        "group": "FWD",
        "description": "Lives off the last touch — rebounds, crosses, and in-the-box instinct.",
    },
    "inside_forward": {
        "label": "Inside Forward",
        "group": "FWD",
        "description": "Wide player who cuts inside onto the stronger foot to shoot and combine.",
    },
    "classic_winger": {
        "label": "Classic Winger",
        "group": "FWD",
        "description": "Stays wide, attacks the fullback on the outside, delivers crosses.",
    },
    "false_nine": {
        "label": "False Nine",
        "group": "FWD",
        "description": "Technical forward who drops deep, drags defenders out, creates space.",
    },
}

POSITION_CANDIDATES: dict[str, list[str]] = {
    "cb": ["ball_playing_defender", "stopper"],
    "rb": ["wing_back", "stopper", "ball_playing_defender"],
    "lb": ["wing_back", "stopper", "ball_playing_defender"],
    "cdm": ["holding_six", "deep_lying_playmaker", "box_to_box"],
    "cm": ["box_to_box", "deep_lying_playmaker", "holding_six", "attacking_playmaker"],
    "10": ["attacking_playmaker", "false_nine", "box_to_box"],
    "st": ["poacher", "target_man", "false_nine", "attacking_playmaker"],
    "lw": ["inside_forward", "classic_winger", "attacking_playmaker", "false_nine"],
    "rw": ["inside_forward", "classic_winger", "attacking_playmaker", "false_nine"],
}

# Normalization ceilings for raw features -> 0..1 traits. Starting points;
# recalibrate against real clips as data accumulates.
INVOLVEMENT_CEILING_PER_MIN = 6.0   # ball-near events/min
DISTANCE_CEILING_PER_MIN = 60.0     # player-heights/min
SPRINT_CEILING_PER_MIN = 3.0        # sprints/min
TOUCH_CEILING_PER_MIN = 8.0
PASS_CEILING_PER_MIN = 4.0
SHOT_CEILING_PER_MIN = 1.0
BALL_NEAR_HEIGHTS = 1.6             # "near" = within 1.6 player-heights
SPRINT_HEIGHTS_PER_SECOND = 2.5
SPRINT_MIN_SECONDS = 0.4


def _ball_track(cache: DetectionCache) -> dict[int, dict[str, Any]]:
    """One trusted ball per sampled frame ({center, size}), with turf-mark rejection.

    The detector fires on turf dots and line intersections — round, white, and
    ball-sized. The separating signal is motion: a real ball moves through
    camera-compensated space; a painted mark hosts "ball" detections at the SAME
    compensated position across much of the clip. Candidates in such static
    clusters are rejected; the per-frame winner among survivors is chosen by
    confidence plus trajectory continuity with the previous accepted ball.
    """
    ball_role = ROLE_INDEX["ball"]
    candidates: list[dict[str, Any]] = []
    for index in range(len(cache.det_role)):
        if int(cache.det_role[index]) != ball_role:
            continue
        confidence = float(cache.det_confidence[index])
        if confidence < BALL_MIN_CONFIDENCE:
            continue
        frame_position = int(cache.det_frame_index[index])
        box = cache.det_bbox[index]
        center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)
        size = max(4.0, float(max(box[2] - box[0], box[3] - box[1])))
        candidates.append(
            {
                "frame": int(cache.frames[frame_position]),
                "compensated": cache.world_point(center, frame_position),
                "center": center,
                "confidence": confidence,
                "size": size,
            }
        )
    if not candidates:
        return {}

    # Static-cluster rejection.
    total_frames = max(1, len(cache.frames))
    compensated = np.stack([candidate["compensated"] for candidate in candidates])
    frames_of = np.array([candidate["frame"] for candidate in candidates])
    keep: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates):
        radius = BALL_STATIC_RADIUS_FACTOR * candidate["size"]
        distances = np.linalg.norm(compensated - compensated[position], axis=1)
        nearby_frames = np.unique(frames_of[distances <= radius])
        if len(nearby_frames) / total_frames > BALL_STATIC_FRACTION:
            continue  # a mark painted on the pitch, not a ball
        keep.append(candidate)
    if not keep:
        return {}

    # Per-frame selection: confidence + continuity with the last accepted ball.
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for candidate in keep:
        by_frame.setdefault(candidate["frame"], []).append(candidate)
    selected: dict[int, dict[str, Any]] = {}
    previous_center: np.ndarray | None = None
    previous_frame: int | None = None
    for frame_number in sorted(by_frame):
        frame_candidates = by_frame[frame_number]
        if previous_center is not None and previous_frame is not None:
            elapsed = max(1, frame_number - previous_frame)

            def continuity(candidate: dict[str, Any]) -> float:
                jump = float(np.linalg.norm(candidate["center"] - previous_center))
                return candidate["confidence"] - 0.15 * min(
                    3.0, jump / (candidate["size"] * elapsed)
                )

            best = max(frame_candidates, key=continuity)
        else:
            best = max(frame_candidates, key=lambda candidate: candidate["confidence"])
        selected[frame_number] = {"center": best["center"], "size": best["size"]}
        previous_center = best["center"]
        previous_frame = frame_number
    return selected


def _player_x_centers(cache: DetectionCache, frame_position: int) -> list[float]:
    roles = {ROLE_INDEX["player"], ROLE_INDEX["goalkeeper"]}
    centers = []
    for index in cache.detections_at(frame_position):
        if int(cache.det_role[index]) in roles and bool(cache.det_on_pitch[index]):
            box = cache.det_bbox[index]
            centers.append(float((box[0] + box[2]) / 2))
    return centers


def _jersey_similarity(cache: DetectionCache, detection_index: int, reference: np.ndarray | None) -> float | None:
    if reference is None:
        return None
    jersey = cache.det_jersey[detection_index].astype(np.float32)
    if np.any(np.isnan(jersey)):
        return None
    return float(np.clip(np.dot(reference, jersey), 0.0, 1.0))


def _find_receiver(
    cache: DetectionCache,
    frame_number: int,
    ball_center: np.ndarray,
    own_box: np.ndarray | None,
    jersey_reference: np.ndarray | None,
) -> dict[str, Any] | None:
    """A player (other than ours) whose feet the ball has arrived at."""
    player_roles = {ROLE_INDEX["player"], ROLE_INDEX["goalkeeper"]}
    frame_position = cache.frame_position(frame_number)
    if int(cache.frames[frame_position]) != frame_number:
        return None
    for detection_index in cache.detections_at(frame_position):
        index = int(detection_index)
        if int(cache.det_role[index]) not in player_roles or not bool(cache.det_on_pitch[index]):
            continue
        box = cache.det_bbox[index]
        if own_box is not None:
            overlap_x = max(0.0, min(box[2], own_box[2]) - max(box[0], own_box[0]))
            overlap_y = max(0.0, min(box[3], own_box[3]) - max(box[1], own_box[1]))
            if overlap_x * overlap_y > 0.3 * (box[2] - box[0]) * (box[3] - box[1]):
                continue  # that's (mostly) our own player
        height = max(24.0, float(box[3] - box[1]))
        feet = np.array([(box[0] + box[2]) / 2, box[3] - 0.15 * height], dtype=np.float32)
        if float(np.linalg.norm(ball_center - feet)) <= 1.0 * height:
            return {
                "index": index,
                "role": int(cache.det_role[index]),
                "teammate": _jersey_similarity(cache, index, jersey_reference),
            }
    return None


def detect_ball_events(
    cache: DetectionCache,
    samples: list[dict[str, Any]],
    ball_track: dict[int, dict[str, Any]],
    jersey_reference: np.ndarray | None,
    fps: float,
) -> dict[str, int]:
    """Estimate passes and shots for the tracked player.

    Possession windows (ball near the feet for >= POSSESSION_MIN_SAMPLES sampled
    frames) are used only to locate each RELEASE; they are no longer reported as
    "touches" (too noisy on tiny-ball footage). Each release is classified:
      pass    -> the ball reaches a same-team player >= PASS_MIN_TRAVEL_HEIGHTS away
      shot    -> a fast release that converges on a detected GOALKEEPER
                 (high confidence: the one goal-directed signal we can observe)
      attempt -> any other fast release nobody on our team receives
                 (low confidence: also catches clearances/crosses; reported apart)
    """
    events = {"passes": 0, "shots": 0, "shot_attempts": 0}
    if not ball_track:
        return events

    boxes: dict[int, np.ndarray] = {}
    for sample in samples:
        bbox = sample["bbox"]
        boxes[sample["frame_number"]] = np.array(
            [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]], dtype=np.float32
        )
    shared_frames = sorted(set(boxes) & set(ball_track))
    if not shared_frames:
        return events

    stride = cache.stride

    def is_near(frame_number: int) -> bool:
        box = boxes[frame_number]
        height = max(24.0, float(box[3] - box[1]))
        feet = np.array([(box[0] + box[2]) / 2, box[3] - 0.15 * height], dtype=np.float32)
        ball = ball_track[frame_number]["center"]
        return float(np.linalg.norm(ball - feet)) <= TOUCH_RADIUS_HEIGHTS * height

    # Possession windows over shared frames (tolerate one missing sample).
    windows: list[tuple[int, int]] = []
    window_start: int | None = None
    previous_near_frame: int | None = None
    for frame_number in shared_frames:
        if is_near(frame_number):
            if window_start is None or (
                previous_near_frame is not None
                and frame_number - previous_near_frame > 3 * stride
            ):
                if window_start is not None and previous_near_frame is not None:
                    windows.append((window_start, previous_near_frame))
                window_start = frame_number
            previous_near_frame = frame_number
    if window_start is not None and previous_near_frame is not None:
        windows.append((window_start, previous_near_frame))
    windows = [
        (start, end)
        for start, end in windows
        if (end - start) // stride + 1 >= POSSESSION_MIN_SAMPLES
    ]
    # Windows are used only to locate each release; their count is NOT reported
    # as touches anymore. Possession volume is captured as "time with ball" in
    # segment_features, which degrades gracefully under intermittent detection.

    receive_window_frames = round(PASS_RECEIVE_WINDOW_SECONDS * fps)
    ball_frames = sorted(ball_track)
    goalkeeper_role = ROLE_INDEX["goalkeeper"]
    for _, release_frame in windows:
        release_box = boxes[release_frame]
        own_height = max(24.0, float(release_box[3] - release_box[1]))
        # Travel is measured from the passer's FEET at release, not from the
        # ball's last near-feet position. TOUCH_RADIUS_HEIGHTS is generous
        # (0.9h), so the window's last "near" frame lands well into the flight
        # — measuring from there made honest passes read ~1.5h and get rejected
        # by PASS_MIN_TRAVEL_HEIGHTS. The feet are the true launch origin.
        release_foot = np.array(
            [(release_box[0] + release_box[2]) / 2, release_box[3] - 0.15 * own_height],
            dtype=np.float32,
        )
        release_ball = ball_track[release_frame]["center"]
        flight = [
            frame
            for frame in ball_frames
            if release_frame < frame <= release_frame + receive_window_frames
        ]
        if not flight:
            continue

        def _travel_heights(frame: int) -> float:
            # Camera-compensated ball displacement from the passer's feet, in
            # player-heights. Compensation matters: a camera that pans to follow
            # the ball leaves the ball nearly static in raw pixels, which
            # collapses travel to ~0 and hides every pass and shot.
            world = ball_track[frame]["center"] - cache.transform_point(
                release_foot, release_frame, frame
            )
            return float(np.linalg.norm(world)) / own_height

        def _launch_heights(frame: int) -> float:
            # Ball's own compensated displacement since release (for launch speed).
            world = ball_track[frame]["center"] - cache.transform_point(
                release_ball, release_frame, frame
            )
            return float(np.linalg.norm(world)) / own_height

        # Scan the flight, but only once the ball has travelled the minimum pass
        # distance — so a teammate the ball merely brushes past early can't
        # disqualify a genuine pass downfield. A goalkeeper convergence is
        # decisive (a shot); a same-team receiver is a pass; an opponent
        # outfielder in the path is skipped (the ball may continue past them).
        reached_goalkeeper = False
        reached_teammate = False
        for frame in flight:
            if _travel_heights(frame) < PASS_MIN_TRAVEL_HEIGHTS:
                continue
            receiver = _find_receiver(
                cache, frame, ball_track[frame]["center"], boxes.get(frame), jersey_reference
            )
            if receiver is None:
                continue
            if receiver["role"] == goalkeeper_role:
                reached_goalkeeper = True
                break
            teammate = receiver["teammate"]
            # Same team when the jersey matches, or when jersey evidence is
            # unavailable on tiny crops (benefit of the doubt).
            if teammate is None or teammate >= PASS_TEAMMATE_MIN_JERSEY:
                reached_teammate = True
                break

        max_travel = max(_travel_heights(frame) for frame in flight)
        # Launch speed from the first flight observation (compensated).
        first = flight[0]
        elapsed_seconds = max(1e-6, (first - release_frame) / fps)
        speed_heights_per_sec = _launch_heights(first) / elapsed_seconds
        fast_release = speed_heights_per_sec >= SHOT_SPEED_HEIGHTS_PER_SEC

        if reached_goalkeeper and fast_release:
            events["shots"] += 1            # high-confidence, goalkeeper-directed
        elif reached_teammate:
            events["passes"] += 1
        elif fast_release and max_travel >= SHOT_ATTEMPT_MIN_TRAVEL_HEIGHTS:
            events["shot_attempts"] += 1    # low-confidence attempt (unreceived fast release)
    return events


def segment_features(video_id: str, segment: dict[str, Any], fps: float) -> dict[str, Any] | None:
    track = segment.get("focused_player_track")
    if not track or track.get("engine") != "tracklet":
        return None
    samples = [
        sample
        for sample in track["samples"]
        if sample.get("state") in ("tracked", "recovered") and sample.get("bbox")
    ]
    if len(samples) < 5:
        return None
    samples.sort(key=lambda sample: sample["frame_number"])

    cache = load_cache_unchecked(cache_path(video_id, segment["segment_id"]))
    ball_track = _ball_track(cache) if cache is not None else {}

    seconds_tracked = (samples[-1]["frame_number"] - samples[0]["frame_number"]) / fps
    distance_heights = 0.0
    sprint_run_seconds = 0.0
    sprint_count = 0
    ball_near_frames = 0
    ball_near_events = 0
    ball_near_run = 0
    possession_frames = 0  # "time with ball": ball within feet-radius of player
    wideness_values: list[float] = []

    previous_center: np.ndarray | None = None
    previous_frame: int | None = None
    for sample in samples:
        bbox = sample["bbox"]
        center = np.array(
            [(bbox["x1"] + bbox["x2"]) / 2, (bbox["y1"] + bbox["y2"]) / 2], dtype=np.float32
        )
        height = max(12.0, bbox["y2"] - bbox["y1"])
        frame_number = sample["frame_number"]

        if previous_center is not None and previous_frame is not None:
            gap = frame_number - previous_frame
            if 0 < gap <= 8:
                displacement = center - previous_center
                if cache is not None:
                    displacement = center - cache.transform_point(
                        previous_center, previous_frame, frame_number
                    )
                heights_moved = float(np.linalg.norm(displacement)) / height
                distance_heights += heights_moved
                speed = heights_moved / (gap / fps)
                if speed >= SPRINT_HEIGHTS_PER_SECOND:
                    sprint_run_seconds += gap / fps
                    if sprint_run_seconds >= SPRINT_MIN_SECONDS and sprint_run_seconds - gap / fps < SPRINT_MIN_SECONDS:
                        sprint_count += 1
                else:
                    sprint_run_seconds = 0.0

        ball = ball_track.get(frame_number)
        if ball is not None:
            if BALL_EVENTS_ENABLED:
                feet = np.array([center[0], bbox["y2"] - 0.15 * height], dtype=np.float32)
                if float(np.linalg.norm(ball["center"] - feet)) <= POSSESSION_RADIUS_HEIGHTS * height:
                    possession_frames += 1  # ball at the player's feet this sample
            if float(np.linalg.norm(ball["center"] - center)) <= BALL_NEAR_HEIGHTS * height:
                ball_near_frames += 1
                ball_near_run += 1
                if ball_near_run == 2:  # two consecutive near samples = one event
                    ball_near_events += 1
            else:
                ball_near_run = 0

        if cache is not None:
            frame_position = cache.frame_position(frame_number)
            if int(cache.frames[frame_position]) == frame_number:
                x_centers = _player_x_centers(cache, frame_position)
                if len(x_centers) >= 5:
                    rank = sum(1 for x in x_centers if x < float(center[0]))
                    percentile = rank / max(1, len(x_centers) - 1)
                    wideness_values.append(abs(percentile - 0.5) * 2)

        previous_center = center
        previous_frame = frame_number

    stride = track.get("frame_stride") or 2
    features = {
        "segment_id": segment["segment_id"],
        "seconds_tracked": round(seconds_tracked, 2),
        "distance_heights": round(distance_heights, 1),
        "sprint_count": sprint_count,
        "ball_near_events": ball_near_events,
        "wideness": round(float(np.mean(wideness_values)), 3) if wideness_values else None,
        "ball_data_available": bool(ball_track),
        "ball_events_enabled": BALL_EVENTS_ENABLED,
    }

    # Ball events (time-with-ball / passes / shots / attempts) — PAUSED unless
    # BALL_EVENTS_ENABLED. Movement stats above are always reported.
    if BALL_EVENTS_ENABLED:
        jersey_reference: np.ndarray | None = None
        selection = segment.get("focused_player_selection") or {}
        jersey_values = selection.get("jersey_descriptor")
        if jersey_values:
            jersey_reference = np.asarray(jersey_values, dtype=np.float32)
        events = (
            detect_ball_events(cache, samples, ball_track, jersey_reference, fps)
            if cache is not None
            else {"passes": 0, "shots": 0, "shot_attempts": 0}
        )
        features.update(
            {
                "time_with_ball_seconds": round(possession_frames * stride / fps, 2),
                "passes": events["passes"],
                "shots": events["shots"],
                "shot_attempts": events["shot_attempts"],
            }
        )
    return features


def aggregate_features(per_segment: list[dict[str, Any]]) -> dict[str, Any]:
    minutes = sum(features["seconds_tracked"] for features in per_segment) / 60
    minutes = max(minutes, 1e-6)
    wideness_values = [f["wideness"] for f in per_segment if f["wideness"] is not None]
    return {
        "minutes_tracked": round(minutes, 2),
        "segments_analyzed": len(per_segment),
        "involvement_per_min": round(sum(f["ball_near_events"] for f in per_segment) / minutes, 2),
        "distance_per_min": round(sum(f["distance_heights"] for f in per_segment) / minutes, 1),
        "sprints_per_min": round(sum(f["sprint_count"] for f in per_segment) / minutes, 2),
        "wideness": round(float(np.mean(wideness_values)), 3) if wideness_values else None,
        "time_with_ball_seconds": round(sum(f.get("time_with_ball_seconds", 0.0) for f in per_segment), 2),
        "passes": sum(f.get("passes", 0) for f in per_segment),
        "shots": sum(f.get("shots", 0) for f in per_segment),
        "shot_attempts": sum(f.get("shot_attempts", 0) for f in per_segment),
        "time_with_ball_per_min": round(
            sum(f.get("time_with_ball_seconds", 0.0) for f in per_segment) / minutes, 1
        ),
        "passes_per_min": round(sum(f.get("passes", 0) for f in per_segment) / minutes, 2),
        "shots_per_min": round(sum(f.get("shots", 0) for f in per_segment) / minutes, 2),
        "shot_attempts_per_min": round(sum(f.get("shot_attempts", 0) for f in per_segment) / minutes, 2),
        "ball_data_available": any(f["ball_data_available"] for f in per_segment),
        "ball_events_enabled": BALL_EVENTS_ENABLED,
    }


def _traits(features: dict[str, Any]) -> dict[str, float]:
    def clamp(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    traits = {
        "involvement": clamp(features["involvement_per_min"] / INVOLVEMENT_CEILING_PER_MIN),
        "work_rate": clamp(features["distance_per_min"] / DISTANCE_CEILING_PER_MIN),
        "explosiveness": clamp(features["sprints_per_min"] / SPRINT_CEILING_PER_MIN),
        "wideness": clamp(features["wideness"]) if features["wideness"] is not None else 0.5,
    }
    if features.get("ball_events_enabled") and features.get("ball_data_available"):
        traits["passing"] = clamp(features.get("passes_per_min", 0.0) / PASS_CEILING_PER_MIN)
        # Shooting weights confirmed (GK-directed) shots fully and low-confidence
        # attempts at half, so an aggressive shooter still registers without
        # letting clearances/crosses masquerade as clean shots.
        effective_shots_per_min = (
            features.get("shots_per_min", 0.0)
            + 0.5 * features.get("shot_attempts_per_min", 0.0)
        )
        traits["shooting"] = clamp(effective_shots_per_min / SHOT_CEILING_PER_MIN)
    return traits


def _archetype_scores(
    candidates: list[str],
    traits: dict[str, float],
    positions: list[str],
    footedness: str,
) -> list[dict[str, Any]]:
    involvement = traits["involvement"]
    work_rate = traits["work_rate"]
    explosiveness = traits["explosiveness"]
    wideness = traits["wideness"]
    passing = traits.get("passing", 0.0)
    shooting = traits.get("shooting", 0.0)

    formulas: dict[str, float] = {
        "ball_playing_defender": 0.40 * involvement + 0.20 * (1 - wideness) + 0.10 * work_rate + 0.20 * passing,
        "stopper": 0.40 * (1 - involvement) + 0.25 * (1 - wideness) + 0.15 * (1 - explosiveness),
        "wing_back": 0.35 * explosiveness + 0.30 * wideness + 0.20 * work_rate,
        "holding_six": 0.30 * (1 - wideness) + 0.25 * (1 - explosiveness) + 0.20 * involvement + 0.10 * passing,
        "deep_lying_playmaker": 0.35 * involvement + 0.25 * (1 - wideness) + 0.10 * (1 - explosiveness) + 0.20 * passing,
        "box_to_box": 0.40 * work_rate + 0.25 * explosiveness + 0.10 * involvement + 0.10 * shooting,
        "attacking_playmaker": 0.40 * involvement + 0.15 * (1 - wideness) + 0.10 * work_rate + 0.20 * passing,
        "target_man": 0.30 * (1 - work_rate) + 0.25 * (1 - explosiveness) + 0.20 * involvement + 0.10 * passing,
        "poacher": 0.30 * (1 - involvement) + 0.25 * explosiveness + 0.05 * (1 - work_rate) + 0.25 * shooting,
        "inside_forward": 0.25 * wideness + 0.20 * explosiveness + 0.15 * involvement + 0.20 * shooting,
        "classic_winger": 0.30 * wideness + 0.25 * explosiveness + 0.10 * (1 - involvement) + 0.10 * passing,
        "false_nine": 0.35 * involvement + 0.15 * (1 - wideness) + 0.10 * work_rate + 0.15 * passing,
    }

    # Footedness shapes the winger split: opposite foot -> cuts inside.
    winger_side = {"lw": "right", "rw": "left"}  # opposite foot per flank
    for position in positions:
        expected_opposite = winger_side.get(position)
        if expected_opposite is None:
            continue
        if footedness == expected_opposite or footedness == "both":
            formulas["inside_forward"] += 0.20
        else:
            formulas["classic_winger"] += 0.20

    scored = [
        {
            "archetype": code,
            "label": ARCHETYPES[code]["label"],
            "score": round(float(np.clip(formulas[code] + 0.15, 0.0, 1.0)), 3),  # +0.15 position-fit base
        }
        for code in candidates
    ]
    scored.sort(key=lambda entry: entry["score"], reverse=True)
    return scored


def _evidence(traits: dict[str, float], features: dict[str, Any], footedness: str) -> list[str]:
    lines: list[str] = []
    involvement, work_rate = traits["involvement"], traits["work_rate"]
    explosiveness, wideness = traits["explosiveness"], traits["wideness"]
    if features["ball_data_available"]:
        if involvement >= 0.6:
            lines.append(
                f"Heavily involved: near the ball {features['involvement_per_min']}x per minute."
            )
        elif involvement <= 0.3:
            lines.append(
                "Selective involvement — most active away from the ball, arriving for moments."
            )
        if features.get("ball_events_enabled"):
            time_with_ball = features.get("time_with_ball_seconds")
            if time_with_ball:
                lines.append(
                    f"Time on the ball: ~{time_with_ball}s across tracked clips (ball near the feet)."
                )
            event_bits = []
            if features.get("passes"):
                event_bits.append(f"{features['passes']} passes to teammates")
            if features.get("shots"):
                shots = features["shots"]
                event_bits.append(f"{shots} shot{'s' if shots != 1 else ''} on the keeper")
            if features.get("shot_attempts"):
                attempts = features["shot_attempts"]
                event_bits.append(
                    f"{attempts} shot attempt{'s' if attempts != 1 else ''} (lower confidence)"
                )
            if event_bits:
                lines.append(
                    f"Estimated on-ball events: {', '.join(event_bits)} "
                    "(from ball trajectory — treat as estimates)."
                )
    else:
        lines.append("Ball detections unavailable in these clips — involvement and events not scored.")
    if work_rate >= 0.6:
        lines.append(f"High work rate: {features['distance_per_min']} body-lengths covered per minute.")
    if explosiveness >= 0.6:
        lines.append(f"Explosive: {features['sprints_per_min']} sprints per minute.")
    elif explosiveness <= 0.25:
        lines.append("Plays at a controlled tempo with few all-out sprints.")
    if features["wideness"] is not None:
        if wideness >= 0.6:
            lines.append("Consistently positioned wide relative to teammates on screen.")
        elif wideness <= 0.35:
            lines.append("Operates centrally relative to teammates on screen.")
    lines.append(f"Declared footedness: {footedness}.")
    return lines


def build_player_profile(video_id: str, result: dict[str, Any]) -> dict[str, Any]:
    info = result.get("player_info") or {}
    positions = [p for p in info.get("positions", []) if p in FIELD_POSITIONS]
    footedness = info.get("footedness", "right")

    fps = result["metadata"]["fps"]
    per_segment = []
    for segment in result["segments"]:
        features = segment_features(video_id, segment, fps)
        if features is not None:
            per_segment.append(features)
    if not per_segment:
        return {
            "status": "insufficient_data",
            "message": "Track your player in at least one clip first, then generate the profile.",
        }

    features = aggregate_features(per_segment)
    traits = _traits(features)

    candidates: list[str] = []
    for position in positions:
        for code in POSITION_CANDIDATES.get(position, []):
            if code not in candidates:
                candidates.append(code)
    if not candidates:
        candidates = list(ARCHETYPES.keys())

    scores = _archetype_scores(candidates, traits, positions, footedness)
    primary = scores[0]
    margin = primary["score"] - (scores[1]["score"] if len(scores) > 1 else 0.0)

    if features["minutes_tracked"] < 0.5:
        confidence = "low"
    elif features["minutes_tracked"] < 2 or margin < 0.05:
        confidence = "medium"
    else:
        confidence = "high"

    return {
        "status": "completed",
        "primary_archetype": primary["archetype"],
        "primary_label": primary["label"],
        "description": ARCHETYPES[primary["archetype"]]["description"],
        "group": ARCHETYPES[primary["archetype"]]["group"],
        "confidence": confidence,
        "scores": scores,
        "traits": {key: round(value, 3) for key, value in traits.items()},
        "features": features,
        "per_segment": per_segment,
        "evidence": _evidence(traits, features, footedness),
        "positions": positions,
        "footedness": footedness,
        "note": (
            "v2 profile from movement, involvement, positioning, time with the ball, "
            "and estimated events (passes and shots from ball trajectory). Shots are "
            "counted only when a fast release reaches the goalkeeper; other fast "
            "releases are reported as lower-confidence attempts. Events are estimates — "
            "the goal is not visible and small fast balls are hard to detect. The "
            "profile sharpens as more clips are tracked."
        ),
    }
