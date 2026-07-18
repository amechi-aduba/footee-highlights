from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException

from app.core.config import (
    APPEARANCE_MAX_OVERLAP_IOU,
    APPEARANCE_TOP_K_CROPS,
    APPEARANCE_MIN_CROP_WIDTH,
    CAMERA_CONFIDENCE_MIN,
    CAMERA_UNCERTAIN_GATE_FACTOR,
    CROSSING_COENDER_WINDOW_FRAMES,
    CROSSING_JOINT_MARGIN,
    INTERPOLATE_MAX_DISTANCE_HEIGHTS,
    INTERPOLATE_MAX_GAP_FRAMES,
    MAX_PLAYER_SPEED_HEIGHTS_PER_SEC,
    RECOVERY_ENABLED,
    RECOVERY_EXCLUDE_TRACKLET_LENGTH,
    RECOVERY_MAX_CONSECUTIVE_MISSES,
    RECOVERY_MAX_DISTANCE_HEIGHTS,
    RECOVERY_MAX_OVERLAP_IOU,
    RECOVERY_MIN_JERSEY_SIMILARITY,
    RECOVERY_SIZE_BAND,
    RECOVERY_UNIQUENESS_RATIO,
    SEARCH_TIMEOUT_FRAMES,
    STITCH_APPEARANCE_WEIGHT,
    STITCH_CROSS_TEAM_APPEARANCE_OVERRIDE,
    STITCH_TIEBREAK_APPEARANCE_MARGIN,
    STITCH_TIEBREAK_MIN_APPEARANCE,
    STITCH_LINK_THRESHOLD,
    STITCH_LONG_GAP_FRAMES,
    STITCH_LONG_GAP_MIN_APPEARANCE,
    STITCH_LONG_GAP_MIN_JERSEY,
    STITCH_MAX_GAP_FRAMES,
    STITCH_POSITION_WEIGHT,
    STITCH_SIZE_WEIGHT,
    STITCH_TEAM_WEIGHT,
    STITCH_WINNER_MARGIN,
    TRACKING_BOX_SMOOTHING_ALPHA,
    TRACKING_MAX_REASSOCIATION_FRAME_FRACTION,
    TRACKLET_AMBIGUITY_RATIO,
    TRACKLET_MAX_MISSED_SAMPLES,
    TRACKLET_MIN_IOU,
    TRACKLET_MIN_LENGTH,
)
from app.services.detection_cache import ROLE_INDEX, DetectionCache, get_or_build_cache


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def _center(bbox: np.ndarray) -> np.ndarray:
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)


def _fps(cache: DetectionCache) -> float:
    return float(cache.header.get("fps") or 30.0)


def _speed_cap(cache: DetectionCache, from_frame: int, to_frame: int) -> float:
    """Kinematic ceiling in player-heights/second for an association spanning
    these frames. Under LOW camera confidence the cap SHRINKS — the camera model
    is guesswork exactly when a hard zoom/whip pan is happening, so only
    near-certain associations may survive (the old code loosened here, which is
    how zoom drift slipped through)."""
    cap = MAX_PLAYER_SPEED_HEIGHTS_PER_SEC
    if cache.min_camera_confidence(from_frame, to_frame) < CAMERA_CONFIDENCE_MIN:
        cap *= CAMERA_UNCERTAIN_GATE_FACTOR
    return cap


def _implied_speed(
    cache: DetectionCache,
    from_frame: int,
    from_center: np.ndarray,
    to_frame: int,
    to_center: np.ndarray,
    height_at_to: float,
) -> float:
    """Player speed this association implies, in camera-compensated
    player-heights/second (measured in the destination frame's scale)."""
    if to_frame == from_frame:
        return 0.0
    mapped = cache.transform_point(from_center, from_frame, to_frame)
    distance = float(np.linalg.norm(np.asarray(to_center, dtype=np.float32) - mapped))
    dt = abs(to_frame - from_frame) / _fps(cache)
    return distance / max(height_at_to, 1.0) / max(dt, 1e-6)


def _size_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    if first_area <= 0 or second_area <= 0:
        return 0.0
    return min(first_area, second_area) / max(first_area, second_area)


@dataclass
class Tracklet:
    tracklet_id: int
    frames: list[int] = field(default_factory=list)          # source frame numbers, observed only
    boxes: list[np.ndarray] = field(default_factory=list)    # parallel to frames
    confidences: list[float] = field(default_factory=list)
    detection_indices: list[int] = field(default_factory=list)
    mean_appearance: np.ndarray | None = None                 # from the K largest crops only
    jersey_descriptor: np.ndarray | None = None
    team_id: int | None = None
    velocity_start: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    velocity_end: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))

    @property
    def start_frame(self) -> int:
        return self.frames[0]

    @property
    def end_frame(self) -> int:
        return self.frames[-1]

    @property
    def length(self) -> int:
        return len(self.frames)


def _endpoint_velocity(
    tracklet: Tracklet,
    cache: DetectionCache,
    from_start: bool,
    window: int = 5,
) -> np.ndarray:
    """Camera-compensated px/source-frame velocity near one end of a tracklet."""
    if tracklet.length < 2:
        return np.zeros(2, dtype=np.float32)
    if from_start:
        frames = tracklet.frames[: window + 1]
        boxes = tracklet.boxes[: window + 1]
    else:
        frames = tracklet.frames[-(window + 1):]
        boxes = tracklet.boxes[-(window + 1):]
    elapsed = max(1, frames[-1] - frames[0])
    # Player motion = final position minus where the STARTING position would sit
    # in the final frame if the player were static (full-affine warp, not a
    # translation subtraction — the latter is wrong under zoom).
    start_mapped = cache.transform_point(_center(boxes[0]), frames[0], frames[-1])
    return ((_center(boxes[-1]) - start_mapped) / elapsed).astype(np.float32)


def _detection_is_contaminated(
    cache: DetectionCache,
    detection_index: int,
    max_overlap: float,
) -> bool:
    """True when this detection's box overlaps another player's box — the crop
    contains BOTH players' pixels. Descriptors from such crops are exactly what
    causes wrong relinks after crossings, so they must never represent identity."""
    player_roles = {ROLE_INDEX["player"], ROLE_INDEX["goalkeeper"]}
    frame_position = int(cache.det_frame_index[detection_index])
    own_box = cache.det_bbox[detection_index]
    for other_index in cache.detections_at(frame_position):
        other = int(other_index)
        if other == detection_index or int(cache.det_role[other]) not in player_roles:
            continue
        if _iou(own_box, cache.det_bbox[other]) > max_overlap:
            return True
    return False


def _finalize_tracklets(raw_tracklets: list[Tracklet], cache: DetectionCache) -> list[Tracklet]:
    tracklets = [tracklet for tracklet in raw_tracklets if tracklet.length >= TRACKLET_MIN_LENGTH]

    for tracklet in tracklets:
        # Appearance: only the K widest crops, and only crops wide enough to carry signal.
        # Tiny 8x20 detections contribute nothing but noise — skip them honestly.
        # Contaminated crops (overlapping another player) are skipped too.
        widths = np.array([box[2] - box[0] for box in tracklet.boxes])
        order = np.argsort(widths)[::-1][: APPEARANCE_TOP_K_CROPS * 2]
        appearance_rows = []
        jersey_rows = []
        for position in order:
            if len(appearance_rows) >= APPEARANCE_TOP_K_CROPS and len(jersey_rows) >= APPEARANCE_TOP_K_CROPS:
                break
            detection_index = tracklet.detection_indices[position]
            if _detection_is_contaminated(cache, detection_index, APPEARANCE_MAX_OVERLAP_IOU):
                continue
            if widths[position] >= APPEARANCE_MIN_CROP_WIDTH and len(appearance_rows) < APPEARANCE_TOP_K_CROPS:
                row = cache.det_appearance[detection_index].astype(np.float32)
                if not np.any(np.isnan(row)):
                    appearance_rows.append(row)
            if len(jersey_rows) < APPEARANCE_TOP_K_CROPS:
                jersey_row = cache.det_jersey[detection_index].astype(np.float32)
                if not np.any(np.isnan(jersey_row)):
                    jersey_rows.append(jersey_row)
        if appearance_rows:
            mean = np.mean(appearance_rows, axis=0)
            tracklet.mean_appearance = mean / (np.linalg.norm(mean) + 1e-8)
        if jersey_rows:
            mean = np.mean(jersey_rows, axis=0)
            tracklet.jersey_descriptor = mean / (np.linalg.norm(mean) + 1e-8)
        tracklet.velocity_start = _endpoint_velocity(tracklet, cache, from_start=True)
        tracklet.velocity_end = _endpoint_velocity(tracklet, cache, from_start=False)

    _assign_teams(tracklets)
    return tracklets


def _assign_teams(tracklets: list[Tracklet]) -> None:
    with_jersey = [tracklet for tracklet in tracklets if tracklet.jersey_descriptor is not None]
    if len(with_jersey) < 4:
        return
    features = np.stack([tracklet.jersey_descriptor for tracklet in with_jersey]).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    _, labels, _ = cv2.kmeans(features, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    for tracklet, label in zip(with_jersey, labels.flatten()):
        tracklet.team_id = int(label)


def build_tracklets(cache: DetectionCache) -> list[Tracklet]:
    """Conservative tracklet building over cached detections.

    Strict IoU gate, camera-compensated prediction, and — critically — tracklets
    TERMINATE at ambiguity instead of guessing. A crossing should end tracklets;
    the stitcher resolves it later with full-clip context.
    """
    player_roles = {ROLE_INDEX["player"], ROLE_INDEX["goalkeeper"]}
    next_id = 0
    active: list[dict[str, Any]] = []   # {tracklet, missed, velocity}
    finished: list[Tracklet] = []

    for frame_position in range(len(cache.frames)):
        frame_number = int(cache.frames[frame_position])
        # The pitch mask only gates NEW tracklets. An existing tracklet may keep
        # matching detections flagged off-pitch — a mask false-negative must not
        # kill a live track (a visible player would lose their box otherwise).
        detection_indices = [
            int(index)
            for index in cache.detections_at(frame_position)
            if int(cache.det_role[index]) in player_roles
        ]
        boxes = {index: cache.det_bbox[index] for index in detection_indices}

        # Predict each active tracklet's bbox at this frame (camera-compensated).
        scores: dict[tuple[int, int], float] = {}
        predictions: dict[int, np.ndarray] = {}
        for slot, state in enumerate(active):
            tracklet: Tracklet = state["tracklet"]
            gap = frame_number - tracklet.end_frame
            # Extrapolate the player in the LAST-SEEN frame's coordinates, then
            # warp the whole box into the current frame — under a zoom the box
            # scales with the camera, keeping the IoU gate meaningful.
            moved = tracklet.boxes[-1].copy()
            displacement = state["velocity"] * gap
            moved[[0, 2]] += displacement[0]
            moved[[1, 3]] += displacement[1]
            predicted = cache.transform_box(moved, tracklet.end_frame, frame_number)
            predictions[slot] = predicted
            speed_cap = _speed_cap(cache, tracklet.end_frame, frame_number)
            last_center = _center(tracklet.boxes[-1])
            for detection_index in detection_indices:
                overlap = _iou(predicted, boxes[detection_index])
                if overlap < TRACKLET_MIN_IOU:
                    continue
                # Kinematic gate: even an overlapping box is rejected when it
                # would imply superhuman speed (bad camera estimate / ID trap).
                candidate_box = boxes[detection_index]
                candidate_height = max(24.0, float(candidate_box[3] - candidate_box[1]))
                if (
                    _implied_speed(
                        cache,
                        tracklet.end_frame,
                        last_center,
                        frame_number,
                        _center(candidate_box),
                        candidate_height,
                    )
                    > speed_cap
                ):
                    continue
                scores[(slot, detection_index)] = overlap

        # Ambiguity detection: a tracklet with two close candidates, or a detection
        # contested by two close tracklets, terminates the tracklet(s) involved.
        ambiguous_slots: set[int] = set()
        by_slot: dict[int, list[float]] = {}
        by_detection: dict[int, list[tuple[int, float]]] = {}
        for (slot, detection_index), overlap in scores.items():
            by_slot.setdefault(slot, []).append(overlap)
            by_detection.setdefault(detection_index, []).append((slot, overlap))
        for slot, values in by_slot.items():
            if len(values) >= 2:
                ranked = sorted(values, reverse=True)
                if ranked[0] / max(ranked[1], 1e-6) < TRACKLET_AMBIGUITY_RATIO:
                    ambiguous_slots.add(slot)
        for detection_index, entries in by_detection.items():
            if len(entries) >= 2:
                ranked = sorted(entries, key=lambda entry: entry[1], reverse=True)
                if ranked[0][1] / max(ranked[1][1], 1e-6) < TRACKLET_AMBIGUITY_RATIO:
                    ambiguous_slots.update(slot for slot, _ in ranked[:2])

        # Greedy assignment among unambiguous pairs.
        assigned_slots: set[int] = set()
        assigned_detections: set[int] = set()
        for (slot, detection_index), overlap in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            if slot in ambiguous_slots or slot in assigned_slots or detection_index in assigned_detections:
                continue
            state = active[slot]
            tracklet = state["tracklet"]
            gap = max(1, frame_number - tracklet.end_frame)
            last_mapped = cache.transform_point(
                _center(tracklet.boxes[-1]), tracklet.end_frame, frame_number
            )
            measured_velocity = (_center(boxes[detection_index]) - last_mapped) / gap
            state["velocity"] = (
                0.6 * state["velocity"] + 0.4 * measured_velocity
            ).astype(np.float32)
            tracklet.frames.append(frame_number)
            tracklet.boxes.append(boxes[detection_index].copy())
            tracklet.confidences.append(float(cache.det_confidence[detection_index]))
            tracklet.detection_indices.append(detection_index)
            state["missed"] = 0
            assigned_slots.add(slot)
            assigned_detections.add(detection_index)

        # Close ambiguous and stale tracklets.
        surviving: list[dict[str, Any]] = []
        for slot, state in enumerate(active):
            if slot in ambiguous_slots:
                finished.append(state["tracklet"])
                continue
            if slot not in assigned_slots:
                state["missed"] += 1
                if state["missed"] > TRACKLET_MAX_MISSED_SAMPLES:
                    finished.append(state["tracklet"])
                    continue
            surviving.append(state)
        active = surviving

        # Unmatched ON-PITCH detections start fresh tracklets (crowd stays out).
        for detection_index in detection_indices:
            if detection_index in assigned_detections or not bool(cache.det_on_pitch[detection_index]):
                continue
            tracklet = Tracklet(tracklet_id=next_id)
            next_id += 1
            tracklet.frames.append(frame_number)
            tracklet.boxes.append(boxes[detection_index].copy())
            tracklet.confidences.append(float(cache.det_confidence[detection_index]))
            tracklet.detection_indices.append(detection_index)
            active.append({"tracklet": tracklet, "missed": 0, "velocity": np.zeros(2, dtype=np.float32)})

    finished.extend(state["tracklet"] for state in active)
    return _finalize_tracklets(finished, cache)


def select_anchor(
    tracklets: list[Tracklet],
    cache: DetectionCache,
    selected_frame: int,
    selected_bbox: dict[str, float],
) -> Tracklet | None:
    """The tracklet containing the user's click. IoU first, center distance fallback."""
    target = np.array(
        [selected_bbox["x1"], selected_bbox["y1"], selected_bbox["x2"], selected_bbox["y2"]],
        dtype=np.float32,
    )
    tolerance = cache.stride * 3
    best: tuple[float, Tracklet] | None = None
    for tracklet in tracklets:
        for frame_number, box in zip(tracklet.frames, tracklet.boxes):
            if abs(frame_number - selected_frame) > tolerance:
                continue
            overlap = _iou(target, box)
            if overlap > 0.1 and (best is None or overlap > best[0]):
                best = (overlap, tracklet)
    if best is not None:
        return best[1]

    # Fallback: nearest center within 1.5 box heights around the selected time.
    target_center = _center(target)
    box_height = max(1.0, float(target[3] - target[1]))
    nearest: tuple[float, Tracklet] | None = None
    for tracklet in tracklets:
        for frame_number, box in zip(tracklet.frames, tracklet.boxes):
            if abs(frame_number - selected_frame) > tolerance:
                continue
            distance = float(np.linalg.norm(_center(box) - target_center))
            if distance <= 1.5 * box_height and (nearest is None or distance < nearest[0]):
                nearest = (distance, tracklet)
    return nearest[1] if nearest else None


def _link_cost(
    earlier: Tracklet,
    later: Tracklet,
    cache: DetectionCache,
    diagonal: float,
    backward: bool,
) -> tuple[float | None, dict[str, Any]]:
    """Cost of linking two tracklets across a gap; None = hard reject."""
    gap = later.start_frame - earlier.end_frame
    detail: dict[str, Any] = {"from": earlier.tracklet_id, "to": later.tracklet_id, "gap": gap}
    if gap <= 0 or gap > STITCH_MAX_GAP_FRAMES:
        return None, detail

    # Position: extrapolate in the source frame's coordinates, then warp the
    # prediction into the frame where the comparison happens (full affine —
    # correct under zoom, unlike translation differencing).
    if backward:
        extrapolated = _center(later.boxes[0]) - later.velocity_start * gap
        predicted = cache.transform_point(extrapolated, later.start_frame, earlier.end_frame)
        actual = _center(earlier.boxes[-1])
        reference_box = later.boxes[0]
        gate_scale = cache.scale_between(later.start_frame, earlier.end_frame)
        comparison_frames = (later.start_frame, earlier.end_frame)
    else:
        extrapolated = _center(earlier.boxes[-1]) + earlier.velocity_end * gap
        predicted = cache.transform_point(extrapolated, earlier.end_frame, later.start_frame)
        actual = _center(later.boxes[0])
        reference_box = earlier.boxes[-1]
        gate_scale = cache.scale_between(earlier.end_frame, later.start_frame)
        comparison_frames = (earlier.end_frame, later.start_frame)
    box_height = max(24.0, float(reference_box[3] - reference_box[1]) * gate_scale)
    gate = min(box_height * (2.0 + 0.1 * gap), diagonal * TRACKING_MAX_REASSOCIATION_FRAME_FRACTION)
    # Uncertain camera model across the gap => SHRINK the gate (never widen).
    camera_confidence = cache.min_camera_confidence(*comparison_frames)
    if camera_confidence < CAMERA_CONFIDENCE_MIN:
        gate *= CAMERA_UNCERTAIN_GATE_FACTOR
        detail["camera_uncertain"] = round(camera_confidence, 3)
    distance = float(np.linalg.norm(predicted - actual))
    position_cost = distance / max(gate, 1e-6)
    detail["position_cost"] = round(position_cost, 3)
    if position_cost > 1.0:
        return None, detail

    # Kinematic gate: reject links implying superhuman speed. A cross-pitch jump
    # under a zoom used to look "close" to the translation-only model; it cannot
    # pass a physical speed cap.
    implied = _implied_speed(
        cache,
        earlier.end_frame,
        _center(earlier.boxes[-1]),
        later.start_frame,
        _center(later.boxes[0]),
        max(24.0, float(later.boxes[0][3] - later.boxes[0][1])),
    )
    detail["implied_speed"] = round(implied, 2)
    if implied > _speed_cap(cache, earlier.end_frame, later.start_frame):
        detail["rejected"] = "kinematic"
        return None, detail

    # Appearance (may be unavailable on tiny-crop tracklets — weights renormalize).
    appearance_similarity: float | None = None
    if earlier.mean_appearance is not None and later.mean_appearance is not None:
        appearance_similarity = float(
            np.clip(np.dot(earlier.mean_appearance, later.mean_appearance), 0.0, 1.0)
        )
    detail["appearance"] = round(appearance_similarity, 3) if appearance_similarity is not None else None
    if gap > STITCH_LONG_GAP_FRAMES:
        # Long gaps demand identity evidence, not just plausible position.
        if appearance_similarity is not None:
            if appearance_similarity < STITCH_LONG_GAP_MIN_APPEARANCE:
                return None, detail
        else:
            # Tiny/distant players have no appearance descriptor. Instead of
            # abandoning them (box disappears while the player is visible),
            # accept jersey-color agreement + a tight position gate as evidence.
            jersey_similarity: float | None = None
            if earlier.jersey_descriptor is not None and later.jersey_descriptor is not None:
                jersey_similarity = float(
                    np.clip(np.dot(earlier.jersey_descriptor, later.jersey_descriptor), 0.0, 1.0)
                )
            detail["jersey"] = round(jersey_similarity, 3) if jersey_similarity is not None else None
            if (
                jersey_similarity is None
                or jersey_similarity < STITCH_LONG_GAP_MIN_JERSEY
                or position_cost > 0.5
            ):
                return None, detail

    # Team color: semi-hard. A different team is fatal UNLESS appearance strongly
    # agrees it is the same player. With OSNet embeddings, different players score
    # genuinely low here, so this gate is what stops the box drifting across teams.
    team_cost = 0.0
    if earlier.team_id is not None and later.team_id is not None:
        if earlier.team_id != later.team_id:
            team_cost = 1.0
            if appearance_similarity is None or appearance_similarity < STITCH_CROSS_TEAM_APPEARANCE_OVERRIDE:
                detail["rejected"] = "team_mismatch"
                return None, detail
    detail["team_cost"] = team_cost

    size_cost = 1.0 - _size_similarity(earlier.boxes[-1], later.boxes[0])

    weights = {
        "pos": STITCH_POSITION_WEIGHT,
        "app": STITCH_APPEARANCE_WEIGHT if appearance_similarity is not None else 0.0,
        "team": STITCH_TEAM_WEIGHT,
        "size": STITCH_SIZE_WEIGHT,
    }
    total_weight = sum(weights.values())
    cost = (
        weights["pos"] * position_cost
        + weights["app"] * (1.0 - (appearance_similarity or 0.0))
        + weights["team"] * team_cost
        + weights["size"] * size_cost
    ) / max(total_weight, 1e-6)
    detail["cost"] = round(cost, 3)
    return cost, detail


def _find_co_ender(
    frontier: Tracklet,
    all_tracklets: list[Tracklet] | None,
    exclude_ids: set[int],
    backward: bool,
) -> Tracklet | None:
    """Another tracklet that ends when and where the frontier ends = a crossing."""
    if not all_tracklets:
        return None
    if backward:
        end_frame = frontier.start_frame
        end_center = _center(frontier.boxes[0])
        end_height = max(24.0, float(frontier.boxes[0][3] - frontier.boxes[0][1]))
    else:
        end_frame = frontier.end_frame
        end_center = _center(frontier.boxes[-1])
        end_height = max(24.0, float(frontier.boxes[-1][3] - frontier.boxes[-1][1]))
    for tracklet in all_tracklets:
        if tracklet.tracklet_id in exclude_ids:
            continue
        other_frame = tracklet.start_frame if backward else tracklet.end_frame
        other_center = _center(tracklet.boxes[0] if backward else tracklet.boxes[-1])
        if (
            abs(other_frame - end_frame) <= CROSSING_COENDER_WINDOW_FRAMES
            and float(np.linalg.norm(other_center - end_center)) <= 3.0 * end_height
        ):
            return tracklet
    return None


def _joint_crossing_choice(
    frontier: Tracklet,
    top_two: list[tuple[float, dict[str, Any], Tracklet]],
    all_tracklets: list[Tracklet] | None,
    cache: DetectionCache,
    diagonal: float,
    backward: bool,
) -> tuple[float, dict[str, Any], Tracklet] | None:
    """Resolve a crossing's 2x2 pairing jointly.

    Greedy linking assigns the frontier its individually best continuation even
    when that steals the OTHER crossing player's continuation. Comparing the two
    total pairings (keep vs swap) fixes the classic crossing identity switch."""
    (first_cost, first_detail, first), (second_cost, second_detail, second) = top_two
    exclude = {frontier.tracklet_id, first.tracklet_id, second.tracklet_id}
    co_ender = _find_co_ender(frontier, all_tracklets, exclude, backward)
    if co_ender is None:
        return None

    def link(earlier: Tracklet, later: Tracklet) -> float:
        if backward:
            earlier, later = later, earlier
        cost, _ = _link_cost(earlier, later, cache, diagonal, backward=False)
        return cost if cost is not None else 2.0  # infeasible = heavy penalty

    keep_total = link(frontier, first) + link(co_ender, second)
    swap_total = link(frontier, second) + link(co_ender, first)
    if swap_total + CROSSING_JOINT_MARGIN < keep_total:
        second_detail["crossing_resolved"] = {"co_ender": co_ender.tracklet_id, "swapped": True}
        return (second_cost, second_detail, second)
    if keep_total + CROSSING_JOINT_MARGIN < swap_total:
        first_detail["crossing_resolved"] = {"co_ender": co_ender.tracklet_id, "swapped": False}
        return (first_cost, first_detail, first)
    return None  # genuinely ambiguous even jointly — leave the gap


def _best_link(
    frontier: Tracklet,
    candidates: list[Tracklet],
    cache: DetectionCache,
    diagonal: float,
    backward: bool,
    decisions: list[dict[str, Any]],
    all_tracklets: list[Tracklet] | None = None,
) -> Tracklet | None:
    """The single best acceptable link from a frontier, or None. Ambiguous links
    are left as gaps — an honest gap beats a confident mistake."""
    if backward:
        scored = [
            (_link_cost(candidate, frontier, cache, diagonal, backward=True), candidate)
            for candidate in candidates
        ]
    else:
        scored = [
            (_link_cost(frontier, candidate, cache, diagonal, backward=False), candidate)
            for candidate in candidates
        ]
    viable = [
        (cost, detail, candidate)
        for (cost, detail), candidate in scored
        if cost is not None and cost <= STITCH_LINK_THRESHOLD
    ]
    if not viable:
        return None
    viable.sort(key=lambda entry: entry[0])
    best_cost, best_detail, best_candidate = viable[0]
    if len(viable) > 1 and viable[1][0] > 1e-6 and best_cost / viable[1][0] > STITCH_WINNER_MARGIN:
        # Ambiguous top-2 = usually a crossing. First try resolving the pairing
        # JOINTLY with the other crossing player's tracklet — the frontier's
        # individually-best continuation may belong to the occluder.
        joint = _joint_crossing_choice(
            frontier, viable[:2], all_tracklets, cache, diagonal, backward
        )
        if joint is not None:
            best_cost, best_detail, best_candidate = joint
        else:
            # Appearance tiebreak: costs are close, but if the best candidate
            # carries decisively stronger identity evidence, accept it rather
            # than abandoning the whole chain direction here.
            best_appearance = best_detail.get("appearance")
            runner_appearance = viable[1][1].get("appearance")
            tiebreak = (
                best_appearance is not None
                and best_appearance >= STITCH_TIEBREAK_MIN_APPEARANCE
                and (
                    runner_appearance is None
                    or best_appearance - runner_appearance >= STITCH_TIEBREAK_APPEARANCE_MARGIN
                )
            )
            if not tiebreak:
                best_detail["rejected"] = "no_clear_winner"
                decisions.append(best_detail)
                return None
            best_detail["tiebreak"] = "appearance"
    best_detail["accepted"] = True
    decisions.append(best_detail)
    return best_candidate


def stitch_tracklets(
    tracklets: list[Tracklet],
    anchors: Tracklet | list[Tracklet],
    cache: DetectionCache,
) -> tuple[list[Tracklet], list[dict[str, Any]]]:
    """Greedy chaining through one or more user-confirmed anchor tracklets.

    Every anchor is ground truth — the user clicked the player there — so all
    anchors are in the chain unconditionally, identity is pinned on both sides
    of each one, and stitching only has to bridge BETWEEN confirmed points
    (plus extend before the first and after the last)."""
    if isinstance(anchors, Tracklet):
        anchors = [anchors]
    unique: dict[int, Tracklet] = {anchor.tracklet_id: anchor for anchor in anchors}
    anchors = sorted(unique.values(), key=lambda tracklet: tracklet.start_frame)
    diagonal = float(
        np.hypot(cache.header.get("frame_width") or 1920, cache.header.get("frame_height") or 1080)
    )
    used = {anchor.tracklet_id for anchor in anchors}
    chain = list(anchors)
    decisions: list[dict[str, Any]] = []

    # Bridge the interval between each pair of consecutive anchors.
    for earlier_anchor, later_anchor in zip(anchors, anchors[1:]):
        frontier = earlier_anchor
        while frontier.end_frame < later_anchor.start_frame:
            candidates = [
                tracklet
                for tracklet in tracklets
                if tracklet.tracklet_id not in used
                and tracklet.start_frame > frontier.end_frame
                and tracklet.end_frame < later_anchor.start_frame
            ]
            best = _best_link(frontier, candidates, cache, diagonal, False, decisions, tracklets)
            if best is None:
                break
            used.add(best.tracklet_id)
            chain.append(best)
            frontier = best

    # Extend backward from the first anchor and forward from the last.
    for backward in (False, True):
        frontier = anchors[0] if backward else anchors[-1]
        while True:
            if backward:
                candidates = [
                    tracklet
                    for tracklet in tracklets
                    if tracklet.tracklet_id not in used and tracklet.end_frame < frontier.start_frame
                ]
            else:
                candidates = [
                    tracklet
                    for tracklet in tracklets
                    if tracklet.tracklet_id not in used and tracklet.start_frame > frontier.end_frame
                ]
            best = _best_link(frontier, candidates, cache, diagonal, backward, decisions, tracklets)
            if best is None:
                break
            used.add(best.tracklet_id)
            chain.append(best)
            frontier = best

    chain.sort(key=lambda tracklet: tracklet.start_frame)
    return chain, decisions


def _sample(
    frame_number: int,
    fps: float,
    segment_start_time: float,
    bbox: np.ndarray | None,
    confidence: float,
    state: str,
    tracklet_id: int | None,
    search_center: np.ndarray | None = None,
) -> dict[str, Any]:
    timestamp = frame_number / fps
    return {
        "frame_number": frame_number,
        "timestamp_seconds": round(timestamp, 3),
        "clip_time_seconds": round(timestamp - segment_start_time, 3),
        "bbox": (
            {key: round(float(value), 2) for key, value in zip(("x1", "y1", "x2", "y2"), bbox)}
            if bbox is not None
            else None
        ),
        "confidence": round(confidence, 4),
        "state": state,
        "predicted": state not in ("tracked", "recovered"),  # backward compatibility
        "tracklet_id": tracklet_id,
        "search_center": (
            [round(float(search_center[0]), 1), round(float(search_center[1]), 1)]
            if search_center is not None
            else None
        ),
    }


@dataclass
class _Knot:
    """A frame where the player's box is actually known (observed or recovered)."""

    frame: int
    box: np.ndarray
    confidence: float
    state: str  # "tracked" | "recovered"
    tracklet_id: int | None


def _find_recovery_detection(
    cache: DetectionCache,
    frame_position: int,
    expected_center: np.ndarray,
    box_height: float,
    jersey_reference: np.ndarray | None,
    claimed: set[int],
) -> tuple[int, np.ndarray] | None:
    """A UNIQUE plausible detection near the expected position, or None.

    This is what puts the box back on a visible player the stitcher missed.
    Gates: position radius, jersey similarity, and a clear-uniqueness margin —
    when two candidates are close we return None rather than guess."""
    player_roles = {ROLE_INDEX["player"], ROLE_INDEX["goalkeeper"]}
    max_distance = RECOVERY_MAX_DISTANCE_HEIGHTS * max(box_height, 24.0)
    candidates: list[tuple[float, int]] = []
    for detection_index in cache.detections_at(frame_position):
        index = int(detection_index)
        if index in claimed or int(cache.det_role[index]) not in player_roles:
            continue
        candidate_box = cache.det_bbox[index]
        candidate_height = float(candidate_box[3] - candidate_box[1])
        # Size gate: a much smaller/larger box is a player at a different depth.
        if not (
            RECOVERY_SIZE_BAND[0] * box_height
            <= candidate_height
            <= RECOVERY_SIZE_BAND[1] * box_height
        ):
            continue
        distance = float(np.linalg.norm(_center(candidate_box) - expected_center))
        if distance > max_distance:
            continue
        # A box merged with / overlapping another player has no trustworthy
        # identity — attaching to it is how the tracker follows the wrong player
        # out of a crossing.
        if _detection_is_contaminated(cache, index, RECOVERY_MAX_OVERLAP_IOU):
            continue
        if jersey_reference is not None:
            jersey = cache.det_jersey[index].astype(np.float32)
            if not np.any(np.isnan(jersey)):
                similarity = float(np.clip(np.dot(jersey_reference, jersey), 0.0, 1.0))
                if similarity < RECOVERY_MIN_JERSEY_SIMILARITY:
                    continue
        candidates.append((distance, index))
    if not candidates:
        return None
    candidates.sort()
    if len(candidates) > 1 and candidates[0][0] > RECOVERY_UNIQUENESS_RATIO * candidates[1][0]:
        return None  # ambiguous: two believable candidates, refuse to guess
    index = candidates[0][1]
    return index, cache.det_bbox[index].copy()


def _chain_knots(chain: list[Tracklet]) -> tuple[list[_Knot], set[int]]:
    """Observed points from the stitched chain, smoothed within each tracklet."""
    knots: list[_Knot] = []
    claimed: set[int] = set()
    for tracklet in chain:
        smoothed: np.ndarray | None = None
        for position, (frame_number, box) in enumerate(zip(tracklet.frames, tracklet.boxes)):
            if smoothed is None or position == 0:
                smoothed = box.copy()
            else:
                alpha = TRACKING_BOX_SMOOTHING_ALPHA
                smoothed = (1 - alpha) * smoothed + alpha * box
            knots.append(
                _Knot(frame_number, smoothed.copy(), tracklet.confidences[position], "tracked", tracklet.tracklet_id)
            )
            claimed.add(tracklet.detection_indices[position])
    knots.sort(key=lambda knot: knot.frame)
    return knots, claimed


def _recover_gaps(
    knots: list[_Knot],
    cache: DetectionCache,
    jersey_reference: np.ndarray | None,
    claimed: set[int],
) -> list[_Knot]:
    """Inside every chain gap, re-attach unique plausible detections.

    Both gap endpoints are known, so the expected path is a reliable linear
    blend — much stronger than velocity extrapolation."""
    if not RECOVERY_ENABLED:
        return knots
    recovered: list[_Knot] = []
    for earlier, later in zip(knots, knots[1:]):
        gap = later.frame - earlier.frame
        if gap <= cache.stride:
            continue
        start_position = cache.frame_position(earlier.frame) + 1
        end_position = cache.frame_position(later.frame)
        for frame_position in range(start_position, end_position):
            frame_number = int(cache.frames[frame_position])
            # Skip recovery where the camera model is untrusted — attaching a
            # box using a wrong camera estimate is how drift happens.
            if float(cache.camera_confidence[frame_position]) < CAMERA_CONFIDENCE_MIN:
                continue
            fraction = (frame_number - earlier.frame) / gap
            # Both endpoints are mapped INTO the target frame before blending, so
            # the expected path stays correct under pan and zoom.
            earlier_mapped = cache.transform_point(_center(earlier.box), earlier.frame, frame_number)
            later_mapped = cache.transform_point(_center(later.box), later.frame, frame_number)
            expected = earlier_mapped * (1 - fraction) + later_mapped * fraction
            height = max(
                24.0,
                float(earlier.box[3] - earlier.box[1])
                * cache.scale_between(earlier.frame, frame_number),
            )
            found = _find_recovery_detection(
                cache, frame_position, expected, height, jersey_reference, claimed
            )
            if found is not None:
                index, box = found
                claimed.add(index)
                recovered.append(
                    _Knot(frame_number, box, float(cache.det_confidence[index]), "recovered", None)
                )
    if recovered:
        knots = sorted(knots + recovered, key=lambda knot: knot.frame)
    return knots


def _extend_ends(
    knots: list[_Knot],
    cache: DetectionCache,
    jersey_reference: np.ndarray | None,
    claimed: set[int],
) -> list[_Knot]:
    """Walk past both chain ends following unique detections, so the box does not
    vanish just because the stitcher ran out of linkable tracklets. Stops at the
    first ambiguity or after RECOVERY_MAX_CONSECUTIVE_MISSES empty samples."""
    if not RECOVERY_ENABLED or not knots:
        return knots
    extensions: list[_Knot] = []
    for forward in (True, False):
        edge = knots[-1] if forward else knots[0]
        # Camera-compensated velocity from the last few known points on this end.
        window = knots[-4:] if forward else knots[:4]
        if len(window) >= 2:
            elapsed = max(1, window[-1].frame - window[0].frame)
            start_mapped = cache.transform_point(
                _center(window[0].box), window[0].frame, window[-1].frame
            )
            velocity = (_center(window[-1].box) - start_mapped) / elapsed
        else:
            velocity = np.zeros(2, dtype=np.float32)
        if not forward:
            velocity = -velocity

        height = max(24.0, float(edge.box[3] - edge.box[1]))
        last_frame = edge.frame
        last_center = _center(edge.box)
        misses = 0
        position = cache.frame_position(edge.frame) + (1 if forward else -1)
        while 0 <= position < len(cache.frames) and misses <= RECOVERY_MAX_CONSECUTIVE_MISSES:
            frame_number = int(cache.frames[position])
            step = abs(frame_number - last_frame)
            # No blind extension while the camera model is untrusted.
            if float(cache.camera_confidence[position]) < CAMERA_CONFIDENCE_MIN:
                misses += 1
                position += 1 if forward else -1
                continue
            expected = cache.transform_point(
                last_center + velocity * step, last_frame, frame_number
            )
            found = _find_recovery_detection(
                cache, position, expected, height, jersey_reference, claimed
            )
            if found is not None:
                # Kinematic gate on the accepted point too.
                index, box = found
                candidate_height = max(24.0, float(box[3] - box[1]))
                if (
                    _implied_speed(
                        cache, last_frame, last_center, frame_number, _center(box), candidate_height
                    )
                    > _speed_cap(cache, last_frame, frame_number)
                ):
                    found = None
            if found is None:
                misses += 1
            else:
                index, box = found
                claimed.add(index)
                new_center = _center(box)
                last_mapped = cache.transform_point(last_center, last_frame, frame_number)
                velocity = (
                    0.6 * velocity + 0.4 * ((new_center - last_mapped) / max(1, step))
                ).astype(np.float32)
                extensions.append(
                    _Knot(frame_number, box, float(cache.det_confidence[index]), "recovered", None)
                )
                last_frame = frame_number
                last_center = new_center
                height = max(24.0, float(box[3] - box[1]))
                misses = 0
            position += 1 if forward else -1
    if extensions:
        knots = sorted(knots + extensions, key=lambda knot: knot.frame)
    return knots


def _emit_chain_samples(
    chain: list[Tracklet],
    cache: DetectionCache,
    fps: float,
    segment_start_time: float,
    jersey_reference: np.ndarray | None = None,
    all_tracklets: list[Tracklet] | None = None,
) -> list[dict[str, Any]]:
    """Known frames -> tracked/recovered; short gaps between known frames ->
    interpolated; longer gaps -> searching with no bbox. Boxes are never invented:
    every rendered box is either a real detection or a blend of two real ones."""
    width = cache.header.get("frame_width") or 0
    height = cache.header.get("frame_height") or 0

    knots, claimed = _chain_knots(chain)
    # Recovery may only claim ORPHAN detections. Anything owned by an established
    # tracklet the stitcher chose NOT to link is presumed to be another player;
    # stealing it is how the box jumps to the wrong person.
    if all_tracklets:
        chained_ids = {tracklet.tracklet_id for tracklet in chain}
        for tracklet in all_tracklets:
            if (
                tracklet.tracklet_id not in chained_ids
                and tracklet.length >= RECOVERY_EXCLUDE_TRACKLET_LENGTH
            ):
                claimed.update(tracklet.detection_indices)
    knots = _recover_gaps(knots, cache, jersey_reference, claimed)
    knots = _extend_ends(knots, cache, jersey_reference, claimed)

    samples: list[dict[str, Any]] = []
    for position, knot in enumerate(knots):
        if position > 0:
            earlier = knots[position - 1]
            gap = knot.frame - earlier.frame
            same_tracklet = (
                earlier.tracklet_id is not None and earlier.tracklet_id == knot.tracklet_id
            )
            max_interpolate = max(cache.stride * 2, INTERPOLATE_MAX_GAP_FRAMES) if not same_tracklet else cache.stride * 2
            # Plausibility: interpolation is only believable when the endpoints
            # are spatially close after removing camera motion. A straight-line
            # blend across a big jump renders as a box floating in empty space.
            knot_height = max(
                24.0,
                float(earlier.box[3] - earlier.box[1])
                * cache.scale_between(earlier.frame, knot.frame),
            )
            endpoint_shift = _center(knot.box) - cache.transform_point(
                _center(earlier.box), earlier.frame, knot.frame
            )
            interpolation_plausible = (
                float(np.linalg.norm(endpoint_shift))
                <= INTERPOLATE_MAX_DISTANCE_HEIGHTS * knot_height
            )
            if 1 < gap <= max_interpolate and interpolation_plausible:
                for step in range(1, gap):
                    fraction = step / gap
                    blended = earlier.box * (1 - fraction) + knot.box * fraction
                    samples.append(
                        _sample(
                            earlier.frame + step,
                            fps,
                            segment_start_time,
                            blended,
                            0.0,
                            "interpolated",
                            knot.tracklet_id,
                        )
                    )
            elif gap > 1:
                # Honest gap. Both endpoints are known, so the search hint follows
                # the linear path between them instead of blind extrapolation.
                for step in range(1, gap):
                    if step > SEARCH_TIMEOUT_FRAMES:
                        break
                    fraction = step / gap
                    center = _center(earlier.box) * (1 - fraction) + _center(knot.box) * fraction
                    if width and height:
                        center = np.clip(center, [0, 0], [width, height])
                    samples.append(
                        _sample(
                            earlier.frame + step,
                            fps,
                            segment_start_time,
                            None,
                            0.0,
                            "searching",
                            None,
                            search_center=center,
                        )
                    )
        samples.append(
            _sample(
                knot.frame,
                fps,
                segment_start_time,
                knot.box,
                knot.confidence,
                knot.state,
                knot.tracklet_id,
            )
        )
    return samples


def track_selected_player_tracklets(
    video_id: str,
    segment_id: str,
    video_path: Path,
    fps: float,
    segment_start_time: float,
    segment_end_time: float,
    selection: dict[str, Any],
    anchors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Tracklet-engine entry point: cache -> tracklets -> anchors -> stitch -> samples.

    Tracks the WHOLE segment (backward from the click too, for free, since
    stitching is symmetric in time). Extra anchors — added by the user where
    tracking was lost — pin identity at multiple points and stitching bridges
    between them."""
    processing_started = perf_counter()
    start_frame = round(segment_start_time * fps)
    end_frame = round(segment_end_time * fps)
    selections = anchors if anchors else [selection]

    cache, cache_hit = get_or_build_cache(
        video_id, segment_id, video_path, fps, start_frame, end_frame
    )
    tracklets = build_tracklets(cache)
    if not tracklets:
        raise HTTPException(status_code=422, detail="No player tracklets found in this segment")

    # Every user click is a confirmed identity point — resolve each to a tracklet.
    anchor_tracklets: list[Tracklet] = []
    for anchor_selection in selections:
        resolved = select_anchor(
            tracklets,
            cache,
            round(anchor_selection["selected_at_time"] * fps),
            anchor_selection["bbox"],
        )
        if resolved is not None and all(
            resolved.tracklet_id != existing.tracklet_id for existing in anchor_tracklets
        ):
            anchor_tracklets.append(resolved)
    if not anchor_tracklets:
        raise HTTPException(
            status_code=422,
            detail="Could not associate the selected player with any tracklet",
        )
    anchor = anchor_tracklets[0]

    chain, decisions = stitch_tracklets(tracklets, anchor_tracklets, cache)
    # Jersey reference for the recovery pass: prefer the anchor's own descriptor,
    # fall back to any chained tracklet that has one.
    jersey_reference = anchor.jersey_descriptor
    if jersey_reference is None:
        jersey_reference = next(
            (tracklet.jersey_descriptor for tracklet in chain if tracklet.jersey_descriptor is not None),
            None,
        )
    samples = _emit_chain_samples(
        chain, cache, fps, segment_start_time, jersey_reference, all_tracklets=tracklets
    )
    if not samples:
        raise HTTPException(status_code=422, detail="Tracking produced no samples")

    total_frames = max(1, end_frame - start_frame)
    tracked = sum(1 for sample in samples if sample["state"] == "tracked")
    recovered = sum(1 for sample in samples if sample["state"] == "recovered")
    interpolated = sum(1 for sample in samples if sample["state"] == "interpolated")
    searching = sum(1 for sample in samples if sample["state"] == "searching")
    gap_frames = [decision["gap"] for decision in decisions if decision.get("accepted")]

    return {
        "status": "completed",
        "tracker": "tracklet-stitch",
        "engine": "tracklet",
        "track_id": anchor.tracklet_id,
        "start_time": samples[0]["timestamp_seconds"],
        "end_time": samples[-1]["timestamp_seconds"],
        "frame_width": cache.header.get("frame_width"),
        "frame_height": cache.header.get("frame_height"),
        "samples": samples,
        "source_frames": total_frames,
        "inference_frames": 0 if cache_hit else len(cache.frames),
        "frame_stride": cache.stride,
        "processing_seconds": round(perf_counter() - processing_started, 3),
        "metrics": {
            "detection_cache_hit": cache_hit,
            "tracklet_count": len(tracklets),
            "stitched_tracklet_count": len(chain),
            "anchor_tracklet_id": anchor.tracklet_id,
            "anchor_tracklet_length": anchor.length,
            "anchor_count": len(anchor_tracklets),
            "coverage": round((tracked + recovered) * cache.stride / total_frames, 3),
            "recovered_samples": recovered,
            "interpolated_fraction": round(interpolated / max(1, len(samples)), 3),
            "searching_fraction": round(searching / max(1, len(samples)), 3),
            "gap_count": len(gap_frames),
            "longest_gap_frames": max(gap_frames, default=0),
            "stitch_decisions": decisions,
        },
    }
