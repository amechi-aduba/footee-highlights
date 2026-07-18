# Footee Highlights — Tracking & Analysis Implementation Plan

Grounded in the actual code as of 2026-07-10 (`app/services/player_tracking.py`, `object_detection.py`, `team_color.py`, `person_reidentification.py`, `core/config.py`, `routes/videos.py`).

---

## 0. Architecture issues found in the current code

These are concrete problems, not hypotheticals:

1. **Greedy online tracking is the root cause of ID switches.** `track_selected_player` makes an irreversible decision every frame. Once it latches onto the wrong ByteTrack ID during a crossing, every later frame reinforces the error (reference descriptor EMA at line ~517 slowly *adapts to the wrong player*). No amount of threshold tuning fixes a greedy architecture — move to tracklets + offline stitching (Section 3).
2. **Tracking ends permanently on loss.** `if missing_frames > TRACKING_MAX_MISSING_FRAMES: break` (line ~499) exits the loop. There is no "searching" state; the player can never be reacquired after ~20 missed inference frames even if they reappear clearly.
3. **Model reloaded per request.** `YOLO(TRACKING_MODEL_PATH)` is constructed inside `track_selected_player` on every call. `object_detection.get_detection_model` is `lru_cache`d; tracking is not. On CPU this costs seconds per request.
4. **Detections are never cached.** Re-tracking the same clip (different player, retry, tuning) reruns full YOLO inference. Detection is the expensive part; association is nearly free.
5. **Synchronous processing in FastAPI request handlers.** `/track-focused-player` and `/process` block the HTTP request for the whole job. No job queue, no progress reporting, request timeouts loom on long clips.
6. **ReID embedding mismatch.** ImageNet ResNet18 on a 8×20 px crop upscaled to 224×224 produces noise. The neural branch likely *hurts* on distant players vs. the HSV/texture fallback.
7. **Velocity in pixel space with a moving handheld camera.** `velocity` mixes player motion and camera motion. During a fast pan, predicted boxes fly off the real player, and `_position_similarity` / `_within_reassociation_radius` compare against garbage predictions.
8. **`_within_reassociation_radius` measures from `last_observed_bbox` in old-frame pixel coordinates** — after a pan, the correct player may be outside the radius while a wrong player is inside it.
9. **The yolo11n path exists but is unused.** Config prefers `models/football-yolo11n.pt`, `scripts/promote_tracking_detector.py` exists, but only `football-yolo11m.pt` is in `models/`. Tracking runs 11m today.
10. **No pitch/crowd gating.** Only class filtering (`player`, `goalkeeper`). Spectators detected as players enter the candidate pool.
11. **Random seeks in `detect_objects_in_window`** (`capture.set(POS_FRAMES)` every 30 frames) force keyframe decode + walk on most codecs; sequential read with frame skipping is much faster.

---

## 1. Prioritized roadmap

Ordered by (impact × low risk) ÷ effort. Each phase is shippable alone.

| Phase | What | Fixes | Effort |
|---|---|---|---|
| **P0** | Detection cache + cached tracking model + fine-tune/promote yolo11n | Speed (biggest win), enables everything below | 1–2 days |
| **P1** | Camera-motion compensation (sparse optical flow global model) | ID switches, bad predictions, reassociation radius | 1–2 days |
| **P2** | Tracklet builder + offline stitcher + forward/backward tracking | ID switches, hallucination, reacquisition | 3–5 days |
| **P3** | Lost/searching state machine + pitch/crowd mask | Hallucination, crowd pollution | 1–2 days |
| **P4** | Background jobs + progress polling | UX, timeouts | 1–2 days |
| **P5** | Metrics/eval harness + debug overlay renderer | Knowing if any of this worked | 1–2 days (build parts of it during P0–P3, not after) |
| **P6** | Analysis layer v1: participation timeline (ball proximity, involvement, movement) | Product value | 3–5 days |

P0 and P5 first — you can't tune what you can't measure, and everything downstream reruns association against cached detections in seconds instead of minutes.

---

## 2. P0 — Detection caching, model management, speed

### New module: `app/services/detection_cache.py`

Run detection **once per clip**, store results, and make tracking a pure association pass over cached detections.

```
storage/detections/{video_id}/{segment_id}.npz   (or .json.gz)
```

```python
# Per-frame record (list per sampled frame)
FrameDetections = {
    "frame_number": int,
    "camera_shift": [dx, dy],          # filled by P1, cumulative from segment start
    "detections": [
        {
            "bbox": [x1, y1, x2, y2],
            "confidence": float,
            "class": "player" | "goalkeeper" | "referee" | "ball",
            "jersey_descriptor": [...] | None,   # computed lazily, cached back
            "appearance": [...] | None,          # computed lazily, cached back
            "on_pitch": bool,                    # filled by P3
        }
    ],
}
CacheHeader = {
    "model": "football-yolo11n.pt", "model_hash": str, "imgsz": int,
    "conf": float, "stride": int, "fps": float, "frame_width": int, "frame_height": int,
}
```

Key functions:

- `get_or_build_detections(video_path, segment, stride, model_key) -> DetectionCache` — sequential decode (`capture.read()` + skip, **no** `capture.set` seeks), batched YOLO inference (`model.predict(list_of_frames)`, batch 8–16 helps even on CPU).
- Invalidate on `model_hash` / `imgsz` / `conf` mismatch.
- Ball detections cached too — the P6 analysis layer needs them and they're free at this point.

### Model changes

- Add `@lru_cache(maxsize=2)` `get_tracking_model()` next to `get_detection_model()` in a shared `model_registry.py`; kill the per-request `YOLO(...)` in `track_selected_player`.
- Fine-tune **yolo11n** on your existing football dataset (reuse `scripts/train_football_detector.py` with `yolo11n.pt` base) and promote to `models/football-yolo11n.pt` via the existing `promote_tracking_detector.py`. Expect ~3–4× per-frame speedup vs 11m at similar imgsz.
- **Two-tier strategy (recommended over a two-stage detector):** 11n at `imgsz=640` for the cache/tracking pass; 11m only for (a) the single frame the user clicks on (`detect-frame` endpoint — already separate) and (b) an optional "verification" pass on hard frames (crossings, reacquisition candidates) — re-detect just those frames at `imgsz=960` with 11m. A true two-stage detector (region proposal + classifier) is not worth it here; the two-tier YOLO approach gets the same effect with code you already have.
- **Adaptive stride, not aggressive stride:** with cached detections, keep stride at 2 for the cache pass. Never let stride exceed ~4 (at 30 fps, a sprinting player moves ~25 px/frame at typical zoom; gaps beyond 4 frames break IoU-based association).
- GPU workers: not now. Design the cache + job layers so a worker could run elsewhere (the cache file is the interface), but CPU + 11n + caching should get a 15 s clip to well under a minute. Revisit only if that fails.

---

## 3. P1 — Camera-motion compensation

### New module: `app/services/camera_motion.py`

```python
def estimate_global_shift(prev_gray, cur_gray) -> tuple[np.ndarray, float]:
    """Returns (2-vector dx,dy, inlier_ratio). Affine optional later."""
```

Implementation: `cv2.goodFeaturesToTrack` (maxCorners=200, qualityLevel=0.01, minDistance=16) on a downscaled gray frame (width 480) → `cv2.calcOpticalFlowPyrLK` → mask out points inside player detections (they're the outliers you don't want) → `cv2.estimateAffinePartial2D(..., method=cv2.RANSAC, ransacReprojThreshold=3)` → take translation (+ scale if reliable). Cost: ~2–4 ms/frame at 480 px — negligible next to YOLO.

Store cumulative `camera_shift` per frame in the detection cache. Then, everywhere player motion is compared:

- `velocity` becomes **camera-compensated**: `measured = (cur_center - prev_center) - camera_delta`.
- `_shift_bbox` predictions add camera delta back: `predicted = last_bbox + player_velocity * dt + camera_shift_delta`.
- `_within_reassociation_radius` compares in **camera-compensated coordinates** — this alone fixes most "reacquired the wrong player after a pan" failures.

Fallback: if `inlier_ratio < 0.4` (motion blur, whole-frame occlusion), set camera delta to 0 and widen the association gate for that frame.

---

## 4. P2 — Tracklets + offline stitching (the core fix for ID switches)

Replace greedy per-frame decisions with: **build conservative tracklets first, then stitch.**

### New module: `app/services/tracklets.py`

**Stage A — conservative tracklet building** (over cached detections):

Run ByteTrack-style association with *strict* gates so tracklets are short but pure: IoU ≥ 0.3 (camera-compensated), no appearance needed. **Critically: when two candidates are ambiguous (both within gate, cost ratio < 1.3), terminate the tracklet instead of guessing.** Ambiguity = crossing = exactly where greedy tracking fails. A tracklet should end at every crossing; the stitcher resolves it with more context than any single frame has.

```python
@dataclass
class Tracklet:
    tracklet_id: int
    start_frame: int
    end_frame: int
    boxes: dict[int, BBox]              # frame -> bbox (observed only, no predictions)
    confidences: dict[int, float]
    mean_appearance: np.ndarray | None  # avg over the K largest crops only (see below)
    jersey_descriptor: np.ndarray | None
    team_id: str | None
    velocity_end: np.ndarray            # camera-compensated, for gap prediction
    velocity_start: np.ndarray          # for backward stitching
    quality: float                      # length * mean_conf * purity heuristics
```

Appearance per tracklet, not per detection: average descriptors only from crops ≥ 24 px wide (skip the 8×20 ones — they add noise). If no crop qualifies, `mean_appearance = None` and stitching for that tracklet relies on motion + team color + size only. This is the honest way to handle tiny players.

**Stage B — stitching** (`stitch_tracklets(tracklets, anchor, camera_shifts, config)`):

The user's click defines the **anchor tracklet** (the one containing/overlapping the selected bbox at the selected frame). Then greedily (or via shortest-path over a DAG if you want to be thorough) attach tracklets forward and backward in time:

Link cost between tracklet A (ends at frame f₁) and B (starts at f₂ > f₁), gap g = f₂ − f₁:

```
cost = w_pos * position_cost      # dist(A.end + A.velocity_end*g [camera-comp], B.start) / gate(g)
     + w_app * appearance_cost    # 1 - cosine sim; skipped (renormalized) if either side is None
     + w_team * team_cost         # 0 if same kmeans team, 1 if different  → SEMI-HARD: reject link if different team AND appearance_cost > 0.5
     + w_size * size_cost
Reject link if: g > MAX_GAP_FRAMES (90), or position outside camera-compensated radius,
or cost > LINK_THRESHOLD, or (best_cost / second_best_cost) > 0.75 (no clear winner → leave gap).
```

Team color as **semi-hard constraint**: hard-reject only when both team assignment differs *and* appearance disagrees. Pure hard constraint fails on white-vs-light-gray amateur kits and lighting shifts; pure soft weight lets a defender in a crossing win. Semi-hard is the right middle.

**Forward + backward:** stitching is symmetric in time, so backward tracking falls out for free — the anchor tracklet stitches toward frame 0 as well as toward the end. The user gets the full trajectory, not just post-click, and the pre-click portion often disambiguates crossings that happen right after the click.

**Output:** ordered list of stitched tracklets + explicit gaps. Gaps are rendered as gaps (or dashed "searching" indicator in the UI) — never as invented boxes. Interpolate *only* gaps ≤ 8 frames where both bounding tracklets agree on velocity (linear interpolation in camera-compensated space, marked `"interpolated": true`).

`track_selected_player` becomes a thin orchestrator: load cache → build tracklets → stitch → emit samples. Keep the old function behind a config flag (`TRACKING_ENGINE=greedy|tracklet`) for A/B comparison during rollout.

---

## 5. P3 — Lost state + crowd filtering

### Lost/searching state (kills hallucination)

With tracklet stitching, "lost" is just a gap between stitched tracklets — but encode it explicitly in the output so the UI can show it:

```python
TrackSample = {
    "frame_number": int, "timestamp_seconds": float, "clip_time_seconds": float,
    "state": "tracked" | "interpolated" | "searching" | "ended",
    "bbox": {...} | None,        # None when searching
    "confidence": float,
    "tracklet_id": int | None,
    "search_center": [x, y] | None,   # camera-comp predicted location, for a subtle UI hint
}
```

Rules: never emit a bbox in `searching` state; `TRACKING_MAX_PREDICTION_FRAMES` applies only to `interpolated`. After `SEARCH_TIMEOUT_FRAMES` (~120 ≈ 4 s) with no stitchable tracklet, emit `ended` — but since stitching is offline over the whole clip, a strong match 6 s later can still legitimately reattach if it passes the strict long-gap gates (appearance ≥ 0.60 as you already have, same team, plausible position).

### New module: `app/services/pitch_mask.py`

Two cheap, broadcast-free filters (compute once per segment on ~10 sampled frames, since handheld cameras move):

1. **Grass mask:** HSV threshold (H 30–90, S ≥ 40, V ≥ 40) → morphological close → largest connected component → convex hull ≈ pitch region. Works on amateur grass/turf; degrades gracefully (if grass covers < 25 % of frame, disable the filter rather than misfire).
2. **Feet test:** a detection is `on_pitch` if the bottom-center of its bbox falls inside the (dilated, +5 % margin) hull. Spectators behind the far touchline fail this; players near the line pass because their *feet* are on grass even when their torso overlaps the crowd.

Optionally add a **horizon prior**: reject detections whose bbox bottom is above the highest grass row minus a margin. Filter applies to the tracklet-building candidate pool *and* `detect_objects_at_timestamp` (so the click UI stops offering fans). Store `on_pitch` in the detection cache; keep filtered detections in the cache flagged rather than dropped, for debugging.

---

## 6. P4 — Background jobs

Minimal, no new infra: FastAPI `BackgroundTasks` is not enough (no progress); use a tiny in-process job registry.

### New module: `app/services/jobs.py`

```python
Job = {
    "job_id": str, "kind": "process" | "detect_cache" | "track",
    "status": "queued" | "running" | "completed" | "failed",
    "progress": float,          # 0..1, updated by workers (frames_done / frames_total)
    "result_ref": str | None,   # where the result landed (existing results JSON)
    "error": str | None,
}
```

`ThreadPoolExecutor(max_workers=1)` (YOLO on CPU won't benefit from parallel jobs; a queue prevents two clips thrashing). Routes change to: `POST /track-focused-player` → returns `{job_id}` immediately; `GET /jobs/{job_id}` → status + progress; frontend polls every 1–2 s and renders a progress bar. Persist job state into the existing results JSON on completion so the current `load_analysis_result` flow is unchanged. This keeps the door open to swap the executor for Celery/Azure later without touching routes.

---

## 7. P5 — Metrics, logging, debug plan

Build this **during** P0–P3, not after.

### Per-run metrics (append to `storage/results/{video_id}.json` under `focused_player_track.metrics` and to a rotating `storage/logs/tracking_metrics.jsonl`):

- `processing_seconds`, `fps_effective` (source_frames / processing_seconds)
- `detection_cache_hit` (bool), `inference_frames`, `detections_per_frame_mean`
- `tracklet_count`, `stitched_tracklet_count`, `anchor_tracklet_length`
- `coverage`: fraction of clip frames in `tracked` state (up from "always draws a box, sometimes wrong" — expect this to *drop* initially and that's good)
- `interpolated_fraction`, `searching_fraction`
- `longest_gap_frames`, `gap_count`
- `stitch_decisions`: list of `{gap, cost, appearance, team_match, accepted}` — the single most useful debugging artifact for tuning thresholds
- `mean_stitch_cost_margin` (best vs second-best) — low margins = ambiguous clips
- `crowd_filtered_count`, `camera_shift_magnitude_p95`

### Ground truth + eval harness: `scripts/eval_tracking.py`

Label 5–10 representative clips (include: a crossing, a long occlusion, tiny distant players, heavy pan, visible crowd). Ground truth = selected player bbox every 10th frame (CVAT — you already have the import script). Report per clip:

- **IDF1-style score**: fraction of GT frames where emitted bbox IoU ≥ 0.3 with GT
- **ID switches**: emitted box jumps to a different GT identity
- **Hallucination rate**: frames with emitted bbox but IoU = 0 with GT (should → ~0 after P3)
- **Reacquisition success**: after each GT-annotated occlusion, does the track resume on the correct player?

Run the harness on every threshold change. This turns "feels better" into numbers.

### Debug overlay renderer: `scripts/render_debug_overlay.py`

Renders an MP4 per tracked clip (debug tool only, not product): all cached detections (gray), crowd-filtered (red X), tracklets in distinct colors with IDs, anchor/stitched track (thick green), interpolated (dashed yellow), searching state (pulsing circle at `search_center`), camera-shift arrow, and per-frame stitch-cost text at tracklet boundaries. One glance at this video tells you *which* stage failed.

---

## 8. P6 — Analysis layer v1

Build **participation/involvement first** — it needs only the stitched track + cached ball detections, tolerates handheld footage, and is immediately meaningful to a high-school player. Passes/carries need reliable ball possession inference, which needs pitch calibration; defer.

### New module: `app/services/player_analysis.py`

```python
AnalysisResult = {
    "player_summary": {
        "minutes_tracked": float, "coverage": float,
        "ball_near_events": int,          # ball within 1.5 player-heights for >= 5 frames
        "involvement_score": float,       # time-weighted ball proximity, 0..100
        "distance_covered_relative": float,  # camera-comp px normalized by player height (no calibration needed)
        "sprint_count": int,              # camera-comp speed > 2.5 player-heights/s sustained >= 0.5 s
        "activity_timeline": [ {"t": float, "activity": "idle"|"moving"|"sprinting", "ball_near": bool} ],
    },
    "highlight_moments": [
        {"start": float, "end": float, "reason": "ball_proximity_peak" | "sprint" | "sustained_involvement", "score": float}
    ],
}
```

Everything here uses player-height as the unit of distance — that's the trick that avoids pitch calibration on zoomed-out handheld footage. `highlight_moments` feeds the existing clip UI: "your most involved 20 seconds." Later phases (needs pitch homography, a separate project): touches/passes/carries, defensive actions, attacking/defending phase classification via team centroid movement.

---

## 9. Config starting points

Add to `config.py` (env-overridable like existing ones):

```python
# P0
TRACKING_ENGINE = "tracklet"                # "greedy" keeps old path for A/B
DETECTION_CACHE_ENABLED = True
DETECTION_CACHE_STRIDE = 2
TRACKING_BATCH_SIZE = 8

# P1 camera motion
CAMERA_MOTION_ENABLED = True
CAMERA_MOTION_DOWNSCALE_WIDTH = 480
CAMERA_MOTION_MIN_INLIER_RATIO = 0.4

# P2 tracklets
TRACKLET_MIN_IOU = 0.30                     # strict on purpose
TRACKLET_AMBIGUITY_RATIO = 1.3              # two candidates closer than this -> end tracklet
TRACKLET_MIN_LENGTH = 3                     # drop 1-2 frame noise tracklets
STITCH_MAX_GAP_FRAMES = 90
STITCH_LINK_THRESHOLD = 0.45                # cost, lower=stricter
STITCH_WINNER_MARGIN = 0.75                 # best/second-best cost ratio
STITCH_WEIGHTS = {"pos": 0.35, "app": 0.30, "team": 0.20, "size": 0.15}
STITCH_LONG_GAP_FRAMES = 30
STITCH_LONG_GAP_MIN_APPEARANCE = 0.60       # matches your existing TRACKING_LONG_GAP_APPEARANCE
APPEARANCE_MIN_CROP_WIDTH = 24              # below this, don't trust/compute appearance
INTERPOLATE_MAX_GAP_FRAMES = 8

# P3
PITCH_MASK_ENABLED = True
PITCH_MASK_MIN_GRASS_FRACTION = 0.25        # below -> disable filter, don't misfire
PITCH_MASK_MARGIN_FRACTION = 0.05
SEARCH_TIMEOUT_FRAMES = 120
```

Tune `STITCH_LINK_THRESHOLD` and `STITCH_WINNER_MARGIN` first, against the P5 harness; leave the rest until you have eval numbers.

---

## 10. Answers to the specific questions, mapped

1. **ID switches at crossings** → P2: end tracklets at ambiguity, resolve at stitch time with team color (semi-hard), appearance (only from big-enough crops), camera-compensated motion. P1 makes the motion term trustworthy.
2. **Hallucinated tracking** → P3 explicit `searching` state with `bbox: None`; interpolation only for short, velocity-consistent gaps; never render predicted boxes as real.
3. **Reacquisition** → offline stitching searches the whole clip, both directions, with gap-scaled gates and the long-gap appearance floor (0.60). Reacquisition stops being a fragile real-time decision.
4. **Crowd filtering** → P3 grass-hull + feet test + `on_pitch` flag in the cache; auto-disables when grass fraction is low.
5. **Speed without quality loss** → P0 detection cache (biggest win: association reruns are free), sequential decode, batched inference, cached model, stride ≤ 2 preserved because detection runs once.
6. **Model strategy** → fine-tuned yolo11n for the cache pass, 11m for click-frame + optional hard-frame verification; no two-stage detector; async jobs yes (P4, in-process); GPU workers deferred, cache file is the future interface.
7. **Camera motion** → P1 sparse LK flow + RANSAC affine on non-player points, ~3 ms/frame, feeding velocity, prediction, and reassociation gating.
8. **Tracklets** → Section 4 in full.
9. **First analysis layer** → involvement/participation timeline in player-height units (P6); passes/touches deferred until pitch calibration exists.
10. **Metrics** → Section 7: coverage, ID switches, hallucination rate, reacquisition success, stitch-decision log, plus the eval harness on 5–10 labeled clips.

## Suggested file layout after all phases

```
app/services/
  model_registry.py        # cached YOLO loaders (11n + 11m)
  detection_cache.py       # P0
  camera_motion.py         # P1
  tracklets.py             # P2 (build + stitch)
  pitch_mask.py            # P3
  jobs.py                  # P4
  player_tracking.py       # thin orchestrator; greedy path kept behind flag
  player_analysis.py       # P6
scripts/
  eval_tracking.py         # P5
  render_debug_overlay.py  # P5
```
