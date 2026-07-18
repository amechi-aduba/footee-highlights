# Footee Vision — Project Overview

**What it is:** a full-stack computer-vision app that turns an amateur soccer highlight reel into a tactical player profile. A player uploads their reel, the app splits it into clips, they click themselves once per clip, the tracker follows them, and the analysis layer classifies which archetype they play like — Ball-Playing Defender, Box-to-Box Midfielder, Inside Forward, and nine others.

**Who it's for:** amateur and high-school players with handheld, shaky, zoomed-out footage. Every design decision assumes NON-broadcast video: no fixed camera, no pitch calibration, tiny players, crowds in frame.

---

## 1. Architecture

```
frontend/  React 19 + Vite + Tailwind v3 (class-based dark mode, semantic tokens)
backend/   FastAPI + OpenCV + PyTorch
  app/routes/videos.py          HTTP API
  app/services/
    transnetv2_detection.py     shot-boundary model + real frame timestamps
    scene_detection.py          cut merging, second-pass scan, segment building
    segment_classifier.py       gameplay vs cutaway (intros, celebrations)
    model_registry.py           cached YOLO loaders (detect + track models)
    detection_cache.py          one detection pass per clip, stored as .npz
    camera_motion.py            global shift estimation (LK flow + RANSAC)
    pitch_mask.py               grass hull + feet test (crowd filtering)
    appearance.py               cheap HSV/texture identity descriptor
    team_color.py               jersey descriptors + team kmeans
    tracklets.py                tracklet build → multi-anchor stitch → recovery
    player_tracking.py          engine dispatcher (tracklet | legacy greedy)
    player_analysis.py          features → traits → archetype classification
  scripts/render_debug_overlay.py   diagnostic MP4 renderer
  storage/                      raw videos, thumbnails, results JSON, detection caches
  models/football-yolo11m.pt    custom 4-class detector (player/goalkeeper/referee/ball)
```

Models: a fine-tuned **YOLO11m** handles the click-frame detection (quality); the tracking/cache pass uses `models/football-yolo11n.pt` when promoted (speed), falling back to 11m. **TransNetV2** finds shot boundaries. Identity matching uses HSV/texture descriptors + jersey color — deliberately not ImageNet embeddings, which are noise on tiny crops.

---

## 2. User workflow

1. **Upload** the highlight reel (drag-and-drop). Stored locally under `storage/raw_videos/`.
2. **Process** — the reel is split into clips, cutaways are filtered, thumbnails generated.
3. **Player setup** — pick main positions (ST/LW/RW/10/CM/CDM/CB/RB/LB) and footedness. This narrows the archetype candidates.
4. **Per clip:** open it from the grid (only one video player exists at a time — this is what keeps playback smooth). Players are auto-detected on the first frame; click yourself; hit *Track*.
5. **Verify** — the overlay plays back over the clip: solid green = observed, cyan = recovered, dashed yellow = interpolated, pulsing amber = searching (honest "lost" state; the app never invents a box).
6. **If tracking is lost**, playback pauses at that exact moment with players pre-detected — click yourself again to add an **anchor** (identity pinned at multiple points), re-track (instant, detections are cached). *Reset selection* undoes a wrong pick entirely.
7. **Generate profile** — once ≥1 clip is tracked. More clips = higher confidence.

---

## 3. Pipeline internals (what happens under the hood)

### 3.1 Clip splitting
- TransNetV2 scores every frame for shot transitions; cuts land at the **end** of each transition run so dissolve residue stays with the previous clip.
- A **second-pass HSV/edge scan** catches hard jump-cuts between visually similar plays that TransNetV2 misses; both cut sets merge (duplicates collapse via minimum-length spacing).
- Segments carry **real decoder timestamps**, immune to variable-frame-rate drift, plus a 0.3 s start trim (`SEGMENT_START_TRIM_SECONDS`).
- Each segment is classified **gameplay vs cutaway** (grass fraction + player count); cutaways are collapsed in the UI, never deleted.

### 3.2 Detection cache (runs once per clip)
Sequential decode, batched YOLO, stride 2. Per sampled frame it stores: all detections (boxes, class, confidence), jersey + appearance descriptors, an **on-pitch flag** (grass-hull feet test filters spectators), and the cumulative **camera shift** (sparse optical flow on background points, RANSAC). Cached as `.npz`, invalidated on model change. Every re-track and the analysis layer reads from this cache — association reruns cost seconds.

### 3.3 Tracking (tracklet engine)
- **Conservative tracklets:** strict IoU association on camera-compensated predictions. At any ambiguity (a crossing), tracklets *terminate* rather than guess.
- **Stitching:** user clicks define anchor tracklets (ground truth). Stitching bridges between anchors and extends outward, weighing position, appearance, team color (semi-hard constraint), and size. Ambiguous links become honest gaps. Crossings get **joint 2×2 resolution** — the pairing of both crossing players is solved together. Appearance descriptors exclude **contaminated crops** (boxes overlapping another player).
- **Recovery:** gaps and chain ends are re-scanned for *orphan* detections (never ones owned by another player's tracklet), gated by uniqueness, jersey similarity, and size.
- **Ball filtering:** turf dots/line marks firing the ball class are rejected because they're static in camera-compensated space; per-frame selection uses confidence + trajectory continuity.

### 3.4 Analysis
Measured per tracked clip, in **player-height units** (no pitch calibration needed):
involvement (ball-near events/min), work rate (distance/min), explosiveness (sprints/min), wideness (x-position percentile among detected teammates).
Declared positions select candidate archetypes; footedness splits the winger types (opposite foot → Inside Forward); transparent scoring formulas rank them (`player_analysis.py`, all weights at the top of the file). The profile reports **evidence and confidence** — and states plainly what it can't measure yet (passes, duels, shots).

**Archetypes:** Ball-Playing Defender, Stopper, Attacking/Inverted Wing-Back, Holding Six, Box-to-Box, Deep-Lying Playmaker, Attacking Playmaker, Target Man, Poacher, Inside Forward, Classic Winger, False Nine.

---

## 4. API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/videos/upload` | store the reel |
| POST | `/api/videos/{id}/process` | split, classify, thumbnail |
| GET | `/api/videos/{id}/result` | full analysis JSON |
| GET | `/api/videos/{id}/video` · `/thumbnail/{seg}` | media |
| POST | `/{id}/segment/{seg}/detect-frame` | detect players on one frame |
| POST | `/{id}/segment/{seg}/focused-player` | select player (`additive: true` = add anchor) |
| DELETE | `/{id}/segment/{seg}/focused-player` | reset selection/anchors/track |
| POST | `/{id}/segment/{seg}/track-focused-player` | run the tracker |
| POST | `/api/videos/{id}/player-info` | positions + footedness |
| POST | `/api/videos/{id}/player-profile` | generate archetype profile |

## 5. Key configuration (env-overridable, `app/core/config.py`)

`TRACKING_ENGINE` (tracklet/greedy) · `DETECTION_CACHE_STRIDE` (2) · `STITCH_LINK_THRESHOLD` (0.55) · `STITCH_MAX_GAP_FRAMES` (90) · `INTERPOLATE_MAX_GAP_FRAMES` (20) · `RECOVERY_*` gates · `PITCH_MASK_*` · `BALL_MIN_CONFIDENCE` (0.30) / `BALL_STATIC_FRACTION` (0.35) · `SEGMENT_START_TRIM_SECONDS` (0.30) · `SCENE_SECOND_PASS_ENABLED` · classifier thresholds (`SEGMENT_GRASS_FRACTION_MIN`, `SEGMENT_GAMEPLAY_MIN_PLAYERS`).

Debugging: every track response includes `metrics` (tracklet counts, coverage, `stitch_decisions` with per-link costs and reasons). `python scripts/render_debug_overlay.py <video_id> <segment_id>` renders an MP4 showing every pipeline layer.

---

## 6. Known limitations (honest list)

- **Identity through occlusion is physics-limited** on handheld footage: two same-kit players fully occluding while both change direction cannot be resolved by any motion/color evidence. Anchors are the designed backstop.
- **Appearance matching is color-based**, not a trained ReID network — the biggest remaining tracking upgrade.
- **No event detection yet** (touches, passes, shots) — profiles use movement/involvement only.
- **Processing is synchronous** — the HTTP request blocks during processing and first-track; fine locally, not for a hosted deployment.
- **No pitch calibration** — all distances are relative (player-heights), which is the right honest unit for this footage, but rules out absolute-distance stats.
- Results persist as JSON files; no accounts/database.

---

## 7. Roadmap

### Tier 1 — required for going live
1. **Background jobs + progress polling** — move processing/tracking into a worker with `/jobs/{id}` progress; the UI shimmer becomes a real progress bar. Without this, hosted requests time out.
2. **Deployment packaging** — Dockerize backend (CPU inference), static-host the frontend, object storage for videos, a queue for jobs. Model files fetched at startup.
3. **Accounts + database** — replace results JSON with a real store keyed by user.

### Tier 2 — biggest product wins
4. **Auto-propagate the player across clips** — after one clip is tracked, use the confirmed jersey/appearance to *suggest* the player in every other clip; the user confirms instead of hunting. Turns 10 clicks into ~2. (The descriptors already exist in the caches.)
5. **Clip export with overlay burned in** — download/share a clip with the tracking spotlight rendered on it. Highlight-reel users will share these; it's the growth loop.
6. **Player comparison** — the profile already stores a trait vector + archetype scores; comparing two players (or "you vs the archetype ideal") is a similarity view away. Later: comparisons to well-known player profiles.
7. **Highlight moments** — surface each player's most-involved stretches (the analysis already computes per-segment involvement) as a "best moments" strip.

### Tier 3 — accuracy compounding
8. **OSNet-x0.25 ReID** — drop-in replacement for the HSV descriptor in `appearance.py` + cache; the single biggest tracking-quality jump available. CPU-friendly.
9. **Promote a fine-tuned yolo11n tracking model** (`scripts/train_football_detector.py` → `promote_tracking_detector.py`) — 3–4× faster cache builds; caches auto-invalidate.
10. **Eval harness** — label 5–10 hard clips (CVAT scripts exist) and score ID-switches/coverage per change, so tuning is measured rather than felt.
11. **Event detection** — ball-possession inference from ball-player proximity + motion, unlocking touches/carries and sharper archetypes.
12. **Pitch homography (stretch)** — line-based calibration where footage allows; unlocks absolute distances, heat maps, and true positional analysis.

### Tier 4 — polish
Multi-video aggregated profiles · profile card image export · keyboard shortcuts (space = play, arrows = scrub) · mobile layout pass · onboarding tour · goalkeeper support.
