"""Central configuration. Every threshold is env-overridable and carries a
comment explaining WHY it exists — tuning should never require reading the
algorithm first.

NOTE: this file was once clobbered by a cloud-sync conflict, which silently
removed most settings and broke server startup with ImportErrors. If settings
ever go missing again, rebuild from the union of `from app.core.config import`
statements across app/ and scripts/.
"""
from pathlib import Path
import os
import tempfile


BACKEND_DIR = Path(__file__).resolve().parents[2]

_configured_storage_dir = os.getenv("FOOTEE_STORAGE_DIR")
if _configured_storage_dir:
    _storage_path = Path(_configured_storage_dir)
    STORAGE_DIR = (
        _storage_path if _storage_path.is_absolute() else BACKEND_DIR / _storage_path
    ).resolve()
elif os.getenv("VERCEL"):
    # Vercel's deployed filesystem is read-only except for the ephemeral temp
    # directory. A container host can set FOOTEE_STORAGE_DIR explicitly.
    STORAGE_DIR = Path(tempfile.gettempdir()) / "footee-vision"
else:
    STORAGE_DIR = BACKEND_DIR / "storage"

RAW_VIDEOS_DIR = STORAGE_DIR / "raw_videos"
THUMBNAILS_DIR = STORAGE_DIR / "thumbnails"
RESULTS_DIR = STORAGE_DIR / "results"
ULTRALYTICS_CONFIG_DIR = STORAGE_DIR / "ultralytics"
MODELS_DIR = BACKEND_DIR / "models"
DETECTIONS_DIR = STORAGE_DIR / "detections"
SEGMENTS_DIR = STORAGE_DIR / "segments"

# Uploaded videos and every artifact derived from them are temporary. The
# browser asks for deletion when a session ends; this server-side expiry is the
# fallback when a tab crashes, loses connectivity, or is force-closed.
UPLOAD_RETENTION_SECONDS = max(300, int(os.getenv("UPLOAD_RETENTION_SECONDS", "3600")))
UPLOAD_CLEANUP_INTERVAL_SECONDS = max(
    60, int(os.getenv("UPLOAD_CLEANUP_INTERVAL_SECONDS", "900"))
)
MAX_UPLOAD_BYTES = max(1, int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))) * 1024 * 1024

_default_cors_origins = "http://localhost:5173,http://127.0.0.1:5173"
CORS_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv("CORS_ORIGINS", _default_cors_origins).split(",")
    if origin.strip()
]

# Render's 512 MB instances cannot safely use the local-quality defaults.
# Depending on the service/runtime generation, Render may expose either its
# boolean flag or only service-specific variables, so recognize both forms.
_running_on_render = (
    os.getenv("RENDER", "false").lower() == "true"
    or any(
        os.getenv(variable_name)
        for variable_name in (
            "RENDER_SERVICE_ID",
            "RENDER_EXTERNAL_URL",
            "RENDER_EXTERNAL_HOSTNAME",
        )
    )
)
LOW_MEMORY_MODE = os.getenv(
    "LOW_MEMORY_MODE", "true" if _running_on_render else "false"
).lower() == "true"

# ---------------------------------------------------------------- scene cuts
SCENE_DETECTION_METHOD = os.getenv("SCENE_DETECTION_METHOD", "transnetv2")
TRANSNETV2_THRESHOLD = float(os.getenv("TRANSNETV2_THRESHOLD", "0.5"))
TRANSNETV2_DEVICE = os.getenv("TRANSNETV2_DEVICE", "auto")
# TransNetV2 was trained with 100-frame windows, but that activation footprint
# is not viable inside a 512 MB web service. Keep the streaming window small by
# default on every host instead of relying on provider-specific environment
# detection. Higher-memory local deployments can explicitly raise this to 50
# or 100; the value is bounded so a typo cannot create an unbounded allocation.
TRANSNETV2_WINDOW_SIZE = max(
    8, min(100, int(os.getenv("TRANSNETV2_WINDOW_SIZE", "25")))
)
TRANSNETV2_CPU_THREADS = max(
    1, int(os.getenv("TRANSNETV2_CPU_THREADS", "1"))
)

SCENE_SAMPLE_EVERY_N_FRAMES = 5
SCENE_DIFF_THRESHOLD = 0.45
MIN_SEGMENT_SECONDS = 1.0
SCENE_HISTOGRAM_GRID_ROWS = 3
SCENE_HISTOGRAM_GRID_COLUMNS = 3
SCENE_EDGE_DIFF_THRESHOLD = 0.035
SCENE_COMBINED_DIFF_THRESHOLD = 0.34
SCENE_STRONG_HSV_DIFF_THRESHOLD = 0.95
SCENE_STRONG_CUT_MIN_EDGE_DIFF = 0.0
SCENE_HSV_DIFF_WEIGHT = 0.75
SCENE_EDGE_DIFF_WEIGHT = 0.25
SCENE_REFINE_WINDOW_FRAMES = 8
SCENE_OBJECT_LAYOUT_ENABLED = os.getenv(
    "SCENE_OBJECT_LAYOUT_ENABLED", "false" if LOW_MEMORY_MODE else "true"
).lower() == "true"
SCENE_OBJECT_LAYOUT_SAMPLE_EVERY_N_FRAMES = int(
    os.getenv("SCENE_OBJECT_LAYOUT_SAMPLE_EVERY_N_FRAMES", "30")
)
SCENE_OBJECT_LAYOUT_GRID_ROWS = 3
SCENE_OBJECT_LAYOUT_GRID_COLUMNS = 3
SCENE_OBJECT_LAYOUT_DIFF_THRESHOLD = float(os.getenv("SCENE_OBJECT_LAYOUT_DIFF_THRESHOLD", "0.62"))
SCENE_OBJECT_LAYOUT_MIN_DETECTIONS = int(os.getenv("SCENE_OBJECT_LAYOUT_MIN_DETECTIONS", "4"))
SCENE_OBJECT_LAYOUT_MIN_SEGMENT_SECONDS = float(
    os.getenv("SCENE_OBJECT_LAYOUT_MIN_SEGMENT_SECONDS", "2.0")
)

# Trim applied to every segment start (except the first): swallows transition
# residue (dissolve tails, previous-clip frames) that lands after the cut.
SEGMENT_START_TRIM_SECONDS = float(os.getenv("SEGMENT_START_TRIM_SECONDS", "0.30"))
# Second-pass visual cut scan, merged with TransNetV2 cuts. Catches hard cuts
# between visually similar plays (same pitch/camera) that TransNetV2 misses.
SCENE_SECOND_PASS_ENABLED = os.getenv(
    "SCENE_SECOND_PASS_ENABLED", "false" if LOW_MEMORY_MODE else "true"
).lower() == "true"
SCENE_SECOND_PASS_DOWNSCALE_WIDTH = int(
    os.getenv("SCENE_SECOND_PASS_DOWNSCALE_WIDTH", "360" if LOW_MEMORY_MODE else "480")
)
# Seamless-cut detection via camera registration: two plays minutes apart at the
# SAME camera angle can look near-identical to color/edge histograms (identity-
# blind), so the clips get merged. But optical-flow registration between the two
# frames FAILS at a real cut — the backgrounds don't correspond. A registration
# inlier ratio below this, combined with a mild color-change floor (to not fire
# on plain motion blur), marks a cut candidate.
SCENE_MOTION_CUT_MAX_INLIER_RATIO = float(os.getenv("SCENE_MOTION_CUT_MAX_INLIER_RATIO", "0.30"))
SCENE_MOTION_CUT_MIN_HSV_DIFF = float(os.getenv("SCENE_MOTION_CUT_MIN_HSV_DIFF", "0.22"))

# Gameplay-vs-cutaway segment classification (intros, title cards, celebrations).
SEGMENT_FILTER_ENABLED = os.getenv("SEGMENT_FILTER_ENABLED", "true").lower() == "true"
SEGMENT_CLASSIFY_SAMPLES = int(
    os.getenv("SEGMENT_CLASSIFY_SAMPLES", "2" if LOW_MEMORY_MODE else "3")
)
SEGMENT_GRASS_FRACTION_MIN = float(os.getenv("SEGMENT_GRASS_FRACTION_MIN", "0.12"))
SEGMENT_GAMEPLAY_MIN_PLAYERS = int(os.getenv("SEGMENT_GAMEPLAY_MIN_PLAYERS", "3"))
SEGMENT_FILTER_USE_PLAYER_MODEL = os.getenv(
    "SEGMENT_FILTER_USE_PLAYER_MODEL", "false" if LOW_MEMORY_MODE else "true"
).lower() == "true"

# ---------------------------------------------------------------- models
FOOTBALL_MODEL_PATH = MODELS_DIR / "football-yolo11m.pt"
FOOTBALL_TRACKING_MODEL_PATH = MODELS_DIR / "football-yolo11n.pt"
YOLO11_PRETRAINED_PATH = BACKEND_DIR / (
    "yolo11n.pt" if LOW_MEMORY_MODE else "yolo11m.pt"
)
_configured_yolo_model = os.getenv("YOLO_MODEL_PATH")
if _configured_yolo_model:
    _configured_path = Path(_configured_yolo_model)
    YOLO_MODEL_PATH = str(
        _configured_path if _configured_path.is_absolute() else BACKEND_DIR / _configured_path
    )
elif FOOTBALL_MODEL_PATH.exists() and not LOW_MEMORY_MODE:
    YOLO_MODEL_PATH = str(FOOTBALL_MODEL_PATH)
else:
    YOLO_MODEL_PATH = str(YOLO11_PRETRAINED_PATH)
YOLO_MODEL_IS_FOOTBALL_SPECIFIC = Path(YOLO_MODEL_PATH).resolve() == FOOTBALL_MODEL_PATH.resolve()
_configured_tracking_model = os.getenv("TRACKING_MODEL_PATH")
if _configured_tracking_model:
    _tracking_path = Path(_configured_tracking_model)
    TRACKING_MODEL_PATH = str(
        _tracking_path if _tracking_path.is_absolute() else BACKEND_DIR / _tracking_path
    )
elif FOOTBALL_TRACKING_MODEL_PATH.exists():
    TRACKING_MODEL_PATH = str(FOOTBALL_TRACKING_MODEL_PATH)
else:
    TRACKING_MODEL_PATH = YOLO_MODEL_PATH
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.25"))
DETECTION_SAMPLE_EVERY_N_FRAMES = int(os.getenv("DETECTION_SAMPLE_EVERY_N_FRAMES", "30"))

# OSNet person-ReID (identity embeddings). Auto-falls back to the HSV descriptor
# when the checkpoint is absent — drop a torchreid osnet_x0_25 .pth at
# models/osnet_x0_25.pth and reprocess to activate.
REID_ENABLED = os.getenv("REID_ENABLED", "true").lower() == "true"
REID_MODEL_PATH = os.getenv("REID_MODEL_PATH", str(MODELS_DIR / "osnet_x0_25.pth"))
REID_INPUT_HEIGHT = int(os.getenv("REID_INPUT_HEIGHT", "256"))
REID_INPUT_WIDTH = int(os.getenv("REID_INPUT_WIDTH", "128"))
REID_EMBED_DIM = int(os.getenv("REID_EMBED_DIM", "512"))
REID_BATCH_SIZE = int(os.getenv("REID_BATCH_SIZE", "8" if LOW_MEMORY_MODE else "32"))

# ---------------------------------------------------------------- legacy greedy tracker
TRACKER_CONFIG = os.getenv("TRACKER_CONFIG", "bytetrack.yaml")
TRACKING_CONFIDENCE_THRESHOLD = float(os.getenv("TRACKING_CONFIDENCE_THRESHOLD", "0.15"))
TRACKING_FRAME_STRIDE = max(
    1, int(os.getenv("TRACKING_FRAME_STRIDE", "3" if LOW_MEMORY_MODE else "2"))
)
TRACKING_IMAGE_SIZE = int(
    os.getenv("TRACKING_IMAGE_SIZE", "512" if LOW_MEMORY_MODE else "640")
)
TRACKING_REID_MAX_CANDIDATES = max(1, int(os.getenv("TRACKING_REID_MAX_CANDIDATES", "3")))
TRACKING_MAX_MISSING_FRAMES = int(os.getenv("TRACKING_MAX_MISSING_FRAMES", "20"))
TRACKING_MAX_PREDICTION_FRAMES = int(os.getenv("TRACKING_MAX_PREDICTION_FRAMES", "12"))
TRACKING_MAX_REASSOCIATION_FRAME_FRACTION = float(
    os.getenv("TRACKING_MAX_REASSOCIATION_FRAME_FRACTION", "0.20")
)
TRACKING_LONG_GAP_FRAMES = int(os.getenv("TRACKING_LONG_GAP_FRAMES", "15"))
TRACKING_LONG_GAP_APPEARANCE = float(os.getenv("TRACKING_LONG_GAP_APPEARANCE", "0.60"))
TRACKING_REASSOCIATION_THRESHOLD = float(os.getenv("TRACKING_REASSOCIATION_THRESHOLD", "0.55"))
TRACKING_MIN_APPEARANCE_SIMILARITY = float(
    os.getenv("TRACKING_MIN_APPEARANCE_SIMILARITY", "0.35")
)
TRACKING_APPEARANCE_WEIGHT = 0.50
TRACKING_POSITION_WEIGHT = 0.25
TRACKING_IOU_WEIGHT = 0.15
TRACKING_SIZE_WEIGHT = 0.10
TRACKING_REFERENCE_UPDATE_RATE = float(os.getenv("TRACKING_REFERENCE_UPDATE_RATE", "0.03"))
TRACKING_BOX_SMOOTHING_ALPHA = float(os.getenv("TRACKING_BOX_SMOOTHING_ALPHA", "0.45"))
TRACKING_USE_NEURAL_REID = os.getenv("TRACKING_USE_NEURAL_REID", "true").lower() == "true"
TRACKING_REID_VALIDATE_EVERY_N_FRAMES = int(
    os.getenv("TRACKING_REID_VALIDATE_EVERY_N_FRAMES", "5")
)
TRACKING_TEAM_COLOR_WEIGHT = float(os.getenv("TRACKING_TEAM_COLOR_WEIGHT", "0.25"))
TRACKING_REASSOCIATION_CONFIRM_FRAMES = int(
    os.getenv("TRACKING_REASSOCIATION_CONFIRM_FRAMES", "2")
)
TRACKING_REASSOCIATION_MARGIN = float(os.getenv("TRACKING_REASSOCIATION_MARGIN", "0.04"))

# ---------------------------------------------------------------- tracklet engine
# "tracklet" = detection cache + tracklet stitching. "greedy" = legacy per-frame path.
TRACKING_ENGINE = os.getenv("TRACKING_ENGINE", "tracklet")

DETECTION_CACHE_ENABLED = os.getenv("DETECTION_CACHE_ENABLED", "true").lower() == "true"
DETECTION_CACHE_STRIDE = max(1, int(os.getenv("DETECTION_CACHE_STRIDE", "2")))
TRACKING_BATCH_SIZE = max(
    1, int(os.getenv("TRACKING_BATCH_SIZE", "1" if LOW_MEMORY_MODE else "8"))
)

# Camera motion compensation (full-affine, BoT-SORT GMC style).
CAMERA_MOTION_ENABLED = os.getenv("CAMERA_MOTION_ENABLED", "true").lower() == "true"
CAMERA_MOTION_DOWNSCALE_WIDTH = int(os.getenv("CAMERA_MOTION_DOWNSCALE_WIDTH", "480"))
CAMERA_MOTION_MIN_INLIER_RATIO = float(os.getenv("CAMERA_MOTION_MIN_INLIER_RATIO", "0.4"))
# Kinematic gate: no association may imply a player moving faster than this,
# in camera-compensated player-heights/second. A sprinting human peaks ~10 m/s
# ≈ 5.5 body-heights/s; 6.5 leaves margin for box jitter. A cross-pitch
# "teleport" (the zoom-drift failure) implies 30-100 heights/s and is rejected,
# which drops the track into the honest `searching` state (=> playback pauses).
MAX_PLAYER_SPEED_HEIGHTS_PER_SEC = float(os.getenv("MAX_PLAYER_SPEED_HEIGHTS_PER_SEC", "6.5"))
# When camera registration confidence (RANSAC inlier ratio) over an association
# span drops below this, the camera model is untrusted (hard zoom / whip pan /
# motion blur). The tracker must get MORE conservative there, never less.
CAMERA_CONFIDENCE_MIN = float(os.getenv("CAMERA_CONFIDENCE_MIN", "0.35"))
# Gate multiplier applied when camera confidence is low: shrinks position gates
# and the speed cap so only near-certain associations survive uncertain camera
# motion. (The old behavior widened gates on failure — backwards for safety.)
CAMERA_UNCERTAIN_GATE_FACTOR = float(os.getenv("CAMERA_UNCERTAIN_GATE_FACTOR", "0.5"))

# Tracklet building: strict on purpose — tracklets TERMINATE at ambiguity and
# the stitcher resolves identity later with full-clip context.
TRACKLET_MIN_IOU = float(os.getenv("TRACKLET_MIN_IOU", "0.30"))
TRACKLET_AMBIGUITY_RATIO = float(os.getenv("TRACKLET_AMBIGUITY_RATIO", "1.3"))
TRACKLET_MIN_LENGTH = int(os.getenv("TRACKLET_MIN_LENGTH", "2"))
TRACKLET_MAX_MISSED_SAMPLES = int(os.getenv("TRACKLET_MAX_MISSED_SAMPLES", "2"))

# Stitching: links between tracklets weigh position/appearance/team/size;
# ambiguous links are left as honest gaps.
STITCH_MAX_GAP_FRAMES = int(os.getenv("STITCH_MAX_GAP_FRAMES", "90"))
STITCH_LINK_THRESHOLD = float(os.getenv("STITCH_LINK_THRESHOLD", "0.55"))
STITCH_WINNER_MARGIN = float(os.getenv("STITCH_WINNER_MARGIN", "0.85"))
# Appearance tiebreak: an ambiguous link may still be accepted when the best
# candidate carries decisively stronger identity evidence than the runner-up.
STITCH_TIEBREAK_MIN_APPEARANCE = float(os.getenv("STITCH_TIEBREAK_MIN_APPEARANCE", "0.60"))
STITCH_TIEBREAK_APPEARANCE_MARGIN = float(os.getenv("STITCH_TIEBREAK_APPEARANCE_MARGIN", "0.15"))
STITCH_POSITION_WEIGHT = float(os.getenv("STITCH_POSITION_WEIGHT", "0.35"))
STITCH_APPEARANCE_WEIGHT = float(os.getenv("STITCH_APPEARANCE_WEIGHT", "0.30"))
STITCH_TEAM_WEIGHT = float(os.getenv("STITCH_TEAM_WEIGHT", "0.20"))
STITCH_SIZE_WEIGHT = float(os.getenv("STITCH_SIZE_WEIGHT", "0.15"))
STITCH_LONG_GAP_FRAMES = int(os.getenv("STITCH_LONG_GAP_FRAMES", "30"))
STITCH_LONG_GAP_MIN_APPEARANCE = float(os.getenv("STITCH_LONG_GAP_MIN_APPEARANCE", "0.60"))
# Fallback identity evidence across long gaps when appearance is unavailable
# (tiny crops): jersey-color agreement plus a tight position gate.
STITCH_LONG_GAP_MIN_JERSEY = float(os.getenv("STITCH_LONG_GAP_MIN_JERSEY", "0.70"))
# A cross-team link is fatal UNLESS appearance strongly overrides — with OSNet
# embeddings, different players genuinely score low, so this stops team drift.
# Calibrate against real clips via metrics.stitch_decisions.
STITCH_CROSS_TEAM_APPEARANCE_OVERRIDE = float(
    os.getenv("STITCH_CROSS_TEAM_APPEARANCE_OVERRIDE", "0.65")
)
# Joint crossing resolution: when two tracklets end together (a crossing), the
# 2x2 pairing with the two continuation tracklets is solved jointly instead of
# greedily — the classic fix for identity swaps at crossings.
CROSSING_COENDER_WINDOW_FRAMES = int(os.getenv("CROSSING_COENDER_WINDOW_FRAMES", "15"))
CROSSING_JOINT_MARGIN = float(os.getenv("CROSSING_JOINT_MARGIN", "0.05"))

# Appearance descriptors: only crops wide enough to carry signal, never crops
# overlapping another player (contaminated pixels cause wrong relinks).
APPEARANCE_MIN_CROP_WIDTH = int(os.getenv("APPEARANCE_MIN_CROP_WIDTH", "24"))
APPEARANCE_TOP_K_CROPS = int(os.getenv("APPEARANCE_TOP_K_CROPS", "5"))
APPEARANCE_MAX_OVERLAP_IOU = float(os.getenv("APPEARANCE_MAX_OVERLAP_IOU", "0.30"))

INTERPOLATE_MAX_GAP_FRAMES = int(os.getenv("INTERPOLATE_MAX_GAP_FRAMES", "20"))
# Interpolation is only believable when the endpoints are spatially close
# (camera-compensated). Beyond this, show "searching" instead of a floating box.
INTERPOLATE_MAX_DISTANCE_HEIGHTS = float(os.getenv("INTERPOLATE_MAX_DISTANCE_HEIGHTS", "3.0"))

# Recovery pass: re-attach visible-but-unstitched detections inside chain gaps
# and past chain ends. Strict uniqueness + jersey gates keep identity safe.
RECOVERY_ENABLED = os.getenv("RECOVERY_ENABLED", "true").lower() == "true"
RECOVERY_MAX_DISTANCE_HEIGHTS = float(os.getenv("RECOVERY_MAX_DISTANCE_HEIGHTS", "1.2"))
RECOVERY_MIN_JERSEY_SIMILARITY = float(os.getenv("RECOVERY_MIN_JERSEY_SIMILARITY", "0.50"))
RECOVERY_UNIQUENESS_RATIO = float(os.getenv("RECOVERY_UNIQUENESS_RATIO", "0.60"))
RECOVERY_MAX_CONSECUTIVE_MISSES = int(os.getenv("RECOVERY_MAX_CONSECUTIVE_MISSES", "8"))
# Recovery may only claim ORPHAN detections — anything owned by another
# established tracklet is another player's; stealing it is identity drift.
RECOVERY_EXCLUDE_TRACKLET_LENGTH = int(os.getenv("RECOVERY_EXCLUDE_TRACKLET_LENGTH", "4"))
# Candidate box height must sit inside this factor band of the expected height —
# stops recovery from grabbing a player at a different depth.
RECOVERY_SIZE_BAND = (0.55, 1.8)
RECOVERY_MAX_OVERLAP_IOU = float(os.getenv("RECOVERY_MAX_OVERLAP_IOU", "0.40"))

# Pitch mask + searching state.
PITCH_MASK_ENABLED = os.getenv("PITCH_MASK_ENABLED", "true").lower() == "true"
PITCH_MASK_MIN_GRASS_FRACTION = float(os.getenv("PITCH_MASK_MIN_GRASS_FRACTION", "0.25"))
PITCH_MASK_MARGIN_FRACTION = float(os.getenv("PITCH_MASK_MARGIN_FRACTION", "0.05"))
SEARCH_TIMEOUT_FRAMES = int(os.getenv("SEARCH_TIMEOUT_FRAMES", "120"))

# ---------------------------------------------------------------- ball + events
# Ball filtering: turf dots and line intersections fire the ball class.
# A real ball MOVES in camera-compensated coordinates; painted marks do not.
BALL_MIN_CONFIDENCE = float(os.getenv("BALL_MIN_CONFIDENCE", "0.30"))
# A ball candidate whose compensated position hosts detections in more than this
# fraction of sampled frames is a static pitch mark, not a ball.
BALL_STATIC_FRACTION = float(os.getenv("BALL_STATIC_FRACTION", "0.35"))
BALL_STATIC_RADIUS_FACTOR = float(os.getenv("BALL_STATIC_RADIUS_FACTOR", "1.0"))

# Ball events are PAUSED until ball detection is reliable on real footage —
# tiny fast balls fragment possession and fabricate counts. Movement stats stay
# on; flipping this re-enables time-with-ball / passes / shots end to end.
BALL_EVENTS_ENABLED = os.getenv("BALL_EVENTS_ENABLED", "false").lower() == "true"
# "Near the feet" radius for possession/time-with-ball, in player-heights.
POSSESSION_RADIUS_HEIGHTS = float(os.getenv("POSSESSION_RADIUS_HEIGHTS", "0.9"))
TOUCH_RADIUS_HEIGHTS = float(os.getenv("TOUCH_RADIUS_HEIGHTS", "0.9"))
# Possession must persist this many sampled frames — single-frame flybys are not
# possession.
POSSESSION_MIN_SAMPLES = int(os.getenv("POSSESSION_MIN_SAMPLES", "2"))
# A pass must travel far enough to be a deliberate ball movement, and be
# received within the window by a same-team player.
PASS_MIN_TRAVEL_HEIGHTS = float(os.getenv("PASS_MIN_TRAVEL_HEIGHTS", "2.0"))
PASS_RECEIVE_WINDOW_SECONDS = float(os.getenv("PASS_RECEIVE_WINDOW_SECONDS", "1.5"))
PASS_TEAMMATE_MIN_JERSEY = float(os.getenv("PASS_TEAMMATE_MIN_JERSEY", "0.45"))
# Shots: fast releases. The goal itself is not visible, so speed + non-reception
# is the signal; goalkeeper-directed releases are the high-confidence subset.
SHOT_SPEED_HEIGHTS_PER_SEC = float(os.getenv("SHOT_SPEED_HEIGHTS_PER_SEC", "8.0"))
# Low-confidence "shot attempt": fast, unreceived, and travelled at least this
# far (also catches clearances/crosses — reported as attempts, not shots).
SHOT_ATTEMPT_MIN_TRAVEL_HEIGHTS = float(os.getenv("SHOT_ATTEMPT_MIN_TRAVEL_HEIGHTS", "3.0"))

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def ensure_storage_directories() -> None:
    for directory in (
        RAW_VIDEOS_DIR,
        THUMBNAILS_DIR,
        RESULTS_DIR,
        ULTRALYTICS_CONFIG_DIR,
        MODELS_DIR,
        DETECTIONS_DIR,
        SEGMENTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
