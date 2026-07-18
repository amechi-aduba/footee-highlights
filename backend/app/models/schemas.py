from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str


class VideoUploadResponse(BaseModel):
    video_id: str
    filename: str
    message: str


class ProcessingStageProgress(BaseModel):
    key: str
    label: str
    status: str
    progress_percent: float
    completed_items: int
    total_items: int | None = None


class VideoProcessingProgressResponse(BaseModel):
    status: str
    progress_percent: float
    current_stage: str | None = None
    message: str
    stages: list[ProcessingStageProgress]


class VideoProcessingStartResponse(BaseModel):
    video_id: str
    status: str
    message: str


class VideoMetadata(BaseModel):
    fps: float
    frame_count: int
    duration_seconds: float
    width: int
    height: int


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class ObjectDetection(BaseModel):
    detection_id: str
    role: str
    model_class: str
    confidence: float
    bbox: BoundingBox
    team_id: str | None = None
    jersey_color_hex: str | None = None
    jersey_descriptor: list[float] | None = None


class DetectionCounts(BaseModel):
    player: int = 0
    ball: int = 0
    goalkeeper: int = 0
    referee: int = 0


class ObjectDetectionSummary(BaseModel):
    status: str
    model_path: str
    start_time: float
    sample_every_n_frames: int
    sampled_frames: int
    counts: DetectionCounts
    counting_note: str
    role_note: str


class FocusedPlayerSelection(BaseModel):
    detection_id: str
    selected_at_time: float
    bbox: BoundingBox
    confidence: float
    team_id: str | None = None
    jersey_color_hex: str | None = None
    jersey_descriptor: list[float] | None = None


class TrackSample(BaseModel):
    frame_number: int
    timestamp_seconds: float
    clip_time_seconds: float
    # None while in "searching" state — the tracker refuses to invent a box.
    bbox: BoundingBox | None = None
    confidence: float
    predicted: bool = False
    state: str = "tracked"  # tracked | recovered | interpolated | searching | ended
    tracklet_id: int | None = None
    search_center: list[float] | None = None


class FocusedPlayerTrack(BaseModel):
    status: str
    tracker: str
    engine: str | None = None
    track_id: int
    start_time: float
    end_time: float
    frame_width: int
    frame_height: int
    samples: list[TrackSample]
    source_frames: int | None = None
    inference_frames: int | None = None
    frame_stride: int | None = None
    processing_seconds: float | None = None
    metrics: dict[str, Any] | None = None


class SegmentAnalysis(BaseModel):
    segment_id: str
    start_time: float
    end_time: float
    kind: str | None = None  # "gameplay" | "cutaway" (intros, title cards, celebrations)
    gameplay_score: float | None = None
    thumbnail_path: str
    focused_player_status: str
    focused_player_selection: FocusedPlayerSelection | None = None
    focused_player_anchors: list[FocusedPlayerSelection] | None = None
    focused_player_track: FocusedPlayerTrack | None = None
    object_detection_summary: ObjectDetectionSummary | None = None
    detected_actions_placeholder: list[Any]
    features_placeholder: dict[str, float | None]
    # Real per-clip stats, computed right after tracking (distance, sprints,
    # ball-near events, touches/passes/shots, wideness).
    features: dict[str, Any] | None = None


class FinalProfilePlaceholder(BaseModel):
    predicted_role: str
    confidence: str
    summary: str


class PlayerInfo(BaseModel):
    positions: list[str]
    footedness: str  # "right" | "left" | "both"


class PlayerInfoRequest(PlayerInfo):
    pass


class VideoAnalysisResult(BaseModel):
    video_id: str
    status: str
    metadata: VideoMetadata
    segments: list[SegmentAnalysis]
    final_profile_placeholder: FinalProfilePlaceholder
    player_info: PlayerInfo | None = None
    player_profile: dict[str, Any] | None = None


class SegmentDetectionRequest(BaseModel):
    clip_time_seconds: float


class FrameDetectionResponse(BaseModel):
    timestamp_seconds: float
    frame_number: int
    frame_width: int
    frame_height: int
    detections: list[ObjectDetection]


class FocusedPlayerSelectionRequest(BaseModel):
    detection_id: str
    clip_time_seconds: float
    bbox: BoundingBox
    confidence: float
    team_id: str | None = None
    jersey_color_hex: str | None = None
    jersey_descriptor: list[float] | None = None
    
    # True = add an anchor for the SAME player (used after tracking was lost);
    # False = fresh selection, replacing all anchors.
    additive: bool = False
