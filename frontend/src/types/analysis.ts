export interface VideoUploadResponse {
  video_id: string;
  filename: string;
  message: string;
}

export type ProcessingStageStatus = "pending" | "active" | "complete" | "failed";

export interface ProcessingStageProgress {
  key: "scene_cuts" | "cutaway_filtering" | "thumbnails";
  label: string;
  status: ProcessingStageStatus;
  progress_percent: number;
  completed_items: number;
  total_items: number | null;
}

export interface VideoProcessingProgress {
  status: "idle" | "processing" | "complete" | "failed";
  progress_percent: number;
  current_stage: ProcessingStageProgress["key"] | null;
  message: string;
  stages: ProcessingStageProgress[];
}

export interface VideoMetadata {
  fps: number;
  frame_count: number;
  duration_seconds: number;
  width: number;
  height: number;
}

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface ObjectDetection {
  detection_id: string;
  role: "player" | "ball" | "goalkeeper" | "referee";
  model_class: string;
  confidence: number;
  bbox: BoundingBox;
  team_id: string | null;
  jersey_color_hex: string | null;
  jersey_descriptor: number[] | null;
}

export interface FrameDetectionResponse {
  timestamp_seconds: number;
  frame_number: number;
  frame_width: number;
  frame_height: number;
  detections: ObjectDetection[];
}

export interface ObjectDetectionSummary {
  status: string;
  model_path: string;
  start_time: number;
  sample_every_n_frames: number;
  sampled_frames: number;
  counts: Record<ObjectDetection["role"], number>;
  counting_note: string;
  role_note: string;
}

export interface SegmentAnalysis {
  segment_id: string;
  start_time: number;
  end_time: number;
  /** "gameplay" | "cutaway" — intros, title cards, celebrations. Missing = gameplay. */
  kind?: string | null;
  gameplay_score?: number | null;
  thumbnail_path: string;
  focused_player_status: string;
  focused_player_selection: {
    detection_id: string;
    selected_at_time: number;
    bbox: BoundingBox;
    confidence: number;
    team_id: string | null;
    jersey_color_hex: string | null;
    jersey_descriptor: number[] | null;
  } | null;
  focused_player_track: FocusedPlayerTrack | null;
  object_detection_summary: ObjectDetectionSummary | null;
  detected_actions_placeholder: unknown[];
  features_placeholder: Record<string, number | null>;
  /** Real per-clip stats, present once the clip is tracked. */
  features?: Record<string, number | string | boolean | null> | null;
}

export type TrackSampleState = "tracked" | "recovered" | "interpolated" | "searching" | "ended";

export interface TrackSample {
  frame_number: number;
  timestamp_seconds: number;
  clip_time_seconds: number;
  /** null while the tracker is in "searching" state — no box is invented. */
  bbox: BoundingBox | null;
  confidence: number;
  predicted: boolean;
  state?: TrackSampleState;
  tracklet_id?: number | null;
  search_center?: [number, number] | null;
}

export interface FocusedPlayerTrack {
  status: string;
  tracker: string;
  engine?: string;
  track_id: number;
  start_time: number;
  end_time: number;
  frame_width: number;
  frame_height: number;
  samples: TrackSample[];
  source_frames?: number;
  inference_frames?: number;
  frame_stride?: number;
  processing_seconds?: number;
  metrics?: {
    detection_cache_hit?: boolean;
    tracklet_count?: number;
    stitched_tracklet_count?: number;
    coverage?: number;
    searching_fraction?: number;
    [key: string]: unknown;
  };
  /** Per-clip stats attached by the track endpoint (also persisted on the segment). */
  clip_features?: Record<string, number | string | boolean | null> | null;
}

export type Footedness = "right" | "left" | "both";

export interface PlayerInfo {
  positions: string[];
  footedness: Footedness;
}

export interface ArchetypeScore {
  archetype: string;
  label: string;
  score: number;
}

export interface PlayerProfile {
  status: string;
  primary_archetype: string;
  primary_label: string;
  description: string;
  group: string;
  confidence: "low" | "medium" | "high";
  scores: ArchetypeScore[];
  traits: Record<string, number>;
  features: {
    minutes_tracked: number;
    segments_analyzed: number;
    involvement_per_min: number;
    distance_per_min: number;
    sprints_per_min: number;
    wideness: number | null;
    time_with_ball_seconds?: number;
    time_with_ball_per_min?: number;
    passes?: number;
    shots?: number;
    shot_attempts?: number;
    passes_per_min?: number;
    shots_per_min?: number;
    shot_attempts_per_min?: number;
    ball_data_available: boolean;
    /** Ball-event section master switch (currently paused). When false, event
     * stats/chips are hidden regardless of the values above. */
    ball_events_enabled?: boolean;
  };
  evidence: string[];
  positions: string[];
  footedness: string;
  note: string;
}

export interface VideoAnalysisResult {
  video_id: string;
  status: string;
  metadata: VideoMetadata;
  segments: SegmentAnalysis[];
  demo?: {
    id: string;
    title: string;
    description: string;
    video_path: string;
    read_only: true;
    tracking_showcase_segment_id?: string;
  } | null;
  final_profile_placeholder: {
    predicted_role: string;
    confidence: string;
    summary: string;
  };
  player_info?: PlayerInfo | null;
  player_profile?: PlayerProfile | null;
}
