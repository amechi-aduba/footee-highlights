import { useRef, useState } from "react";
import {
  API_BASE_URL,
  detectSegmentFrame,
  resetFocusedPlayer,
  selectFocusedPlayer,
  trackFocusedPlayer,
} from "../api/client";
import type {
  FrameDetectionResponse,
  ObjectDetection,
  FocusedPlayerTrack,
  SegmentAnalysis,
} from "../types/analysis";
import { clipLabel } from "../utils/video";

function formatTimestamp(totalSeconds: number) {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, "0")}`;
}

interface SegmentCardProps {
  segment: SegmentAnalysis;
  videoId: string;
  videoUrl?: string;
  thumbnailUrl?: string;
  readOnly?: boolean;
  /** Present when the card is opened from the clip grid — shows a close button. */
  onClose?: () => void;
  /** Lets the clip grid update its status badge after select/track/reset. */
  onStatusChange?: (status: "selected" | "tracked" | "not_selected") => void;
}

export function SegmentCard({
  segment,
  videoId,
  videoUrl: providedVideoUrl,
  thumbnailUrl,
  readOnly = false,
  onClose,
  onStatusChange,
}: SegmentCardProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const clipDuration = Math.max(0, segment.end_time - segment.start_time);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [frameDetections, setFrameDetections] = useState<FrameDetectionResponse | null>(null);
  const [selectedTeamId, setSelectedTeamId] = useState<string>("all");
  const [focusedPlayerId, setFocusedPlayerId] = useState(
    segment.focused_player_selection?.detection_id ?? "",
  );
  const [focusedTrack, setFocusedTrack] = useState<FocusedPlayerTrack | null>(
    segment.focused_player_track,
  );
  const [isTracking, setIsTracking] = useState(false);
  const [anchorAdded, setAnchorAdded] = useState(false);
  const [error, setError] = useState("");
  const [isDetectingFrame, setIsDetectingFrame] = useState(false);
  const [lostAtTime, setLostAtTime] = useState<number | null>(null);
  const [clipStats, setClipStats] = useState<Record<string, number | string | boolean | null> | null>(
    segment.features ?? null,
  );
  const hasAutoDetected = useRef(false);
  const lastLostPauseTime = useRef<number | null>(null);
  const videoUrl = providedVideoUrl ?? `${API_BASE_URL}/api/videos/${videoId}/video`;

  function seekToClipTime(clipTime: number) {
    const video = videoRef.current;
    if (!video) return;
    const nextTime = Math.min(Math.max(clipTime, 0), clipDuration);
    video.currentTime = segment.start_time + nextTime;
    setCurrentTime(nextTime);
    setFrameDetections(null);
    setSelectedTeamId("all");
  }

  function handleTimeUpdate() {
    const video = videoRef.current;
    if (!video) return;
    const clipTime = video.currentTime - segment.start_time;
    if (clipTime >= clipDuration) {
      video.pause();
      seekToClipTime(clipDuration);
      return;
    }
    if (clipTime < 0) {
      seekToClipTime(0);
      return;
    }
    setCurrentTime(clipTime);
    maybePauseAtLostMoment(clipTime);
  }

  /** Pause playback the moment the tracker loses the player, using the REAL
   * playhead time (the pre-computed "first lost" estimate can be off). Pauses
   * once per gap so resuming playback doesn't immediately re-pause. */
  function maybePauseAtLostMoment(clipTime: number) {
    const video = videoRef.current;
    if (!video || video.paused || !focusedTrack || anchorAdded) return;
    const samples = focusedTrack.samples;
    if (samples.length === 0) return;

    const nearest = samples.reduce((closest, sample) =>
      Math.abs(sample.clip_time_seconds - clipTime) < Math.abs(closest.clip_time_seconds - clipTime)
        ? sample
        : closest,
    );
    const nearestState = nearest.state ?? (nearest.predicted ? "interpolated" : "tracked");
    const lastSample = samples[samples.length - 1];
    const isSearching =
      Math.abs(nearest.clip_time_seconds - clipTime) < 0.5 && nearestState === "searching";
    const isPastTrackEnd =
      clipTime > lastSample.clip_time_seconds + 0.3 &&
      lastSample.clip_time_seconds < clipDuration - 0.75;
    if (!isSearching && !isPastTrackEnd) return;
    if (
      lastLostPauseTime.current != null &&
      Math.abs(clipTime - lastLostPauseTime.current) < 1.5
    ) {
      return; // user resumed inside the same gap — don't fight them
    }
    lastLostPauseTime.current = clipTime;
    video.pause();
    setLostAtTime(clipTime);
    void handleDetectFrame(clipTime); // boxes ready for a re-anchor click
  }

  async function togglePlayback() {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      if (currentTime >= clipDuration) seekToClipTime(0);
      await video.play();
    } else {
      video.pause();
    }
  }

  async function handleDetectFrame(atTime?: number) {
    videoRef.current?.pause();
    setError("");
    setIsDetectingFrame(true);
    try {
      const detections = await detectSegmentFrame(videoId, segment.segment_id, atTime ?? currentTime);
      setFrameDetections(detections);
      setSelectedTeamId("all");
    } catch (detectionError) {
      setError(detectionError instanceof Error ? detectionError.message : "Frame detection failed.");
    } finally {
      setIsDetectingFrame(false);
    }
  }

  async function handleResetSelection() {
    setError("");
    try {
      await resetFocusedPlayer(videoId, segment.segment_id);
      setFocusedPlayerId("");
      setFocusedTrack(null);
      setAnchorAdded(false);
      setLostAtTime(null);
      lastLostPauseTime.current = null;
      onStatusChange?.("not_selected");
    } catch (resetError) {
      setError(resetError instanceof Error ? resetError.message : "Could not reset the selection.");
    }
  }

  async function handleSelectPlayer(detection: ObjectDetection) {
    setError("");
    // A selection made AFTER a track exists means: "this is the same player at
    // a point where you lost them" — an extra anchor, not a fresh start.
    const isAnchor = focusedTrack != null;
    try {
      await selectFocusedPlayer(videoId, segment.segment_id, {
        detection_id: detection.detection_id,
        clip_time_seconds: currentTime,
        bbox: detection.bbox,
        confidence: detection.confidence,
        team_id: detection.team_id,
        jersey_color_hex: detection.jersey_color_hex,
        jersey_descriptor: detection.jersey_descriptor,
        additive: isAnchor,
      });
      setFocusedPlayerId(detection.detection_id);
      if (isAnchor) {
        setAnchorAdded(true); // keep the current track visible until re-track
      } else {
        setFocusedTrack(null);
      }
      onStatusChange?.("selected");
    } catch (selectionError) {
      setError(selectionError instanceof Error ? selectionError.message : "Player selection failed.");
    }
  }

  async function handleTrackPlayer() {
    setError("");
    setIsTracking(true);
    try {
      const trackResult = await trackFocusedPlayer(videoId, segment.segment_id);
      setFocusedTrack(trackResult);
      setClipStats(trackResult.clip_features ?? null);
      setFrameDetections(null); // selection boxes served their purpose
      setAnchorAdded(false);
      setLostAtTime(null);
      lastLostPauseTime.current = null;
      onStatusChange?.("tracked");
    } catch (trackingError) {
      setError(trackingError instanceof Error ? trackingError.message : "Player tracking failed.");
    } finally {
      setIsTracking(false);
    }
  }

  const activeTrackSample = focusedTrack?.samples.reduce((closest, sample) =>
    Math.abs(sample.clip_time_seconds - currentTime) < Math.abs(closest.clip_time_seconds - currentTime)
      ? sample
      : closest,
  );
  // Older tracks have no `state`; derive it from `predicted` for compatibility.
  const activeSampleState =
    activeTrackSample?.state ?? (activeTrackSample?.predicted ? "interpolated" : "tracked");
  const isActiveSampleNearby =
    activeTrackSample != null && Math.abs(activeTrackSample.clip_time_seconds - currentTime) < 0.5;

  // Where did tracking lose the player? First "searching" run, or an early end.
  const firstLostTime = (() => {
    if (!focusedTrack) return null;
    const searching = focusedTrack.samples.find((sample) => sample.state === "searching");
    if (searching) return searching.clip_time_seconds;
    const lastSample = focusedTrack.samples[focusedTrack.samples.length - 1];
    if (lastSample && lastSample.clip_time_seconds < clipDuration - 0.75) {
      return lastSample.clip_time_seconds;
    }
    return null;
  })();

  async function jumpToLostPoint() {
    if (firstLostTime == null) return;
    const target = Math.min(clipDuration, firstLostTime + 0.2);
    videoRef.current?.pause();
    seekToClipTime(target);
    await handleDetectFrame(target);
  }

  const detectedTeams = Array.from(
    new Map(
      (frameDetections?.detections ?? [])
        .filter((detection) => detection.team_id && detection.jersey_color_hex)
        .map((detection) => [
          detection.team_id as string,
          { id: detection.team_id as string, color: detection.jersey_color_hex as string },
        ]),
    ).values(),
  );
  const visibleDetections = (frameDetections?.detections ?? []).filter(
    (detection) =>
      selectedTeamId === "all" ||
      detection.role === "ball" ||
      detection.team_id === selectedTeamId,
  );

  return (
    <article className="card overflow-hidden">
      <div className="relative aspect-video bg-slate-900">
        <video
          ref={videoRef}
          className="h-full w-full object-contain"
          preload="metadata"
          poster={thumbnailUrl ?? `${API_BASE_URL}${segment.thumbnail_path}`}
          src={videoUrl}
          onLoadedMetadata={() => {
            seekToClipTime(0);
            // Detect players immediately on open so the user sees selectable
            // boxes on the first frame without an extra click. Skip if this
            // clip was already tracked — the verify overlay is what matters.
            if (!readOnly && !hasAutoDetected.current && !focusedTrack) {
              hasAutoDetected.current = true;
              void handleDetectFrame();
            }
          }}
          onPause={() => setIsPlaying(false)}
          onPlay={() => {
            setIsPlaying(true);
            // Frame detections belong to one paused frame — stale and messy
            // once the video moves. The focused-player track has its own overlay.
            setFrameDetections(null);
          }}
          onTimeUpdate={handleTimeUpdate}
        />
        {frameDetections && visibleDetections.map((detection) => {
          const { bbox } = detection;
          const isSelectable = detection.role === "player" || detection.role === "goalkeeper";
          const isSelected = focusedPlayerId === detection.detection_id;
          return (
            <button
              key={detection.detection_id}
              className={`absolute border-2 ${isSelected ? "border-emerald-400 bg-emerald-300/20" : "border-amber-300 bg-amber-300/10"} ${isSelectable ? "cursor-pointer" : "cursor-default"}`}
              style={{
                left: `${(bbox.x1 / frameDetections.frame_width) * 100}%`,
                top: `${(bbox.y1 / frameDetections.frame_height) * 100}%`,
                width: `${((bbox.x2 - bbox.x1) / frameDetections.frame_width) * 100}%`,
                height: `${((bbox.y2 - bbox.y1) / frameDetections.frame_height) * 100}%`,
              }}
              type="button"
              disabled={!isSelectable}
              title={`${detection.role} (${Math.round(detection.confidence * 100)}%)${detection.team_id ? ` - ${detection.team_id.replace("_", " ")}` : ""}`}
              onClick={() => handleSelectPlayer(detection)}
            />
          );
        })}
        {/* Broadcast-style spotlight at the tracked player's feet. Only rendered
            during playback — on pause the overlay clears (a stale marker on a
            paused frame reads as tracking the wrong spot). */}
        {focusedTrack &&
          activeTrackSample &&
          isActiveSampleNearby &&
          isPlaying &&
          activeTrackSample.bbox &&
          (() => {
            const bbox = activeTrackSample.bbox;
            const spotWidth = (bbox.x2 - bbox.x1) * 1.7;
            const spotHeight = spotWidth * 0.36;
            const centerX = (bbox.x1 + bbox.x2) / 2;
            const isInterpolated = activeSampleState === "interpolated";
            return (
              <div
                className={`pointer-events-none absolute rounded-[50%] border-2 ${
                  isInterpolated
                    ? "border-dashed border-yellow-300/80 opacity-60"
                    : "border-emerald-400 bg-emerald-400/10 shadow-[0_0_14px_2px_rgba(52,211,153,0.45)]"
                }`}
                style={{
                  left: `${((centerX - spotWidth / 2) / focusedTrack.frame_width) * 100}%`,
                  top: `${((bbox.y2 - spotHeight / 2) / focusedTrack.frame_height) * 100}%`,
                  width: `${(spotWidth / focusedTrack.frame_width) * 100}%`,
                  height: `${(spotHeight / focusedTrack.frame_height) * 100}%`,
                }}
              />
            );
          })()}
        {focusedTrack &&
          activeTrackSample &&
          isActiveSampleNearby &&
          isPlaying &&
          !activeTrackSample.bbox &&
          activeSampleState === "searching" &&
          activeTrackSample.search_center && (
            <div
              className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2"
              style={{
                left: `${(activeTrackSample.search_center[0] / focusedTrack.frame_width) * 100}%`,
                top: `${(activeTrackSample.search_center[1] / focusedTrack.frame_height) * 100}%`,
              }}
            >
              <span className="block h-6 w-6 animate-ping rounded-full border-2 border-amber-400/80" />
              <span className="absolute left-1/2 top-8 -translate-x-1/2 whitespace-nowrap rounded bg-slate-900/70 px-1.5 py-0.5 text-[10px] font-semibold text-amber-300">
                Searching…
              </span>
            </div>
          )}
      </div>
      <div className="p-4">
        <div className="mb-4 flex items-center gap-3">
          <button className="btn-primary btn-sm" type="button" onClick={togglePlayback}>
            {isPlaying ? "Pause" : "Play"}
          </button>
          {!readOnly && (
            <button
              className="btn-accent btn-sm whitespace-nowrap"
              type="button"
              disabled={!focusedPlayerId || isTracking}
              onClick={handleTrackPlayer}
            >
              {isTracking && <span className="spinner" />}
              {isTracking ? "Tracking…" : "Track selected player"}
            </button>
          )}
          <input
            className="min-w-0 flex-1 accent-primary"
            type="range"
            min="0"
            max={clipDuration}
            step="0.1"
            value={currentTime}
            aria-label={`Scrub ${segment.segment_id}`}
            onChange={(event) => seekToClipTime(Number(event.target.value))}
          />
          <span className="whitespace-nowrap text-xs text-mute">
            {formatTimestamp(currentTime)} / {formatTimestamp(clipDuration)}
          </span>
          {onClose && (
            <button
              className="btn-ghost btn-sm"
              type="button"
              onClick={() => {
                videoRef.current?.pause();
                onClose();
              }}
            >
              Close
            </button>
          )}
        </div>
        {!readOnly && (
          <div className="mb-4 flex flex-wrap gap-2">
            <button
              className="btn-primary btn-sm"
              type="button"
              disabled={isDetectingFrame}
              onClick={() => handleDetectFrame()}
            >
              {isDetectingFrame && <span className="spinner" />}
              {isDetectingFrame ? "Detecting…" : "Detect players in this frame"}
            </button>
            {(focusedPlayerId || focusedTrack) && (
              <button
                className="rounded-lg border border-red-400/50 px-3 py-1.5 text-xs font-semibold text-red-500 transition-colors hover:bg-red-500/10"
                type="button"
                onClick={handleResetSelection}
                title="Wrong player? Clear the selection, anchors, and track to start over."
              >
                Reset selection
              </button>
            )}
          </div>
        )}
        {detectedTeams.length > 0 && (
          <div className="mb-4 rounded-lg border border-line bg-surface2/60 p-3">
            <p className="mb-2 text-xs font-semibold text-ink">Filter player boxes by jersey color</p>
            <div className="flex flex-wrap gap-2">
              <button
                className={`rounded-md border px-3 py-1.5 text-xs font-semibold transition-colors ${selectedTeamId === "all" ? "border-primary bg-primary text-white" : "border-line bg-surface text-mute hover:text-ink"}`}
                type="button"
                onClick={() => setSelectedTeamId("all")}
              >
                All players
              </button>
              {detectedTeams.map((team, index) => (
                <button
                  key={team.id}
                  className={`flex items-center gap-2 rounded-md border px-3 py-1.5 text-xs font-semibold transition-colors ${selectedTeamId === team.id ? "border-primary bg-primary/10 text-primary" : "border-line bg-surface text-mute hover:text-ink"}`}
                  type="button"
                  onClick={() => setSelectedTeamId(team.id)}
                >
                  <span
                    className="h-3.5 w-3.5 rounded-full border border-black/20"
                    style={{ backgroundColor: team.color }}
                  />
                  Team {index + 1}
                </button>
              ))}
            </div>
          </div>
        )}
        <div className="flex items-center justify-between gap-3">
          <h3 className="font-semibold">{clipLabel(segment.segment_id)}</h3>
          <span className="text-xs text-mute">
            {formatTimestamp(segment.start_time)} - {formatTimestamp(segment.end_time)}
          </span>
        </div>
        {readOnly ? (
          <p className="mt-2 text-xs leading-relaxed text-mute">
            Preprocessed sample clip. Playback and scrubbing use static assets only; upload
            your own reel to select and track a player.
          </p>
        ) : (
          <>
            <p className="mt-2 text-xs uppercase tracking-wide text-mute">
              Focused player: {segment.focused_player_status.replace("_", " ")}
            </p>
            <p className="mt-2 text-xs text-mute">
              Players are detected automatically when the clip opens — click a box to
              select, then track. Scrub and re-detect any time to pick from a
              different frame.
            </p>
          </>
        )}
        {error && <p className="mt-3 text-xs text-red-500">{error}</p>}
        {frameDetections && (
          <p className="mt-3 text-xs text-mute">
            Found {frameDetections.detections.length} objects on this frame. Showing {visibleDetections.length}.
          </p>
        )}
        {focusedPlayerId && (
          <p className="mt-3 text-xs font-semibold text-emerald-500">
            Player selected — hit “Track selected player” to follow them.
          </p>
        )}
        {focusedTrack && (
          <p className="mt-3 text-xs font-semibold text-primary">
            {focusedTrack.engine === "tracklet" ? "Tracklet engine" : "ByteTrack"} followed this
            player across {focusedTrack.samples.length} frames
            {focusedTrack.processing_seconds != null
              ? ` in ${focusedTrack.processing_seconds.toFixed(1)} seconds.`
              : "."}
            {focusedTrack.metrics?.coverage != null &&
              ` Coverage ${Math.round((focusedTrack.metrics.coverage as number) * 100)}%.`}
            {typeof focusedTrack.metrics?.anchor_count === "number" &&
              (focusedTrack.metrics.anchor_count as number) > 1 &&
              ` ${focusedTrack.metrics.anchor_count as number} anchors.`}
          </p>
        )}
        {focusedTrack && anchorAdded && (
          <p className="mt-2 rounded-lg bg-primary/10 p-2.5 text-xs font-semibold text-primary">
            Anchor added. Press “Track selected player” to re-track through all
            anchors — it uses cached detections, so it only takes a moment.
          </p>
        )}
        {focusedTrack && !anchorAdded && (lostAtTime ?? firstLostTime) != null && (
          <div className="mt-2 flex items-center gap-2 rounded-lg border border-accent/30 bg-accent/10 p-2.5">
            <p className="flex-1 text-xs text-accent">
              {lostAtTime != null ? (
                <>
                  <span className="font-semibold">
                    Tracking lost here ({formatTimestamp(lostAtTime)}) — playback paused.
                  </span>{" "}
                  Click your player below to pin their identity, then re-track.
                </>
              ) : (
                <>
                  <span className="font-semibold">
                    Tracking lost around {formatTimestamp(firstLostTime as number)}.
                  </span>{" "}
                  Jump there and click your player again to pin their identity —
                  then re-track.
                </>
              )}
            </p>
            {lostAtTime == null && (
              <button
                className="btn-accent btn-sm whitespace-nowrap"
                type="button"
                onClick={jumpToLostPoint}
              >
                Go to lost point
              </button>
            )}
          </div>
        )}
        {(() => {
          // Real per-clip stats appear once the clip is tracked. Prefer the
          // freshly computed track's numbers when available on this render.
          const stats = clipStats;
          if (!stats || !focusedTrack) return null;
          const items: { label: string; value: string }[] = [
            { label: "Time tracked", value: `${stats.seconds_tracked ?? "–"}s` },
            { label: "Distance", value: `${stats.distance_heights ?? "–"} body-lengths` },
            { label: "Sprints", value: `${stats.sprint_count ?? 0}` },
          ];
          // Ball-event stats are paused (BALL_EVENTS_ENABLED). Only show them
          // when the backend reports the section is enabled.
          if (stats.ball_events_enabled) {
            items.push(
              { label: "Time with ball", value: `${stats.time_with_ball_seconds ?? 0}s` },
              { label: "Passes", value: `${stats.passes ?? 0}` },
              { label: "Shots", value: `${stats.shots ?? 0}` },
              { label: "Attempts", value: `${stats.shot_attempts ?? 0}` },
            );
          }
          return (
            <div className="mt-3">
              <p className="mb-1.5 text-[11px] font-bold uppercase tracking-wider text-mute">
                Clip stats {stats.ball_data_available === false && "(no ball data — events unavailable)"}
              </p>
              <dl className="grid grid-cols-3 gap-2 text-xs text-mute sm:grid-cols-6">
                {items.map((item) => (
                  <div key={item.label} className="rounded-md bg-surface2/60 p-2">
                    <dt>{item.label}</dt>
                    <dd className="mt-1 font-semibold text-ink">{item.value}</dd>
                  </div>
                ))}
              </dl>
            </div>
          );
        })()}
      </div>
    </article>
  );
}
