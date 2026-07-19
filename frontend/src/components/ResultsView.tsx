import { useState } from "react";
import { API_BASE_URL } from "../api/client";
import type { VideoAnalysisResult } from "../types/analysis";
import { clipLabel } from "../utils/video";
import { PlayerProfilePanel } from "./PlayerProfilePanel";
import { SegmentCard } from "./SegmentCard";

function formatTimestamp(totalSeconds: number) {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, "0")}`;
}

const STATUS_BADGES: Record<string, { label: string; classes: string }> = {
  demo: { label: "Preprocessed", classes: "bg-emerald-500/15 text-emerald-500" },
  not_selected: { label: "No player selected", classes: "bg-surface2 text-mute" },
  selected: { label: "Player selected", classes: "bg-accent/15 text-accent" },
  tracked: { label: "Tracked ✓", classes: "bg-emerald-500/15 text-emerald-500" },
};

export function ResultsView({ result }: { result: VideoAnalysisResult }) {
  const { metadata } = result;
  const demo = result.demo ?? null;
  // Only ONE segment is ever "open" — so only one <video> element exists at a
  // time. Mounting a player per card made every card decode the full source
  // video in parallel, which is what froze playback.
  const [activeSegmentId, setActiveSegmentId] = useState<string | null>(null);
  const [statusOverrides, setStatusOverrides] = useState<Record<string, string>>({});
  const [showCutaways, setShowCutaways] = useState(false);

  const activeSegment =
    result.segments.find((segment) => segment.segment_id === activeSegmentId) ?? null;

  function statusFor(segmentId: string, fallback: string) {
    return statusOverrides[segmentId] ?? fallback;
  }

  const trackedCount = result.segments.filter(
    (segment) => statusFor(segment.segment_id, segment.focused_player_status) === "tracked",
  ).length;

  const gameplaySegments = result.segments.filter((segment) => segment.kind !== "cutaway");
  const cutawaySegments = result.segments.filter((segment) => segment.kind === "cutaway");

  function resolveThumbnail(path: string) {
    return demo ? path : `${API_BASE_URL}${path}`;
  }

  const metadataChips = [
    { label: "Duration", value: `${metadata.duration_seconds}s` },
    { label: "FPS", value: `${metadata.fps}` },
    { label: "Frames", value: `${metadata.frame_count}` },
    { label: "Resolution", value: `${metadata.width}×${metadata.height}` },
  ];

  function renderSegmentButton(
    segment: VideoAnalysisResult["segments"][number],
    index: number,
  ) {
    const status = statusFor(segment.segment_id, segment.focused_player_status);
    const badge = STATUS_BADGES[status] ?? STATUS_BADGES.not_selected;
    const isActive = segment.segment_id === activeSegmentId;
    return (
      <button
        key={segment.segment_id}
        type="button"
        style={{ animationDelay: `${Math.min(index, 8) * 60}ms` }}
        className={`group animate-fade-up overflow-hidden rounded-xl border text-left transition-all duration-300 hover:-translate-y-1 hover:shadow-lift ${
          isActive
            ? "border-primary ring-2 ring-primary/40"
            : "border-line hover:border-primary/50"
        } bg-surface`}
        onClick={() => setActiveSegmentId(isActive ? null : segment.segment_id)}
      >
        <div className="relative overflow-hidden">
          <img
            src={resolveThumbnail(segment.thumbnail_path)}
            alt={`Preview of ${segment.segment_id}`}
            className="aspect-video w-full object-cover transition-transform duration-500 group-hover:scale-[1.05]"
            loading="lazy"
          />
          <span className="absolute inset-0 flex items-center justify-center bg-ink/0 transition-colors duration-300 group-hover:bg-ink/20">
            <span className="flex h-11 w-11 scale-75 items-center justify-center rounded-full bg-surface/90 text-primary opacity-0 shadow-card transition-all duration-300 group-hover:scale-100 group-hover:opacity-100">
              <svg viewBox="0 0 24 24" className="ml-0.5 h-5 w-5" fill="currentColor" aria-hidden>
                <path d="M8 5.5v13l11-6.5-11-6.5z" />
              </svg>
            </span>
          </span>
        </div>
        <div className="space-y-1.5 p-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-bold tracking-tight">{clipLabel(segment.segment_id)}</span>
            <span className="text-xs text-mute">
              {formatTimestamp(segment.start_time)} – {formatTimestamp(segment.end_time)}
            </span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            <span className={`chip ${badge.classes}`}>{badge.label}</span>
            {segment.kind === "cutaway" && (
              <span className="chip bg-surface2 text-mute">cutaway</span>
            )}
          </div>
        </div>
      </button>
    );
  }

  return (
    <section className="space-y-8">
      {demo ? (
        <div className="animate-fade-up rounded-2xl border border-emerald-500/30 bg-emerald-500/10 p-5">
          <div className="flex flex-wrap items-center gap-2">
            <span className="chip bg-emerald-500/20 text-emerald-500">Static demo</span>
            <h2 className="text-lg font-bold tracking-tight">{demo.title}</h2>
          </div>
          <p className="mt-2 text-sm leading-relaxed text-mute">{demo.description}</p>
          <p className="mt-2 text-xs font-semibold text-emerald-600 dark:text-emerald-400">
            Clip playback and thumbnails come from Vercel. This view does not wake Azure or
            run detection, tracking, or profile inference.
          </p>
        </div>
      ) : (
        <PlayerProfilePanel
          videoId={result.video_id}
          initialInfo={result.player_info ?? null}
          initialProfile={result.player_profile ?? null}
          trackedCount={trackedCount}
        />
      )}

      <div className="animate-fade-up flex flex-wrap items-center gap-2">
        {metadataChips.map((chip) => (
          <span key={chip.label} className="chip border border-line bg-surface text-mute">
            {chip.label}&nbsp;<span className="font-bold text-ink">{chip.value}</span>
          </span>
        ))}
      </div>

      {activeSegment && (
        <div className="animate-fade-up">
          <div className="mb-3 flex items-center gap-3">
            <p className="kicker">Now reviewing</p>
            <span className="text-sm font-bold">{clipLabel(activeSegment.segment_id)}</span>
          </div>
          <SegmentCard
            key={activeSegment.segment_id}
            segment={activeSegment}
            videoId={result.video_id}
            videoUrl={demo?.video_path}
            thumbnailUrl={resolveThumbnail(activeSegment.thumbnail_path)}
            readOnly={demo?.read_only ?? false}
            onClose={() => setActiveSegmentId(null)}
            onStatusChange={(status) =>
              setStatusOverrides((current) => ({
                ...current,
                [activeSegment.segment_id]: status,
              }))
            }
          />
        </div>
      )}

      <div>
        <div className="mb-1 flex items-baseline justify-between">
          <h2 className="text-lg font-bold tracking-tight">
            {demo ? "Preprocessed gameplay clips" : "Gameplay clips"}{" "}
            <span className="text-sm font-semibold text-mute">
              ({gameplaySegments.length})
            </span>
          </h2>
          <span className="text-xs font-semibold text-mute">
            {demo ? "Zero analysis compute" : `${trackedCount}/${gameplaySegments.length} tracked`}
          </span>
        </div>
        <p className="mb-4 text-sm text-mute">
          {demo
            ? "Open a clip to play or scrub the cached sample. Upload your own reel for live player detection and tracking."
            : "Open a clip to detect players, pick yourself, and verify the tracking."}
        </p>
        <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
          {gameplaySegments.map(renderSegmentButton)}
        </div>
      </div>

      {cutawaySegments.length > 0 && (
        <div>
          <button
            type="button"
            className="text-sm font-semibold text-mute transition-colors hover:text-ink"
            onClick={() => setShowCutaways((current) => !current)}
          >
            <span
              className={`mr-1 inline-block transition-transform duration-200 ${showCutaways ? "rotate-90" : ""}`}
            >
              ▸
            </span>
            Skipped clips ({cutawaySegments.length}) — intros, title cards, celebrations
          </button>
          {showCutaways && (
            <div className="mt-4 grid gap-4 opacity-80 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
              {cutawaySegments.map(renderSegmentButton)}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
