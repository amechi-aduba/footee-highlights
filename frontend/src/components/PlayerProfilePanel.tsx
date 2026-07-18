import { useState } from "react";
import { generatePlayerProfile, savePlayerInfo } from "../api/client";
import type { Footedness, PlayerInfo, PlayerProfile } from "../types/analysis";
import { SimilarPlayers } from "./SimilarPlayers";

const POSITIONS: { code: string; label: string }[] = [
  { code: "st", label: "ST" },
  { code: "lw", label: "LW" },
  { code: "rw", label: "RW" },
  { code: "10", label: "10 / CAM" },
  { code: "cm", label: "CM" },
  { code: "cdm", label: "CDM" },
  { code: "cb", label: "CB" },
  { code: "rb", label: "RB" },
  { code: "lb", label: "LB" },
];

const FOOT_OPTIONS: { code: Footedness; label: string }[] = [
  { code: "right", label: "Right footed" },
  { code: "left", label: "Left footed" },
  { code: "both", label: "Both feet" },
];

const TRAIT_LABELS: Record<string, string> = {
  involvement: "Involvement",
  work_rate: "Work rate",
  explosiveness: "Explosiveness",
  wideness: "Width",
  passing: "Passing",
  shooting: "Shooting",
};

const CONFIDENCE_CLASSES: Record<string, string> = {
  high: "bg-emerald-500/15 text-emerald-500",
  medium: "bg-accent/15 text-accent",
  low: "bg-surface2 text-mute",
};

interface PlayerProfilePanelProps {
  videoId: string;
  initialInfo: PlayerInfo | null;
  initialProfile: PlayerProfile | null;
  trackedCount: number;
}

export function PlayerProfilePanel({
  videoId,
  initialInfo,
  initialProfile,
  trackedCount,
}: PlayerProfilePanelProps) {
  const [info, setInfo] = useState<PlayerInfo | null>(initialInfo ?? null);
  const [profile, setProfile] = useState<PlayerProfile | null>(initialProfile ?? null);
  const [isEditing, setIsEditing] = useState(initialInfo == null);
  const [draftPositions, setDraftPositions] = useState<string[]>(initialInfo?.positions ?? []);
  const [draftFoot, setDraftFoot] = useState<Footedness>(initialInfo?.footedness ?? "right");
  const [isSaving, setIsSaving] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState("");

  function togglePosition(code: string) {
    setDraftPositions((current) =>
      current.includes(code) ? current.filter((p) => p !== code) : [...current, code],
    );
  }

  async function handleSave() {
    setError("");
    setIsSaving(true);
    try {
      const nextInfo: PlayerInfo = { positions: draftPositions, footedness: draftFoot };
      await savePlayerInfo(videoId, nextInfo);
      setInfo(nextInfo);
      setProfile(null); // info changed — old profile is stale
      setIsEditing(false);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Could not save player info.");
    } finally {
      setIsSaving(false);
    }
  }

  async function handleGenerate() {
    setError("");
    setIsGenerating(true);
    try {
      setProfile(await generatePlayerProfile(videoId));
    } catch (profileError) {
      setError(profileError instanceof Error ? profileError.message : "Profile generation failed.");
    } finally {
      setIsGenerating(false);
    }
  }

  return (
    <div className="card animate-fade-up relative overflow-hidden p-6">
      {/* Accent hairline across the top */}
      <span
        aria-hidden
        className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-primary via-sky-400 to-accent"
      />
      <p className="kicker">Player profile</p>

      {isEditing ? (
        <div className="mt-3 space-y-5">
          <div>
            <h2 className="text-xl font-bold tracking-tight">Tell us about the player</h2>
            <p className="mt-1 text-sm text-mute">
              Pick their main position(s) and stronger foot — the analysis uses this to
              narrow the archetypes it considers.
            </p>
          </div>
          <div>
            <p className="mb-2 text-xs font-bold uppercase tracking-wider text-mute">
              Main positions
            </p>
            <div className="flex flex-wrap gap-2">
              {POSITIONS.map((position) => (
                <button
                  key={position.code}
                  type="button"
                  className={`rounded-lg border px-3.5 py-2 text-xs font-bold transition-all duration-200 active:scale-95 ${
                    draftPositions.includes(position.code)
                      ? "border-primary bg-primary/15 text-primary shadow-card"
                      : "border-line bg-surface text-mute hover:border-primary/50 hover:text-ink"
                  }`}
                  onClick={() => togglePosition(position.code)}
                >
                  {position.label}
                </button>
              ))}
            </div>
          </div>
          <div>
            <p className="mb-2 text-xs font-bold uppercase tracking-wider text-mute">
              Stronger foot
            </p>
            <div className="flex flex-wrap gap-2">
              {FOOT_OPTIONS.map((option) => (
                <button
                  key={option.code}
                  type="button"
                  className={`rounded-lg border px-3.5 py-2 text-xs font-bold transition-all duration-200 active:scale-95 ${
                    draftFoot === option.code
                      ? "border-accent bg-accent/15 text-accent shadow-card"
                      : "border-line bg-surface text-mute hover:border-accent/50 hover:text-ink"
                  }`}
                  onClick={() => setDraftFoot(option.code)}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
          <button
            type="button"
            className="btn-primary"
            disabled={draftPositions.length === 0 || isSaving}
            onClick={handleSave}
          >
            {isSaving && <span className="spinner" />}
            {isSaving ? "Saving…" : "Save player info"}
          </button>
        </div>
      ) : profile ? (
        <div className="mt-3 grid gap-6 lg:grid-cols-[minmax(0,1fr)_19rem] lg:items-start">
          <div className="min-w-0 space-y-5">
          <div className="flex flex-wrap items-center gap-3">
            <h2 className="text-3xl font-extrabold tracking-tight">
              {profile.primary_label}
            </h2>
            <span className="chip bg-primary/10 text-primary">{profile.group}</span>
            <span className={`chip ${CONFIDENCE_CLASSES[profile.confidence]}`}>
              {profile.confidence} confidence
            </span>
          </div>
          <p className="max-w-2xl text-sm text-mute">{profile.description}</p>

          {profile.features.ball_events_enabled && profile.features.ball_data_available && (
            <div className="flex flex-wrap gap-2">
              <span className="chip border border-line bg-surface text-mute">
                Time with ball&nbsp;<span className="font-bold text-ink">{profile.features.time_with_ball_seconds ?? 0}s</span>
              </span>
              <span className="chip border border-line bg-surface text-mute">
                Passes&nbsp;<span className="font-bold text-ink">{profile.features.passes ?? 0}</span>
              </span>
              <span className="chip border border-line bg-surface text-mute">
                Shots&nbsp;<span className="font-bold text-ink">{profile.features.shots ?? 0}</span>
              </span>
              <span className="chip border border-line bg-surface text-mute">
                Attempts&nbsp;<span className="font-bold text-ink">{profile.features.shot_attempts ?? 0}</span>
              </span>
              <span className="chip bg-surface2 text-mute">shots = on target to keeper; attempts &amp; more from ball trajectory (estimates)</span>
            </div>
          )}

          <div className="grid gap-6 sm:grid-cols-2">
            <div>
              <p className="mb-3 text-xs font-bold uppercase tracking-wider text-mute">
                Measured traits
              </p>
              <div className="space-y-3">
                {Object.entries(profile.traits).map(([trait, value], index) => (
                  <div key={trait}>
                    <div className="flex justify-between text-xs font-semibold">
                      <span className="text-ink">{TRAIT_LABELS[trait] ?? trait}</span>
                      <span className="text-mute">{Math.round(value * 100)}</span>
                    </div>
                    <div className="mt-1 h-2 overflow-hidden rounded-full bg-surface2">
                      <div
                        className="bar-grow h-full animate-grow-x rounded-full bg-gradient-to-r from-primary to-sky-400"
                        style={{
                          width: `${Math.max(4, Math.round(value * 100))}%`,
                          animationDelay: `${index * 120}ms`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <p className="mb-3 text-xs font-bold uppercase tracking-wider text-mute">
                Archetype fit
              </p>
              <ul className="space-y-2">
                {profile.scores.slice(0, 4).map((entry) => (
                  <li key={entry.archetype} className="flex items-center justify-between text-xs">
                    <span
                      className={
                        entry.archetype === profile.primary_archetype
                          ? "font-bold text-ink"
                          : "font-medium text-mute"
                      }
                    >
                      {entry.archetype === profile.primary_archetype && (
                        <span className="mr-1 text-accent">★</span>
                      )}
                      {entry.label}
                    </span>
                    <span className="font-mono font-semibold text-mute">
                      {Math.round(entry.score * 100)}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="rounded-xl border border-line bg-surface2/60 p-4">
            <p className="mb-2 text-xs font-bold uppercase tracking-wider text-mute">Why</p>
            <ul className="space-y-1.5 text-xs text-mute">
              {profile.evidence.map((line) => (
                <li key={line} className="flex gap-2">
                  <span className="text-primary">›</span>
                  {line}
                </li>
              ))}
            </ul>
          </div>

          <p className="text-[11px] leading-relaxed text-mute/80">
            Based on {profile.features.minutes_tracked} tracked minutes across{" "}
            {profile.features.segments_analyzed} clip
            {profile.features.segments_analyzed === 1 ? "" : "s"}. {profile.note}
          </p>
          <div className="flex gap-2">
            <button
              type="button"
              className="btn-primary btn-sm"
              disabled={isGenerating}
              onClick={handleGenerate}
            >
              {isGenerating && <span className="spinner" />}
              {isGenerating ? "Analyzing…" : "Regenerate"}
            </button>
            <button type="button" className="btn-ghost btn-sm" onClick={() => setIsEditing(true)}>
              Edit player info
            </button>
          </div>
          </div>
          <SimilarPlayers
            archetype={profile.primary_archetype}
            footedness={info?.footedness ?? profile.footedness}
          />
        </div>
      ) : (
        <div className="mt-3 space-y-4">
          <h2 className="text-xl font-bold tracking-tight">Ready to analyze</h2>
          <div className="flex flex-wrap gap-1.5">
            {info?.positions.map((position) => (
              <span key={position} className="chip bg-primary/10 text-primary">
                {position.toUpperCase()}
              </span>
            ))}
            <span className="chip bg-accent/15 text-accent">
              {FOOT_OPTIONS.find((option) => option.code === info?.footedness)?.label}
            </span>
          </div>
          {trackedCount === 0 ? (
            <p className="rounded-lg border border-accent/30 bg-accent/10 p-3 text-xs font-medium text-accent">
              Track your player in at least one clip below, then generate the profile.
              More tracked clips = a sharper profile.
            </p>
          ) : (
            <p className="text-xs font-semibold text-mute">
              {trackedCount} tracked clip{trackedCount === 1 ? "" : "s"} ready for analysis.
            </p>
          )}
          <div className="flex gap-2">
            <button
              type="button"
              className="btn-primary"
              disabled={trackedCount === 0 || isGenerating}
              onClick={handleGenerate}
            >
              {isGenerating && <span className="spinner" />}
              {isGenerating ? "Analyzing…" : "Generate player profile"}
            </button>
            <button type="button" className="btn-ghost" onClick={() => setIsEditing(true)}>
              Edit player info
            </button>
          </div>
        </div>
      )}
      {error && <p className="mt-4 text-xs font-medium text-red-500">{error}</p>}
    </div>
  );
}
