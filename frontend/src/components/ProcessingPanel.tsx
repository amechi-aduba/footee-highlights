import type {
  ProcessingStageProgress,
  ProcessingStageStatus,
  VideoProcessingProgress,
} from "../types/analysis";

interface ProcessingPanelProps {
  videoId: string;
  isProcessing: boolean;
  progress: VideoProcessingProgress | null;
  onProcess: () => void;
}

const PIPELINE_STEPS: Array<Pick<ProcessingStageProgress, "key" | "label">> = [
  { key: "scene_cuts", label: "Scene cuts (TransNetV2)" },
  { key: "cutaway_filtering", label: "Cutaway filtering" },
  { key: "thumbnails", label: "Thumbnails" },
];

function fallbackStages(isProcessing: boolean): ProcessingStageProgress[] {
  return PIPELINE_STEPS.map((step, index) => ({
    ...step,
    status: isProcessing && index === 0 ? "active" : "pending",
    progress_percent: 0,
    completed_items: 0,
    total_items: null,
  }));
}

function stageClasses(status: ProcessingStageStatus): string {
  if (status === "complete") {
    return "border-emerald-400/50 bg-emerald-500/10 text-emerald-500";
  }
  if (status === "active") {
    return "border-primary/50 bg-primary/10 text-primary";
  }
  if (status === "failed") {
    return "border-red-400/50 bg-red-500/10 text-red-500";
  }
  return "border-line bg-surface2 text-mute";
}

function StageIcon({ status }: { status: ProcessingStageStatus }) {
  if (status === "complete") {
    return (
      <span className="flex h-4 w-4 items-center justify-center rounded-full bg-emerald-500 text-[10px] font-black text-white">
        ✓
      </span>
    );
  }
  if (status === "active") return <span className="spinner" />;
  if (status === "failed") {
    return (
      <span className="flex h-4 w-4 items-center justify-center rounded-full bg-red-500 text-[10px] font-black text-white">
        !
      </span>
    );
  }
  return <span className="h-3.5 w-3.5 rounded border border-current opacity-50" />;
}

export function ProcessingPanel({
  videoId,
  isProcessing,
  progress,
  onProcess,
}: ProcessingPanelProps) {
  const stages = progress?.stages ?? fallbackStages(isProcessing);
  const percent = Math.round(
    Math.max(0, Math.min(100, progress?.progress_percent ?? (isProcessing ? 1 : 0))),
  );
  const isComplete = progress?.status === "complete";
  const hasFailed = progress?.status === "failed";
  const showProgress = isProcessing || progress !== null;

  return (
    <section className="card animate-fade-up p-6">
      <div className="flex items-center gap-3">
        <span className="chip bg-primary/10 text-primary">Step 2</span>
        <h2 className="text-lg font-bold tracking-tight">Split the reel into clips</h2>
      </div>
      <p className="mt-2 text-sm text-mute">
        Footee finds every cut, drops intros and cutaways, and preps each clip for
        player tracking.
      </p>
      <p className="mt-1 break-all text-xs text-mute">
        Video <span className="font-mono text-ink/80">{videoId.slice(0, 12)}...</span> ready.
      </p>

      {showProgress ? (
        <div className="mt-5 space-y-4" aria-live="polite">
          <div className="flex items-center justify-between gap-4 text-sm font-semibold">
            <span
              className={
                hasFailed
                  ? "text-red-500"
                  : isComplete
                    ? "text-emerald-500"
                    : "text-ink"
              }
            >
              {isProcessing && !isComplete && !hasFailed && (
                <span className="spinner mr-2 inline-block align-[-0.1em] text-primary" />
              )}
              {progress?.message ?? "Starting reel analysis"}
            </span>
            <span className="shrink-0 tabular-nums text-mute">{percent}%</span>
          </div>

          <div className="flex flex-wrap gap-2">
            {stages.map((stage) => (
              <span
                key={stage.key}
                className={`chip border px-3 py-1.5 transition-colors duration-300 ${stageClasses(stage.status)}`}
              >
                <StageIcon status={stage.status} />
                <span>{stage.label}</span>
                {stage.status === "active" &&
                  stage.total_items != null &&
                  stage.total_items > 0 && (
                    <span className="ml-1 tabular-nums opacity-80">
                      {stage.completed_items}/{stage.total_items}
                    </span>
                  )}
              </span>
            ))}
          </div>

          <div
            className="h-2.5 w-full overflow-hidden rounded-full bg-surface2"
            role="progressbar"
            aria-label="Reel splitting progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={percent}
          >
            <div
              className={`h-full rounded-full transition-[width] duration-500 ease-out ${
                hasFailed
                  ? "bg-red-500"
                  : isComplete
                    ? "bg-emerald-500"
                    : "bg-gradient-to-r from-primary to-accent"
              }`}
              style={{ width: `${percent}%` }}
            />
          </div>

          {hasFailed && (
            <button className="btn-primary" type="button" onClick={onProcess}>
              Retry processing
            </button>
          )}
        </div>
      ) : (
        <button className="btn-primary mt-5" type="button" onClick={onProcess}>
          Process video
        </button>
      )}
    </section>
  );
}
