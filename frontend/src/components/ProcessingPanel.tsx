interface ProcessingPanelProps {
  videoId: string;
  isProcessing: boolean;
  onProcess: () => void;
}

const PIPELINE_STEPS = [
  "Scene cuts (TransNetV2)",
  "Cutaway filtering",
  "Thumbnails",
];

export function ProcessingPanel({
  videoId,
  isProcessing,
  onProcess,
}: ProcessingPanelProps) {
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
        Video <span className="font-mono text-ink/80">{videoId.slice(0, 12)}…</span> ready.
      </p>

      {isProcessing ? (
        <div className="mt-5 space-y-3">
          <div className="flex items-center gap-3 text-sm font-semibold text-ink">
            <span className="spinner text-primary" />
            Analyzing your reel — this runs locally and can take a couple of minutes.
          </div>
          <div className="flex flex-wrap gap-2">
            {PIPELINE_STEPS.map((step, index) => (
              <span
                key={step}
                className="chip animate-fade-up border border-line bg-surface2 text-mute"
                style={{ animationDelay: `${index * 140}ms` }}
              >
                {step}
              </span>
            ))}
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface2">
            <div className="h-full w-1/3 animate-shimmer rounded-full bg-gradient-to-r from-transparent via-primary to-transparent bg-[length:200%_100%]" />
          </div>
        </div>
      ) : (
        <button className="btn-primary mt-5" type="button" onClick={onProcess}>
          Process video
        </button>
      )}
    </section>
  );
}
