import type { DemoSample } from "../data/demoSamples";

interface DemoSampleGalleryProps {
  samples: readonly DemoSample[];
  activeSampleId?: string | null;
  loadingSampleId?: string | null;
  onSelect: (sample: DemoSample) => void;
}

export function DemoSampleGallery({
  samples,
  activeSampleId,
  loadingSampleId,
  onSelect,
}: DemoSampleGalleryProps) {
  return (
    <section className="card animate-fade-up p-6">
      <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-end">
        <div>
          <div className="flex items-center gap-3">
            <span className="chip bg-emerald-500/15 text-emerald-500">Instant demos</span>
            <h2 className="text-lg font-bold tracking-tight">No reel ready?</h2>
          </div>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-mute">
            Open a preprocessed reel to explore scene splitting, clip review, and a featured
            player-tracking showcase. The saved tracks are served as static files, so the demos
            do not upload anything or run the analysis backend.
          </p>
        </div>
        <span className="whitespace-nowrap text-xs font-semibold text-mute">
          No Azure inference
        </span>
      </div>

      <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {samples.map((sample) => {
          const isActive = activeSampleId === sample.id;
          const isLoading = loadingSampleId === sample.id;
          return (
            <button
              key={sample.id}
              type="button"
              className={`group overflow-hidden rounded-xl border bg-surface text-left transition-all duration-300 hover:-translate-y-1 hover:shadow-lift ${
                isActive
                  ? "border-primary ring-2 ring-primary/30"
                  : "border-line hover:border-primary/50"
              }`}
              disabled={loadingSampleId != null}
              aria-pressed={isActive}
              onClick={() => onSelect(sample)}
            >
              <div className="relative aspect-video overflow-hidden bg-slate-900">
                <img
                  src={sample.posterPath}
                  alt={`Preview of ${sample.title}'s sample reel`}
                  className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105"
                  loading="lazy"
                />
                <span className="absolute bottom-2 right-2 rounded-md bg-slate-950/80 px-2 py-1 font-mono text-[10px] font-bold text-white">
                  {sample.duration}
                </span>
              </div>
              <div className="p-3">
                <p className="text-sm font-bold text-ink">{sample.title}</p>
                <p className="mt-1 min-h-8 text-xs leading-relaxed text-mute">
                  {sample.subtitle}
                </p>
                <span className="mt-3 inline-flex items-center gap-2 text-xs font-bold text-primary">
                  {isLoading && <span className="spinner" />}
                  {isLoading ? "Loading demo…" : isActive ? "Demo open" : "Open demo"}
                </span>
                <span className="ml-2 mt-3 inline-flex text-[10px] font-bold uppercase tracking-wide text-emerald-500">
                  Tracking included
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}
