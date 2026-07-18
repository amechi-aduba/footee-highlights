import { useEffect, useRef, useState } from "react";

const DISCLAIMER_KEY = "footee-disclaimer-acknowledged";

const LIMITATIONS = [
  {
    title: "Video clipping is still improving",
    detail: "Clip boundaries may be incomplete, late, or miss parts of a play.",
  },
  {
    title: "Player tracking is experimental",
    detail: "Tracking can lose the selected player or briefly switch to someone else.",
  },
  {
    title: "Player analysis is first-stage",
    detail:
      "Profiles currently use frame-by-frame player movement only, not full tactical context or complete on-ball events.",
  },
];

function hasAcknowledgedDisclaimer() {
  try {
    return sessionStorage.getItem(DISCLAIMER_KEY) === "true";
  } catch {
    return false;
  }
}

export function DisclaimerModal() {
  const [isOpen, setIsOpen] = useState(() => !hasAcknowledgedDisclaimer());
  const acknowledgeButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!isOpen) return;

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    acknowledgeButtonRef.current?.focus();

    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [isOpen]);

  function acknowledge() {
    try {
      sessionStorage.setItem(DISCLAIMER_KEY, "true");
    } catch {
      // The modal can still close when browser storage is unavailable.
    }
    setIsOpen(false);
  }

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-950/75 p-4 backdrop-blur-sm">
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="disclaimer-title"
        aria-describedby="disclaimer-description"
        className="card animate-fade-up relative max-h-[90vh] w-full max-w-xl overflow-y-auto p-6 sm:p-8"
      >
        <span
          aria-hidden
          className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-primary via-sky-400 to-accent"
        />
        <div className="flex items-start gap-4">
          <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-accent/15 text-accent">
            <svg
              viewBox="0 0 24 24"
              className="h-6 w-6"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              aria-hidden="true"
            >
              <path d="M12 3L2.8 19h18.4L12 3z" strokeLinejoin="round" />
              <path d="M12 9v4.5M12 17h.01" strokeLinecap="round" />
            </svg>
          </span>
          <div>
            <p className="kicker">Early-stage preview</p>
            <h2 id="disclaimer-title" className="mt-2 text-2xl font-extrabold tracking-tight">
              Before you analyze your reel
            </h2>
            <p id="disclaimer-description" className="mt-2 text-sm leading-relaxed text-mute">
              Footee Vision is an experimental tool. Please keep these current limitations in
              mind when reviewing its output.
            </p>
          </div>
        </div>

        <ul className="mt-6 space-y-3">
          {LIMITATIONS.map((limitation, index) => (
            <li key={limitation.title} className="flex gap-3 rounded-xl bg-surface2/70 p-4">
              <span className="font-mono text-xs font-bold text-primary">
                {String(index + 1).padStart(2, "0")}
              </span>
              <div>
                <p className="text-sm font-bold text-ink">{limitation.title}</p>
                <p className="mt-1 text-xs leading-relaxed text-mute">{limitation.detail}</p>
              </div>
            </li>
          ))}
        </ul>

        <p className="mt-5 rounded-xl border border-accent/25 bg-accent/10 p-3 text-xs leading-relaxed text-mute">
          Treat the results as an early development aid, not a definitive scouting or
          performance assessment.
        </p>
        <button
          ref={acknowledgeButtonRef}
          type="button"
          className="btn-primary mt-6 w-full justify-center"
          onClick={acknowledge}
        >
          I understand — continue
        </button>
      </section>
    </div>
  );
}
