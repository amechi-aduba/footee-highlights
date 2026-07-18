import { useEffect, useState } from "react";
import {
  deleteVideoData,
  getProcessingProgress,
  processVideo,
  requestVideoDataDeletion,
  uploadVideo,
  waitForBackend,
} from "./api/client";
import footeeVisionLogo from "./assets/footee-vision-logo.png";
import { DisclaimerModal } from "./components/DisclaimerModal";
import { ProcessingPanel } from "./components/ProcessingPanel";
import { ResultsView } from "./components/ResultsView";
import { VideoUploader } from "./components/VideoUploader";
import type { VideoAnalysisResult, VideoProcessingProgress } from "./types/analysis";

function useTheme() {
  const [dark, setDark] = useState<boolean>(() => {
    const stored = localStorage.getItem("footee-theme");
    if (stored) return stored === "dark";
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? true;
  });
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("footee-theme", dark ? "dark" : "light");
  }, [dark]);
  return { dark, toggle: () => setDark((current) => !current) };
}

function ThemeToggle({ dark, onToggle }: { dark: boolean; onToggle: () => void }) {
  return (
    <button
      type="button"
      aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
      onClick={onToggle}
      className="relative flex h-9 w-16 items-center rounded-full border border-line bg-surface2 px-1 transition-colors duration-300 hover:border-mute"
    >
      <span
        className={`flex h-7 w-7 items-center justify-center rounded-full bg-surface text-sm shadow-card transition-transform duration-300 ${
          dark ? "translate-x-7" : "translate-x-0"
        }`}
      >
        {dark ? "🌙" : "☀️"}
      </span>
    </button>
  );
}

function LogoMark() {
  return (
    <span className="flex h-10 w-10 shrink-0 overflow-hidden rounded-xl shadow-lift">
      <img
        src={footeeVisionLogo}
        alt=""
        aria-hidden="true"
        className="h-full w-full object-cover"
      />
    </span>
  );
}

function App() {
  const { dark, toggle } = useTheme();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoId, setVideoId] = useState("");
  const [result, setResult] = useState<VideoAnalysisResult | null>(null);
  const [error, setError] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] =
    useState<VideoProcessingProgress | null>(null);

  useEffect(() => {
    if (!videoId) return;

    const cleanUpSession = () => requestVideoDataDeletion(videoId);
    window.addEventListener("pagehide", cleanUpSession);
    return () => window.removeEventListener("pagehide", cleanUpSession);
  }, [videoId]);

  useEffect(() => {
    if (!videoId || !isProcessing) return;

    let cancelled = false;
    const refreshProgress = async () => {
      try {
        const nextProgress = await getProcessingProgress(videoId);
        if (!cancelled) setProcessingProgress(nextProgress);
      } catch {
        // The main processing request reports actionable errors. A transient
        // polling failure should not interrupt that request.
      }
    };

    void refreshProgress();
    const intervalId = window.setInterval(refreshProgress, 750);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [isProcessing, videoId]);

  async function handleUpload() {
    if (!selectedFile) return;
    const previousVideoId = videoId;
    setError("");
    setResult(null);
    setVideoId("");
    setProcessingProgress(null);
    setIsUploading(true);
    setUploadStatus("Connecting to the analysis server…");
    try {
      if (previousVideoId) {
        await deleteVideoData(previousVideoId).catch(() => undefined);
      }
      await waitForBackend((attempt, totalAttempts) => {
        setUploadStatus(
          attempt === 1
            ? "Connecting to the analysis server…"
            : `Waking the analysis server… ${attempt}/${totalAttempts}`,
        );
      });
      setUploadStatus("Uploading video…");
      const response = await uploadVideo(selectedFile);
      setVideoId(response.video_id);
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : "Upload failed.");
    } finally {
      setIsUploading(false);
      setUploadStatus("");
    }
  }

  async function handleProcess() {
    setError("");
    setProcessingProgress(null);
    setIsProcessing(true);
    try {
      const processedResult = await processVideo(videoId);
      const finalProgress = await getProcessingProgress(videoId).catch(() => null);
      if (finalProgress) setProcessingProgress(finalProgress);
      setResult(processedResult);
    } catch (processError) {
      const failedProgress = await getProcessingProgress(videoId).catch(() => null);
      if (failedProgress) setProcessingProgress(failedProgress);
      setError(processError instanceof Error ? processError.message : "Processing failed.");
    } finally {
      setIsProcessing(false);
    }
  }

  return (
    <div className="relative min-h-screen overflow-x-clip">
      <DisclaimerModal />
      {/* Ambient glow backdrop */}
      <div aria-hidden className="pointer-events-none fixed inset-0 -z-10">
        <div className="absolute -top-40 left-[8%] h-[420px] w-[420px] rounded-full bg-primary/20 blur-[130px]" />
        <div className="absolute bottom-[-160px] right-[4%] h-[380px] w-[380px] rounded-full bg-accent/15 blur-[130px]" />
      </div>

      <header className="sticky top-0 z-40 border-b border-line/70 bg-app/80 backdrop-blur-md transition-colors duration-300">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-3">
          <div className="flex items-center gap-3">
            <LogoMark />
            <div className="leading-tight">
              <p className="text-sm font-extrabold tracking-tight">
                Footee <span className="text-primary">Vision</span> V1.0
              </p>
              <p className="text-[11px] font-medium text-mute">
                Soccer highlight intelligence
              </p>
            </div>
          </div>
          <ThemeToggle dark={dark} onToggle={toggle} />
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-5 pb-24 pt-10">
        {!result && (
          <section className="animate-fade-up py-8 text-center sm:py-14">
            <p className="kicker">Computer vision for your game</p>
            <h1 className="mx-auto mt-4 max-w-3xl text-4xl font-extrabold tracking-tight sm:text-5xl">
              Turn your highlight reel into a{" "}
              <span className="bg-gradient-to-r from-primary via-sky-400 to-accent bg-clip-text text-transparent">
                player profile
              </span>
            </h1>
            <p className="mx-auto mt-5 max-w-xl text-base text-mute sm:text-lg">
              Upload a reel, pick yourself in each clip, and Footee tracks your
              movement to reveal the archetype you play like.
            </p>
            <div className="mt-8 flex flex-wrap items-center justify-center gap-2 text-xs font-semibold text-mute">
              {["Upload", "Detect", "Track", "Profile"].map((step, index) => (
                <span key={step} className="flex items-center gap-2">
                  <span className="chip border border-line bg-surface text-ink">
                    <span className="text-primary">{index + 1}</span> {step}
                  </span>
                  {index < 3 && <span className="text-line">→</span>}
                </span>
              ))}
            </div>
          </section>
        )}

        <div className="space-y-6">
          <VideoUploader
            selectedFile={selectedFile}
            isUploading={isUploading}
            uploadStatus={uploadStatus}
            onFileChange={setSelectedFile}
            onUpload={handleUpload}
          />
          {videoId && (
            <ProcessingPanel
            videoId={videoId}
            isProcessing={isProcessing}
            progress={processingProgress}
            onProcess={handleProcess}
            />
          )}
          {error && (
            <p className="animate-fade-up rounded-xl border border-red-400/40 bg-red-500/10 p-4 text-sm font-medium text-red-500">
              {error}
            </p>
          )}
          {result && <ResultsView result={result} />}
        </div>
      </main>

      <footer className="border-t border-line/60 py-6 text-center text-xs text-mute">
        Footee Vision — built for players, on players' footage.
      </footer>
    </div>
  );
}

export default App;
