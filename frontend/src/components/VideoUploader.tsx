import { useRef, useState } from "react";

interface VideoUploaderProps {
  selectedFile: File | null;
  isUploading: boolean;
  uploadStatus?: string;
  onFileChange: (file: File | null) => void;
  onUpload: () => void;
}

function formatBytes(bytes: number) {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  return `${Math.round(bytes / 1e3)} KB`;
}

export function VideoUploader({
  selectedFile,
  isUploading,
  uploadStatus,
  onFileChange,
  onUpload,
}: VideoUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  function handleDrop(event: React.DragEvent) {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file && file.type.startsWith("video/")) onFileChange(file);
  }

  return (
    <section className="card animate-fade-up p-6">
      <div className="flex items-center gap-3">
        <span className="chip bg-primary/10 text-primary">Step 1</span>
        <h2 className="text-lg font-bold tracking-tight">Upload your highlight reel</h2>
      </div>

      <button
        type="button"
        className={`mt-5 flex w-full flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-10 transition-all duration-300 ${
          isDragging
            ? "scale-[1.01] border-primary bg-primary/10"
            : selectedFile
              ? "border-primary/50 bg-primary/5"
              : "border-line bg-surface2/60 hover:border-primary/60 hover:bg-primary/5"
        }`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        <span
          className={`flex h-12 w-12 items-center justify-center rounded-full bg-primary/15 text-primary transition-transform duration-300 ${
            isDragging ? "scale-110" : ""
          }`}
        >
          <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
            <path d="M12 16V5m0 0l-4 4m4-4l4 4" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M4 15v3a2 2 0 002 2h12a2 2 0 002-2v-3" strokeLinecap="round" />
          </svg>
        </span>
        {selectedFile ? (
          <span className="flex flex-wrap items-center justify-center gap-2 text-sm">
            <span className="max-w-[280px] truncate font-semibold text-ink">
              {selectedFile.name}
            </span>
            <span className="chip bg-surface text-mute">{formatBytes(selectedFile.size)}</span>
            <span className="text-xs text-mute">— click to swap</span>
          </span>
        ) : (
          <span className="text-sm text-mute">
            <span className="font-semibold text-ink">Drop a video here</span> or click to
            browse · MP4, MOV, MKV…
          </span>
        )}
      </button>
      <input
        ref={inputRef}
        className="hidden"
        type="file"
        accept="video/*"
        onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
      />

      <div className="mt-5 flex items-center gap-3">
        <button
          className="btn-primary"
          type="button"
          disabled={!selectedFile || isUploading}
          onClick={onUpload}
        >
          {isUploading && <span className="spinner" />}
          {isUploading ? uploadStatus || "Uploading…" : "Upload video"}
        </button>
        <p className="text-xs text-mute">
          Temporarily uploaded for analysis, then automatically deleted.
        </p>
      </div>
    </section>
  );
}
