import type {
  BoundingBox,
  FocusedPlayerTrack,
  FrameDetectionResponse,
  ObjectDetectionSummary,
  PlayerInfo,
  PlayerProfile,
  VideoAnalysisResult,
  VideoProcessingProgress,
  VideoUploadResponse,
} from "../types/analysis";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const BACKEND_WAKE_ATTEMPTS = 12;
const BACKEND_WAKE_DELAY_MS = 5_000;
const BACKEND_HEALTH_TIMEOUT_MS = 10_000;
const PROCESSING_POLL_DELAY_MS = 1_000;

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

export async function waitForBackend(
  onAttempt?: (attempt: number, totalAttempts: number) => void,
): Promise<void> {
  for (let attempt = 1; attempt <= BACKEND_WAKE_ATTEMPTS; attempt += 1) {
    onAttempt?.(attempt, BACKEND_WAKE_ATTEMPTS);
    const controller = new AbortController();
    const timeoutId = window.setTimeout(
      () => controller.abort(),
      BACKEND_HEALTH_TIMEOUT_MS,
    );
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        cache: "no-store",
        signal: controller.signal,
      });
      if (response.ok) return;
    } catch {
      // A scale-to-zero backend can refuse requests while its container starts.
      // The health probe itself triggers the wake-up, so retry before upload.
    } finally {
      window.clearTimeout(timeoutId);
    }
    if (attempt < BACKEND_WAKE_ATTEMPTS) {
      await delay(BACKEND_WAKE_DELAY_MS);
    }
  }
  throw new Error(
    "The free analysis server did not wake up in time. Please wait a moment and retry.",
  );
}

async function readResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? "Something went wrong. Please try again.");
  }
  return response.json() as Promise<T>;
}

export async function uploadVideo(file: File): Promise<VideoUploadResponse> {
  const formData = new FormData();
  formData.append("video", file);
  return readResponse(
    await fetch(`${API_BASE_URL}/api/videos/upload`, {
      method: "POST",
      body: formData,
    }),
  );
}

export async function processVideo(videoId: string): Promise<VideoAnalysisResult> {
  await readResponse(
    await fetch(`${API_BASE_URL}/api/videos/${videoId}/process`, {
      method: "POST",
    }),
  );

  // Processing can exceed Azure Container Apps' per-request timeout. The POST
  // starts the task; these short requests also keep a scale-to-zero replica
  // active until the result has been persisted.
  while (true) {
    const progress = await getProcessingProgress(videoId);
    if (progress.status === "complete") {
      return getVideoResult(videoId);
    }
    if (progress.status === "failed") {
      throw new Error(progress.message || "Processing failed. Please retry.");
    }
    await delay(PROCESSING_POLL_DELAY_MS);
  }
}

export async function getProcessingProgress(
  videoId: string,
): Promise<VideoProcessingProgress> {
  return readResponse(
    await fetch(`${API_BASE_URL}/api/videos/${videoId}/processing-progress`, {
      cache: "no-store",
    }),
  );
}

export async function getVideoResult(videoId: string): Promise<VideoAnalysisResult> {
  return readResponse(
    await fetch(`${API_BASE_URL}/api/videos/${videoId}/result`, {
      cache: "no-store",
    }),
  );
}

export async function deleteVideoData(videoId: string): Promise<void> {
  await readResponse(
    await fetch(`${API_BASE_URL}/api/videos/${videoId}`, {
      method: "DELETE",
    }),
  );
}

export function requestVideoDataDeletion(videoId: string): void {
  const cleanupUrl = `${API_BASE_URL}/api/videos/${videoId}/cleanup`;
  try {
    if (navigator.sendBeacon?.(cleanupUrl)) return;
  } catch {
    // Fall through to a keepalive request when Beacon is unavailable or blocked.
  }

  void fetch(cleanupUrl, {
    method: "POST",
    keepalive: true,
  }).catch(() => {
    // The backend's retention policy is the fallback if page-exit cleanup fails.
  });
}

export async function detectSegmentFrame(
  videoId: string,
  segmentId: string,
  clipTimeSeconds: number,
): Promise<FrameDetectionResponse> {
  return readResponse(
    await fetch(`${API_BASE_URL}/api/videos/${videoId}/segment/${segmentId}/detect-frame`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clip_time_seconds: clipTimeSeconds }),
    }),
  );
}

export async function analyzeSegmentObjects(
  videoId: string,
  segmentId: string,
  clipTimeSeconds: number,
): Promise<ObjectDetectionSummary> {
  return readResponse(
    await fetch(`${API_BASE_URL}/api/videos/${videoId}/segment/${segmentId}/detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clip_time_seconds: clipTimeSeconds }),
    }),
  );
}

export async function selectFocusedPlayer(
  videoId: string,
  segmentId: string,
  selection: {
    detection_id: string;
    clip_time_seconds: number;
    bbox: BoundingBox;
    confidence: number;
    team_id?: string | null;
    jersey_color_hex?: string | null;
    jersey_descriptor?: number[] | null;
    /** true = add an anchor for the same player where tracking was lost. */
    additive?: boolean;
  },
): Promise<void> {
  await readResponse(
    await fetch(`${API_BASE_URL}/api/videos/${videoId}/segment/${segmentId}/focused-player`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(selection),
    }),
  );
}

export async function resetFocusedPlayer(videoId: string, segmentId: string): Promise<void> {
  await readResponse(
    await fetch(`${API_BASE_URL}/api/videos/${videoId}/segment/${segmentId}/focused-player`, {
      method: "DELETE",
    }),
  );
}

export async function savePlayerInfo(
  videoId: string,
  info: PlayerInfo,
): Promise<void> {
  await readResponse(
    await fetch(`${API_BASE_URL}/api/videos/${videoId}/player-info`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(info),
    }),
  );
}

export async function generatePlayerProfile(videoId: string): Promise<PlayerProfile> {
  return readResponse(
    await fetch(`${API_BASE_URL}/api/videos/${videoId}/player-profile`, {
      method: "POST",
    }),
  );
}

export async function trackFocusedPlayer(
  videoId: string,
  segmentId: string,
): Promise<FocusedPlayerTrack> {
  return readResponse(
    await fetch(`${API_BASE_URL}/api/videos/${videoId}/segment/${segmentId}/track-focused-player`, {
      method: "POST",
    }),
  );
}
