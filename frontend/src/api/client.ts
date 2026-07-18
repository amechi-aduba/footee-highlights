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
  return readResponse(
    await fetch(`${API_BASE_URL}/api/videos/${videoId}/process`, {
      method: "POST",
    }),
  );
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
