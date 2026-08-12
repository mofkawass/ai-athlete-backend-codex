import { fetch } from 'expo/fetch';
import { File } from 'expo-file-system';

export type AnalysisStatus =
  | 'idle'
  | 'requesting_upload'
  | 'uploading'
  | 'creating_job'
  | 'analyzing'
  | 'done'
  | 'error';

export type JobResult = {
  sport?: string;
  summary?: string;
  drills?: string[];
  overlay_url?: string;
  analysis?: {
    metrics?: Record<string, number | null>;
    recommendations?: string[];
  };
  focus?: string;
  focus_tips?: Array<{ id: number; text: string }>;
};

const API_BASE = process.env.EXPO_PUBLIC_API_BASE_URL?.replace(/\/$/, '');

function apiBase(): string {
  if (!API_BASE) {
    throw new Error('EXPO_PUBLIC_API_BASE_URL is not configured.');
  }
  return API_BASE;
}

export async function healthCheck(): Promise<boolean> {
  const response = await fetch(`${apiBase()}/health`);
  if (!response.ok) return false;
  const data = await response.json();
  return data?.ok === true;
}

export async function uploadAndAnalyze(
  videoUri: string,
  options: {
    sport?: string | null;
    focus?: string | null;
    onStatus?: (status: AnalysisStatus) => void;
  } = {},
): Promise<JobResult> {
  const { sport = null, focus = null, onStatus } = options;
  const setStatus = (status: AnalysisStatus) => onStatus?.(status);

  try {
    setStatus('requesting_upload');
    const objectName = `${Date.now()}.mp4`;
    const signed = await fetch(
      `${apiBase()}/signed-upload?name=${encodeURIComponent(objectName)}&contentType=${encodeURIComponent('video/mp4')}`,
    );
    if (!signed.ok) throw new Error(`Could not prepare upload (${signed.status}).`);
    const { url, objectPath } = await signed.json();

    setStatus('uploading');
    const file = new File(videoUri);
    const put = await fetch(url, {
      method: 'PUT',
      headers: { 'Content-Type': 'video/mp4' },
      body: file,
    });
    if (!put.ok) throw new Error(`Video upload failed (${put.status}).`);

    setStatus('creating_job');
    const payload: Record<string, unknown> = { objectPath };
    if (sport) payload.sport = sport;
    if (focus) payload.focus = focus;

    const create = await fetch(`${apiBase()}/jobs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!create.ok) throw new Error(`Could not create analysis job (${create.status}).`);
    const { id } = await create.json();

    setStatus('analyzing');
    const started = Date.now();
    const timeoutMs = 180_000;

    while (Date.now() - started < timeoutMs) {
      await new Promise((resolve) => setTimeout(resolve, 1400));
      const response = await fetch(`${apiBase()}/status/${id}`);
      if (!response.ok) throw new Error(`Could not read job status (${response.status}).`);
      const job = await response.json();

      if (job.status === 'DONE') {
        setStatus('done');
        return job.result as JobResult;
      }
      if (job.status === 'ERROR') {
        throw new Error(job.result?.error || 'Analysis failed.');
      }
    }

    throw new Error('Analysis took too long. Please try again.');
  } catch (error) {
    setStatus('error');
    throw error;
  }
}
