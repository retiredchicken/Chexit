const defaultBase = 'http://127.0.0.1:8000';

export type PredictResponse = {
  diagnosis: string;
  risk_score: number;
  confidence_label: string;
  heatmap: string;
};

export type PredictUiState = {
  loading: boolean;
  error: string | null;
  data: PredictResponse | null;
};

function apiBase(): string {
  const base = import.meta.env.VITE_CHEXIT_API_URL?.trim();
  return base && base.length > 0 ? base.replace(/\/$/, '') : defaultBase;
}

function parseErrorDetail(body: unknown): string {
  if (!body || typeof body !== 'object') return 'Request failed';
  const d = (body as { detail?: unknown }).detail;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) {
    return d
      .map((e) => (typeof e === 'object' && e && 'msg' in e ? String((e as { msg: string }).msg) : String(e)))
      .join(', ');
  }
  return 'Request failed';
}

export async function predictImage(file: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${apiBase()}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    let message = res.statusText;
    try {
      message = parseErrorDetail(await res.json());
    } catch {
      /* ignore */
    }
    throw new Error(message);
  }

  return res.json() as Promise<PredictResponse>;
}
