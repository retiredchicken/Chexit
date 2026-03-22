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

/**
 * Predict URL:
 * - No VITE_CHEXIT_API_URL → always same-origin `/api/predict` (Vite dev + preview proxy → :8000).
 *   Works for any dev hostname (localhost, 127.0.0.1, Cursor tunnel, etc.).
 * - VITE_CHEXIT_API_URL set → direct URL (required for Vercel / production API).
 */
function predictUrl(): string {
  const trimmed = import.meta.env.VITE_CHEXIT_API_URL?.trim();
  if (trimmed) {
    return `${trimmed.replace(/\/$/, '')}/predict`;
  }
  return '/api/predict';
}

function apiLabelForErrors(): string {
  const trimmed = import.meta.env.VITE_CHEXIT_API_URL?.trim();
  if (trimmed) {
    return trimmed.replace(/\/$/, '');
  }
  return '/api (Vite → http://127.0.0.1:8000)';
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

function networkErrorHint(label: string): string {
  const httpsPage =
    typeof window !== 'undefined' && window.location.protocol === 'https:';
  const directHttp = (import.meta.env.VITE_CHEXIT_API_URL?.trim() ?? '').startsWith(
    'http://',
  );
  if (httpsPage && directHttp) {
    return (
      `Cannot call ${label} from an HTTPS page (mixed content). ` +
      `Remove VITE_CHEXIT_API_URL from .env so dev uses the Vite /api proxy, or use an https:// API URL, or open the app at http://localhost:5173.`
    );
  }
  return (
    `Cannot reach the API (${label}). Start FastAPI: cd chexit-backend && ./run_dev.sh — check http://127.0.0.1:8000/docs. ` +
    `Use npm run dev or npm run preview (not opening dist/ directly) so /api proxies to port 8000. ` +
    `Unset VITE_CHEXIT_API_URL for local proxy. Avoid uvicorn --reload during long /predict.`
  );
}

/** Score-CAM + TF can run several minutes; browsers rarely cancel, but proxies might. */
const PREDICT_TIMEOUT_MS = 10 * 60 * 1000;

function predictAbortSignal(): AbortSignal {
  if (typeof AbortSignal !== 'undefined' && typeof AbortSignal.timeout === 'function') {
    return AbortSignal.timeout(PREDICT_TIMEOUT_MS);
  }
  const c = new AbortController();
  setTimeout(() => c.abort(), PREDICT_TIMEOUT_MS);
  return c.signal;
}

function isAbortError(e: unknown): boolean {
  if (e instanceof DOMException && e.name === 'AbortError') {
    return true;
  }
  return e instanceof Error && e.name === 'AbortError';
}

function logClient(stage: string, detail?: Record<string, unknown>): void {
  const ts = new Date().toISOString();
  // Use console.log so messages show with default DevTools filters (Info is often hidden).
  if (detail) {
    console.log(`[Chexit ${ts}]`, stage, detail);
  } else {
    console.log(`[Chexit ${ts}]`, stage);
  }
}

export async function predictImage(file: File): Promise<PredictResponse> {
  const url = predictUrl();
  const label = apiLabelForErrors();
  logClient('predict: starting request', {
    url,
    fileName: file.name,
    fileSizeBytes: file.size,
    fileType: file.type,
  });

  const formData = new FormData();
  formData.append('file', file);

  const t0 = performance.now();
  let res: Response;
  try {
    logClient('predict: fetch POST /predict (server runs U-Net → MobileNet → Score-CAM; may take several minutes)');
    res = await fetch(url, {
      method: 'POST',
      body: formData,
      signal: predictAbortSignal(),
    });
    logClient('predict: response headers received', {
      status: res.status,
      ok: res.ok,
      elapsedSec: Math.round((performance.now() - t0) / 1000),
    });
  } catch (e) {
    logClient('predict: fetch failed', {
      error: e instanceof Error ? e.message : String(e),
      elapsedSec: Math.round((performance.now() - t0) / 1000),
    });
    if (isAbortError(e)) {
      throw new Error(
        `Analyze timed out after ${Math.round(PREDICT_TIMEOUT_MS / 60000)} minutes, or the request was cancelled. ` +
          `Try a smaller image, or run the API without --reload (see chexit-backend/run_dev.sh).`,
      );
    }
    const failedFetch =
      e instanceof TypeError &&
      (e.message === 'Failed to fetch' || e.message.includes('Load failed'));
    if (failedFetch) {
      throw new Error(networkErrorHint(label));
    }
    throw e;
  }

  const bodyText = await res.text();
  const contentType = res.headers.get('content-type') ?? '';
  logClient('predict: response body received', {
    status: res.status,
    ok: res.ok,
    contentType,
    bodyChars: bodyText.length,
    elapsedSec: Math.round((performance.now() - t0) / 1000),
  });

  let raw: unknown;
  try {
    raw = bodyText.length ? JSON.parse(bodyText) : null;
  } catch {
    const looksHtml =
      bodyText.trimStart().toLowerCase().startsWith('<!') ||
      bodyText.trimStart().toLowerCase().startsWith('<html');
    const hint = looksHtml
      ? 'The server returned HTML instead of JSON. On Vercel/static hosts, /api is not proxied — set VITE_CHEXIT_API_URL to your FastAPI base URL. Locally use npm run dev so /api forwards to port 8000.'
      : 'The response was not valid JSON (connection cut, proxy error, or wrong endpoint).';
    logClient('predict: JSON.parse failed', {
      hint,
      snippet: bodyText.slice(0, 200).replace(/\s+/g, ' '),
    });
    throw new Error(`${hint} First chars: ${bodyText.slice(0, 140).replace(/\s+/g, ' ')}`);
  }

  if (!res.ok) {
    let message = res.statusText;
    if (raw && typeof raw === 'object') {
      try {
        message = parseErrorDetail(raw);
      } catch {
        /* keep statusText */
      }
    }
    logClient('predict: HTTP error', { message });
    throw new Error(message || `Request failed (${res.status})`);
  }

  const out = normalizePredictResponse(raw);
  logClient('predict: success', {
    diagnosis: out.diagnosis,
    risk_score: out.risk_score,
    confidence_label: out.confidence_label,
    heatmapBase64Chars: out.heatmap.length,
    totalElapsedSec: Math.round((performance.now() - t0) / 1000),
  });
  return out;
}

function pickStr(o: Record<string, unknown>, ...keys: string[]): string {
  for (const k of keys) {
    const v = o[k];
    if (v != null && String(v).trim() !== '') {
      return String(v);
    }
  }
  return '';
}

function pickNum(o: Record<string, unknown>, ...keys: string[]): number {
  for (const k of keys) {
    const v = o[k];
    if (typeof v === 'number' && Number.isFinite(v)) {
      return v;
    }
    if (typeof v === 'string' && v.trim() !== '') {
      const n = Number(v);
      if (Number.isFinite(n)) {
        return n;
      }
    }
  }
  return NaN;
}

function normalizePredictResponse(raw: unknown): PredictResponse {
  if (!raw || typeof raw !== 'object') {
    throw new Error('Invalid API response: expected a JSON object.');
  }
  const o = raw as Record<string, unknown>;
  const diagnosis = pickStr(o, 'diagnosis', 'Diagnosis');
  const risk_score = pickNum(o, 'risk_score', 'riskScore');
  if (!Number.isFinite(risk_score)) {
    throw new Error('Invalid API response: risk_score is not a number.');
  }
  const confidence_label = pickStr(o, 'confidence_label', 'confidenceLabel');
  let heatmap = pickStr(o, 'heatmap', 'Heatmap');
  if (heatmap.startsWith('data:')) {
    const comma = heatmap.indexOf(',');
    if (comma !== -1) {
      heatmap = heatmap.slice(comma + 1);
    }
  }
  if (!heatmap.trim()) {
    throw new Error('Invalid API response: empty heatmap.');
  }
  return { diagnosis, risk_score, confidence_label, heatmap };
}
