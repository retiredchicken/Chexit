# Chexit

TB screening UI (Vite + React + MUI) and FastAPI inference backend.

## Run everything locally

1. **Backend** — Python 3.11, venv, and deps (see `chexit-backend/README.md`).
2. **Two terminals** (or one combined command):

   ```bash
   npm run dev:stack
   ```

   This starts **Vite** (UI) and **uvicorn** (API on port 8000). Then open **http://localhost:5173/dashboard** for the analyzer.

3. **Flow** — Browse an X-ray → **Analyze** (calls `/api/predict` → `http://127.0.0.1:8000/predict`) → scrolls to **AI-assisted TB overview**: input preview, diagnosis/risk, base64 heatmap.

4. **Preview build** — `npm run build && npm run preview` also proxies `/api` to port 8000 when you use `localhost`.

5. **Landing** — `/` is sign-in; use **Open TB analyzer** or go to `/dashboard` directly.

Optional: `CHEXIT_SKIP_SCORECAM=1` on the API for faster local heatmaps (see `chexit-backend/README.md`).
