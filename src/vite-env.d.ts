/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_TEMPLATE_IMAGE_URL?: string;
  /** Set to "1" to run `npm run test:integration` against live Firebase. */
  readonly VITE_RUN_FIREBASE_INTEGRATION?: string;
  /** Override API host (no trailing slash). If unset, production uses https://chexit.onrender.com */
  readonly VITE_CHEXIT_API_URL?: string;
  /** Set to "1" to allow same-origin /api/predict on non-localhost (e.g. vite preview over LAN). */
  readonly VITE_USE_RELATIVE_API?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

