/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_TEMPLATE_IMAGE_URL?: string;
  /** Set to "1" to run `npm run test:integration` against live Firebase. */
  readonly VITE_RUN_FIREBASE_INTEGRATION?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

