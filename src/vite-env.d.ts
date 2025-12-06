/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_TEMPLATE_IMAGE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

