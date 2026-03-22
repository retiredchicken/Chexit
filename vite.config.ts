import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

const apiProxy = {
  '/api': {
    target: 'http://127.0.0.1:8000',
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/api/, ''),
  },
} as const

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    extensions: ['.tsx', '.ts', '.jsx', '.js', '.json'],
  },
  server: {
    // Dev + HTTPS preview: same-origin /api → FastAPI (no mixed content)
    proxy: { ...apiProxy },
  },
  preview: {
    // `npm run preview` sets DEV=false; proxy must exist here too or Analyze hits 127.0.0.1 directly
    proxy: { ...apiProxy },
  },
  test: {
    environment: 'node',
  },
})

