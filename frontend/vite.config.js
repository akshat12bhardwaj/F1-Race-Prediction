import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server proxies /api to the FastAPI backend (port 8010 by default; another app uses 8000). Override with API_PORT.
// The built site is served by FastAPI itself and uses same-origin /api, so it works on any port.
const API_PORT = process.env.API_PORT || '8010'
export default defineConfig({
  plugins: [react()],
  server: { port: 5174, proxy: { '/api': `http://127.0.0.1:${API_PORT}` } },
  build: { chunkSizeWarningLimit: 5000 },
})
