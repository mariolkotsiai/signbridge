import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// Το dev server προωθεί /api και /ws στο FastAPI, οπότε ο browser μιλά με ένα μόνο origin
// (χωρίς CORS και χωρίς hard-coded θύρες στον κώδικα του client).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const backend = env.VITE_BACKEND_URL || 'http://127.0.0.1:8000';
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        '/api': { target: backend, changeOrigin: true },
        '/ws': { target: backend, ws: true, changeOrigin: true },
      },
    },
  };
});
