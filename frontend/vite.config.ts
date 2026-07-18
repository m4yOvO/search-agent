import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  // vis-network is isolated behind a lazy boundary; its own minified chunk is
  // intentionally just over Vite's generic 500 kB warning threshold.
  build: { chunkSizeWarningLimit: 550 },
  server: {
    port: 5173,
    proxy: {
      "/chat": "http://127.0.0.1:8000",
      "/graph": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/ready": "http://127.0.0.1:8000"
    }
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts",
    css: true
  }
});
