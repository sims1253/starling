import { defineConfig } from "vite-plus";
import react from "@vitejs/plugin-react";

const env = process.env;

const apiTarget = env.STARLING_API_TARGET ?? "http://127.0.0.1:8181";

export default defineConfig({
  base: "./",
  plugins: [react()],
  clearScreen: false,
  server: {
    // IPv4 loopback so the dev launcher's 127.0.0.1 health poll and renderer URL resolve.
    host: "127.0.0.1",
    port: 1420,
    strictPort: true,
    proxy: {
      // Listed before "/api" so the longer prefix wins: WebSocket upgrades
      // for live streaming need ws:true and the same /api strip.
      "/api/stream": {
        target: apiTarget.replace(/^http/, "ws"),
        ws: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
      "/api": {
        target: apiTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  envPrefix: ["VITE_", "ELECTRON_"],
  build: {
    target: "chrome138",
    minify: !env.ELECTRON_DEBUG,
    sourcemap: Boolean(env.ELECTRON_DEBUG),
  },
});
