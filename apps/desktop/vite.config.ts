import { defineConfig } from "vite-plus";
import react from "@vitejs/plugin-react";

const env = process.env;

const apiTarget = env.STARLING_API_TARGET ?? "http://127.0.0.1:8181";

export default defineConfig({
  base: "./",
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    proxy: {
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
