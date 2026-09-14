import { build } from "esbuild";
import { rm } from "node:fs/promises";

const shared = {
  bundle: true,
  logLevel: "info" as const,
  packages: "bundle" as const,
  platform: "node" as const,
  sourcemap: process.env.ELECTRON_DEBUG === "1",
  target: "node22",
  external: ["electron"],
};

await rm("dist-electron", { recursive: true, force: true });

await Promise.all([
  build({
    ...shared,
    entryPoints: ["electron/main.ts"],
    format: "esm",
    outfile: "dist-electron/main.mjs",
  }),
  build({
    ...shared,
    entryPoints: ["electron/preload.ts"],
    format: "cjs",
    outfile: "dist-electron/preload.cjs",
  }),
]);
