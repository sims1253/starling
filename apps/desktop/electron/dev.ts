import { spawn, type ChildProcess } from "node:child_process";
import { createRequire } from "node:module";

const url = "http://127.0.0.1:1420";

const executable: string = createRequire(import.meta.url)("electron");

const detached = process.platform !== "win32";

// pnpm puts node_modules/.bin on PATH; Windows needs a shell to launch the .cmd shim.
const vite = spawn("vp", ["dev"], {
  stdio: "inherit",
  detached,
  shell: process.platform === "win32",
});

let electron: ChildProcess | undefined;

function terminate(child: ChildProcess | undefined): void {
  if (!child?.pid || child.exitCode !== null) return;

  try {
    if (process.platform === "win32") {
      spawn("taskkill", ["/pid", String(child.pid), "/t", "/f"]);
    } else {
      process.kill(-child.pid, "SIGTERM");
    }
  } catch {
    child.kill();
  }
}

process.on("uncaughtException", (error) => {
  terminate(electron);
  terminate(vite);
  console.error(error);
  process.exit(1);
});

process.on("unhandledRejection", (error) => {
  throw error;
});

vite.on("exit", (code) => {
  if (electron) return;

  console.error(`Vite exited with code ${code ?? 1} before the renderer was ready.`);
  process.exit(code || 1);
});

for (let attempt = 0; attempt < 100; attempt += 1) {
  try {
    const response = await fetch(url);

    if (response.ok) break;
  } catch {
    // Vite is still starting.
  }

  await new Promise((resolve) => setTimeout(resolve, 100));

  if (attempt === 99) throw new Error("Vite did not start within 10 seconds.");
}

electron = spawn(executable, ["dist-electron/main.mjs"], {
  stdio: "inherit",
  detached,
  env: { ...process.env, STARLING_RENDERER_URL: url },
});

const stop = (): void => {
  terminate(electron);
  terminate(vite);
};

process.on("SIGINT", stop);

process.on("SIGTERM", stop);

electron.on("exit", (code) => {
  terminate(vite);
  process.exit(code ?? 0);
});
