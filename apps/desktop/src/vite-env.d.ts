/// <reference types="vite/client" />

import type { StarlingDesktopBridge } from "../electron/ipc.js";

declare global {
  interface Window {
    starlingDesktop?: StarlingDesktopBridge;
  }
}
