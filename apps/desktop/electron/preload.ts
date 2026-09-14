import { contextBridge, ipcRenderer } from "electron";
import type { HealthInput, StarlingDesktopBridge, TranscribeInput } from "./ipc.js";

const bridge: StarlingDesktopBridge = Object.freeze({
  health: (input: HealthInput) => ipcRenderer.invoke("starling:health", input),
  transcribe: (input: TranscribeInput) => ipcRenderer.invoke("starling:transcribe", input),
  diagnostics: () => ipcRenderer.invoke("starling:diagnostics"),
  ready: () => ipcRenderer.send("starling:renderer-ready"),
  onToggleRecording: (callback: () => void) => {
    const listener = (): void => callback();
    ipcRenderer.on("starling:toggle-recording", listener);

    return () => ipcRenderer.removeListener("starling:toggle-recording", listener);
  },
});

contextBridge.exposeInMainWorld("starlingDesktop", bridge);
