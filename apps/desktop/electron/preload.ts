import { contextBridge, ipcRenderer } from "electron";
import type {
  DesktopDiagnostics,
  HealthInput,
  PendingAudioState,
  RefinementKeyLoadInput,
  RefinementKeyLoadResult,
  RefinementKeySaveInput,
  RefinementKeySaveResult,
  ServerHealth,
  StarlingDesktopBridge,
  TranscribeInput,
  TranscriptionResult,
} from "./ipc.js";
import { readableRejection } from "./ipc-errors.js";

async function invoke<T>(channel: string, ...args: ReadonlyArray<unknown>): Promise<T> {
  try {
    return await ipcRenderer.invoke(channel, ...args);
  } catch (cause) {
    throw new Error(readableRejection(channel, cause));
  }
}

const bridge: StarlingDesktopBridge = Object.freeze({
  health: (input: HealthInput) => invoke<ServerHealth>("starling:health", input),
  transcribe: (input: TranscribeInput) => invoke<TranscriptionResult>("starling:transcribe", input),
  storeRefinementKey: (input: RefinementKeySaveInput) =>
    invoke<RefinementKeySaveResult>("starling:refine-key:save", input),
  loadRefinementKey: (input: RefinementKeyLoadInput) =>
    invoke<RefinementKeyLoadResult>("starling:refine-key:load", input),
  diagnostics: () => invoke<DesktopDiagnostics>("starling:diagnostics"),
  ready: () => ipcRenderer.send("starling:renderer-ready"),
  setPendingAudio: (state: PendingAudioState) => ipcRenderer.send("starling:pending-audio", state),
  onToggleRecording: (callback: () => void) => {
    const listener = (): void => callback();
    ipcRenderer.on("starling:toggle-recording", listener);

    return () => ipcRenderer.removeListener("starling:toggle-recording", listener);
  },
  onDiscardPending: (callback: () => void) => {
    const listener = (): void => callback();
    ipcRenderer.on("starling:discard-pending", listener);

    return () => ipcRenderer.removeListener("starling:discard-pending", listener);
  },
  discardCleanedUp: () => ipcRenderer.send("starling:discard-cleaned"),
});

contextBridge.exposeInMainWorld("starlingDesktop", bridge);
