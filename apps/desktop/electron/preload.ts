import { contextBridge, ipcRenderer, type IpcRendererEvent } from "electron";
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
  StreamCloseInput,
  StreamCommandInput,
  StreamCommandResult,
  StreamEventMessage,
  StreamOpenInput,
  StreamSendInput,
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
  // Live-streaming transport over the main process (B01): the renderer never
  // opens a non-loopback WebSocket itself, so the static CSP stays loopback-
  // only. Events flow back on starling:stream:event, tagged with streamId.
  streamOpen: (input: StreamOpenInput) =>
    invoke<{ streamId: number }>("starling:stream:open", input),
  streamSend: (input: StreamSendInput) => invoke<void>("starling:stream:send", input),
  streamCommand: (input: StreamCommandInput) =>
    invoke<StreamCommandResult>("starling:stream:command", input),
  streamClose: (input: StreamCloseInput) => ipcRenderer.send("starling:stream:close", input),
  onStreamEvent: (callback: (message: StreamEventMessage) => void) => {
    const listener = (_event: IpcRendererEvent, message: StreamEventMessage): void =>
      callback(message);

    ipcRenderer.on("starling:stream:event", listener);

    return () => ipcRenderer.removeListener("starling:stream:event", listener);
  },
});

contextBridge.exposeInMainWorld("starlingDesktop", bridge);
