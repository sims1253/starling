import { contextBridge, ipcRenderer } from "electron";
import type {
  DesktopDiagnostics,
  HealthInput,
  ServerHealth,
  StarlingDesktopBridge,
  TranscribeInput,
  TranscriptionResult,
} from "./ipc.js";

async function invoke<T>(channel: string, ...args: ReadonlyArray<unknown>): Promise<T> {
  try {
    return await ipcRenderer.invoke(channel, ...args);
  } catch (cause) {
    throw new Error(readableRejection(channel, cause));
  }
}

// Request*Error messages reach the UI verbatim; only transport failures need
// rewording, and the remaining internal error-class tags are stripped.
function readableRejection(channel: string, cause: unknown): string {
  const channelPrefix = `Error invoking remote method '${channel}': `;
  const raw = cause instanceof Error ? cause.message : String(cause);
  let message = raw.startsWith(channelPrefix) ? raw.slice(channelPrefix.length) : raw;

  if (message.startsWith("RequestTransportError: ")) {
    return `Could not reach the transcription server. ${message.slice("RequestTransportError: ".length)}`;
  }

  message = message.replace(/^(?:Request(?:Input|Timeout|Http)Error):\s*/, "");

  return message;
}

const bridge: StarlingDesktopBridge = Object.freeze({
  health: (input: HealthInput) => invoke<ServerHealth>("starling:health", input),
  transcribe: (input: TranscribeInput) => invoke<TranscriptionResult>("starling:transcribe", input),
  diagnostics: () => invoke<DesktopDiagnostics>("starling:diagnostics"),
  ready: () => ipcRenderer.send("starling:renderer-ready"),
  onToggleRecording: (callback: () => void) => {
    const listener = (): void => callback();
    ipcRenderer.on("starling:toggle-recording", listener);

    return () => ipcRenderer.removeListener("starling:toggle-recording", listener);
  },
});

contextBridge.exposeInMainWorld("starlingDesktop", bridge);
