import { Option, Schema } from "effect";
import { PendingAudioStateSchema, type PendingAudioState } from "./ipc.js";

/**
 * Decides whether closing Starling now would destroy audio that exists only in
 * renderer memory (#121), and describes exactly what is at risk. Returns
 * undefined when everything the user could lose is already durable.
 */
export function pendingAudioWarning(state: PendingAudioState): string | undefined {
  const atRisk: Array<string> = [];

  if (state.recording) atRisk.push("the live recording");

  if (state.finalizing) atRisk.push("a recording that is still being saved");

  if (state.unsavedCount > 0)
    atRisk.push(`${state.unsavedCount} unsaved recording${state.unsavedCount === 1 ? "" : "s"}`);

  if (atRisk.length === 0) return undefined;

  return `Closing now permanently deletes ${joinAtRisk(atRisk)}.`;
}

function joinAtRisk(items: ReadonlyArray<string>): string {
  if (items.length === 1) return items[0];

  if (items.length === 2) return `${items[0]} and ${items[1]}`;

  return `${items.slice(0, -1).join(", ")}, and ${items[items.length - 1]}`;
}

/**
 * Re-validates and normalizes the renderer's pending-audio mirror before the
 * close guard trusts it; malformed payloads keep the previous mirror instead
 * of disarming it.
 */
export function parsePendingAudio(state: PendingAudioState): PendingAudioState | undefined {
  const decoded = Schema.decodeUnknownOption(PendingAudioStateSchema)(state);

  if (Option.isNone(decoded)) return undefined;

  return {
    recording: decoded.value.recording,
    finalizing: decoded.value.finalizing,
    unsavedCount: Math.max(0, Math.trunc(decoded.value.unsavedCount)),
  };
}
