import { Option, Schema } from "effect";
import { PendingAudioStateSchema, type PendingAudioState } from "./ipc.js";

/**
 * Decides whether closing Starling now would destroy audio that exists only in
 * renderer memory (#121), and describes exactly what is at risk. Returns
 * undefined when everything the user could lose is already durable.
 */
export function pendingAudioWarning(state: PendingAudioState): string | undefined {
  const atRisk: Array<string> = [];

  if (state.recording) {
    // A journaled (streaming) recording is durable up to its last chunk;
    // Discard removes that journal too, so it is deleted with the take —
    // but the wording must not imply no audio exists yet.
    atRisk.push(
      state.journaled === true
        ? "the live recording (including the audio saved so far)"
        : "the live recording",
    );
  }

  if (state.finalizing) atRisk.push("a recording that is still being saved");

  if (state.unsavedCount > 0)
    atRisk.push(`${state.unsavedCount} unsaved recording${state.unsavedCount === 1 ? "" : "s"}`);

  if (atRisk.length === 0) return undefined;

  return `Closing now permanently deletes ${joinAtRisk(atRisk)}.`;
}

/** Mutable draft of the normalized mirror; `journaled` stays absent unless true. */
type MutablePendingAudio = {
  recording: boolean;
  finalizing: boolean;
  unsavedCount: number;
  journaled?: boolean;
};

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

  const normalized: MutablePendingAudio = {
    recording: decoded.value.recording,
    finalizing: decoded.value.finalizing,
    unsavedCount: Math.max(0, Math.trunc(decoded.value.unsavedCount)),
  };

  if (decoded.value.journaled === true) normalized.journaled = true;

  return normalized;
}
