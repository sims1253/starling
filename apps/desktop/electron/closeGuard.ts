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

  // A journaled take between Stop and the end of its finalize is durable
  // either way — the journal resurrects it, or the session is already
  // persisted — so closing cannot delete it and it is never listed.
  if (state.finalizing && state.journaled !== true)
    atRisk.push("a recording that is still being saved");

  if (state.unsavedCount > 0)
    atRisk.push(`${state.unsavedCount} unsaved recording${state.unsavedCount === 1 ? "" : "s"}`);

  if (atRisk.length === 0) return undefined;

  return `Closing now permanently deletes ${joinAtRisk(atRisk)}.`;
}

/**
 * Wording for the reload gate. Unlike closing, a reload cannot give the
 * renderer time to delete a durable streaming journal mid-unload, so a
 * journaled take is NOT deleted: recovery offers it again on next start,
 * and the dialog must not promise otherwise. Memory-only audio is still
 * destroyed by the reload and keeps the deletion wording.
 */
export function pendingAudioReloadWarning(state: PendingAudioState): string | undefined {
  if (state.journaled !== true) return pendingAudioWarning(state);

  const atRisk: Array<string> = [];

  if (state.unsavedCount > 0)
    atRisk.push(`${state.unsavedCount} unsaved recording${state.unsavedCount === 1 ? "" : "s"}`);

  // A journaled take is durable whether live or finalizing: the reload
  // interrupts it, but recovery offers the saved audio again on next start,
  // so it is never listed as deleted. Only memory-only audio keeps the
  // deletion wording.
  if (!state.recording && !state.finalizing && atRisk.length === 0) return undefined;

  const kept = state.recording
    ? "Reload stops the live recording; the audio saved so far is kept and offered as a recovered take on the next start."
    : "Reload interrupts the save; the audio saved so far is kept and offered as a recovered take on the next start.";

  if (atRisk.length === 0) return kept;

  return `${kept} Reload also permanently deletes ${joinAtRisk(atRisk)}.`;
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
