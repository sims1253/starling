import { describe, expect, it } from "vite-plus/test";

import type { DictationSession } from "@starling/dictation";
import {
  DELETE_WHILE_REFINING,
  DELETE_WHILE_TRANSCRIBING,
  SessionDeleteDialog,
  deletionWarning,
} from "./sessionDeletion";

interface WarningFixture {
  readonly transcript?: { text: string; segments: [] };
  readonly history?: readonly { text: string; segments: [] }[];
  readonly refined?: { text: string; model: string; createdAt: number };
}

/** Mutable draft for fixture sessions — the SessionDraft pattern from
 * storage.ts — so optional fields are added only when present. */
interface WarningDraft {
  id: string;
  createdAt: string;
  updatedAt: string;
  status: "transcribed";
  wav: Blob;
  attemptCount: number;
  transcript?: { text: string; segments: [] };
  transcriptHistory?: readonly { text: string; segments: [] }[];
  refined?: { text: string; model: string; createdAt: number };
}

/** Minimal session literal — the fields the deletion warning reads. */
function take(fields: WarningFixture = {}): DictationSession {
  const session: WarningDraft = {
    id: "take",
    createdAt: "2026-09-20T10:00:00.000Z",
    updatedAt: "2026-09-20T10:00:00.000Z",
    status: "transcribed",
    wav: new Blob(),
    attemptCount: 1,
  };

  if (fields.transcript !== undefined) session.transcript = fields.transcript;

  if (fields.history !== undefined) session.transcriptHistory = fields.history;

  if (fields.refined !== undefined) session.refined = fields.refined;

  return Object.freeze(session);
}

describe("SessionDeleteDialog", () => {
  it("requires an explicit confirm before any delete happens (B05)", () => {
    const dialog = new SessionDeleteDialog();

    expect(dialog.request("take", { transcribing: false, refining: false }).kind).toBe("confirm");
    expect(dialog.requested()).toBe("take");

    // Nothing is deletable yet: cancel closes the dialog with no victim.
    dialog.cancel();

    expect(dialog.requested()).toBe(undefined);
  });

  it("hands out the id to delete exactly once", () => {
    const dialog = new SessionDeleteDialog();

    dialog.request("take", { transcribing: false, refining: false });

    expect(dialog.confirm()).toBe("take");
    expect(dialog.confirm()).toBe(undefined);
    expect(dialog.requested()).toBe(undefined);
  });

  it("refuses to open while the take is transcribing, with the reason", () => {
    const dialog = new SessionDeleteDialog();

    const intent = dialog.request("take", { transcribing: true, refining: false });

    expect(intent).toEqual({ kind: "blocked", message: DELETE_WHILE_TRANSCRIBING });
    expect(dialog.requested()).toBe(undefined);
  });

  it("refuses to open while the take is refining, with the reason", () => {
    const dialog = new SessionDeleteDialog();

    const intent = dialog.request("take", { transcribing: false, refining: true });

    expect(intent).toEqual({ kind: "blocked", message: DELETE_WHILE_REFINING });
    expect(dialog.requested()).toBe(undefined);
  });

  it("re-targets the dialog when another take is requested first", () => {
    const dialog = new SessionDeleteDialog();

    dialog.request("one", { transcribing: false, refining: false });
    dialog.request("two", { transcribing: false, refining: false });

    expect(dialog.requested()).toBe("two");
    expect(dialog.confirm()).toBe("two");
  });
});

describe("deletionWarning", () => {
  it("names the audio and every transcript version, and says there is no undo", () => {
    const warning = deletionWarning(
      take({
        transcript: { text: "raw", segments: [] },
        history: [{ text: "older", segments: [] }],
        refined: { text: "Refined.", model: "llama3.1", createdAt: 1 },
      }),
    );

    expect(warning).toContain("the audio");
    expect(warning).toContain("every saved transcript version");
    expect(warning).toContain("the refined copy");
    expect(warning).toContain("no undo");
  });

  it("covers transcript attempts even when only history entries exist", () => {
    const warning = deletionWarning(take({ history: [{ text: "older", segments: [] }] }));

    expect(warning).toContain("the audio");
    expect(warning).toContain("every saved transcript version");
    expect(warning).not.toContain("the refined copy");
  });

  it("warns about the audio alone for a take with no transcript yet", () => {
    const warning = deletionWarning(take());

    expect(warning).toContain("the audio");
    expect(warning).not.toContain("transcript version");
    expect(warning).toContain("no undo");
  });
});
