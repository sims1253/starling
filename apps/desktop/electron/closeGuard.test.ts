import assert from "node:assert/strict";
import { describe, it } from "vite-plus/test";
import { parsePendingAudio, pendingAudioWarning } from "./closeGuard.js";
import type { PendingAudioState } from "./ipc.js";

const clean: PendingAudioState = { recording: false, finalizing: false, unsavedCount: 0 };

// A mutable mirror shape so tests can delete keys the schema requires.
type WritablePendingAudio = {
  recording?: boolean;
  finalizing?: boolean;
  unsavedCount?: number;
};

describe("pendingAudioWarning", () => {
  it("stays silent when nothing is at risk", () => {
    assert.equal(pendingAudioWarning(clean), undefined);
  });

  it("names a live recording", () => {
    assert.equal(
      pendingAudioWarning({ ...clean, recording: true }),
      "Closing now permanently deletes the live recording.",
    );
  });

  it("names a capture that is still being saved", () => {
    assert.equal(
      pendingAudioWarning({ ...clean, finalizing: true }),
      "Closing now permanently deletes a recording that is still being saved.",
    );
  });

  it("names unsaved recovery recordings, singular and plural", () => {
    assert.equal(
      pendingAudioWarning({ ...clean, unsavedCount: 1 }),
      "Closing now permanently deletes 1 unsaved recording.",
    );
    assert.equal(
      pendingAudioWarning({ ...clean, unsavedCount: 3 }),
      "Closing now permanently deletes 3 unsaved recordings.",
    );
  });

  it("combines everything at risk in one sentence", () => {
    assert.equal(
      pendingAudioWarning({ recording: true, finalizing: true, unsavedCount: 2 }),
      "Closing now permanently deletes the live recording, a recording that is still being saved, and 2 unsaved recordings.",
    );
  });
});

describe("parsePendingAudio", () => {
  it("accepts the renderer's mirror state", () => {
    assert.deepEqual(parsePendingAudio({ recording: true, finalizing: false, unsavedCount: 3 }), {
      recording: true,
      finalizing: false,
      unsavedCount: 3,
    });
  });

  it("clamps counts to whole non-negative numbers", () => {
    assert.equal(parsePendingAudio({ ...clean, unsavedCount: -4 })?.unsavedCount, 0);
    assert.equal(parsePendingAudio({ ...clean, unsavedCount: 2.7 })?.unsavedCount, 2);
  });

  it("rejects malformed payloads instead of disarming the guard", () => {
    assert.equal(
      parsePendingAudio({ recording: true, finalizing: false, unsavedCount: Number.NaN }),
      undefined,
    );

    const incomplete: WritablePendingAudio = { ...clean };

    delete incomplete.recording;

    // SAFETY: deliberately incomplete payload proving the schema still gates it.
    assert.equal(parsePendingAudio(incomplete as PendingAudioState), undefined);
  });
});
