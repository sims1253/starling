import { describe, expect, it } from "vite-plus/test";

import type { DictationSession, TranscriptAttempt } from "@starling/dictation";
import {
  attemptProvenanceLabel,
  canTranscribeAgain,
  refinedTranscriptStamp,
  sessionTitle,
  transcriptExportText,
  transcribeAgainLabel,
} from "./transcriptAttempts";

/** Deterministic stand-in for the drawer's formatWhen. */
const when = (iso: string) => `@${iso}`;

interface TakeFixture {
  readonly id?: string;
  readonly status?: DictationSession["status"];
  /** Undefined defaults to "raw words"; null omits the transcript entirely. */
  readonly text?: string | null;
  readonly history?: readonly TranscriptAttempt[];
  readonly refined?: DictationSession["refined"];
}

/** Mutable draft for fixture sessions — the SessionDraft pattern from
 * storage.ts — so optional fields are added only when present. */
interface TakeDraft {
  id: string;
  createdAt: string;
  updatedAt: string;
  status: DictationSession["status"];
  wav: Blob;
  attemptCount: number;
  transcript?: { text: string; segments: [] };
  transcriptHistory?: readonly TranscriptAttempt[];
  refined?: { text: string; model: string; createdAt: number };
}

/** Minimal transcribed session literal — the fields attempt logic reads. */
function take(fields: TakeFixture = {}): DictationSession {
  const session: TakeDraft = {
    id: fields.id ?? "take",
    createdAt: "2026-09-20T10:00:00.000Z",
    updatedAt: "2026-09-20T10:00:00.000Z",
    status: fields.status ?? "transcribed",
    wav: new Blob(),
    attemptCount: 1,
  };

  if (fields.text !== null) {
    session.transcript = { text: fields.text ?? "raw words", segments: [] };
  }

  if (fields.history !== undefined) session.transcriptHistory = fields.history;

  if (fields.refined !== undefined) session.refined = fields.refined;

  return Object.freeze(session);
}

describe("canTranscribeAgain", () => {
  it("offers retranscription for a successful result (B04)", () => {
    expect(canTranscribeAgain("transcribed", false)).toBe(true);
  });

  it("offers retranscription for an empty transcript without re-recording (B04)", () => {
    expect(canTranscribeAgain("transcribed", false)).toBe(true);
  });

  it("offers retranscription for failed and never-attempted takes", () => {
    expect(canTranscribeAgain("failed", false)).toBe(true);
    expect(canTranscribeAgain("captured", false)).toBe(true);
  });

  it("never offers retranscription while an attempt is running", () => {
    expect(canTranscribeAgain("transcribing", false)).toBe(false);
    expect(canTranscribeAgain("failed", true)).toBe(false);
    expect(canTranscribeAgain("transcribed", true)).toBe(false);
  });
});

describe("transcribeAgainLabel", () => {
  it("labels the action per state: Retry, Transcribe again, Transcribe", () => {
    expect(transcribeAgainLabel("failed")).toBe("Retry");
    expect(transcribeAgainLabel("transcribed")).toBe("Transcribe again");
    expect(transcribeAgainLabel("captured")).toBe("Transcribe");
  });
});

describe("sessionTitle", () => {
  it("uses the transcript text when present", () => {
    expect(sessionTitle(take({ text: "hello" }))).toBe("hello");
  });

  it("marks an empty transcript as review-needed, not in-flight (B04)", () => {
    expect(sessionTitle(take({ text: "" }))).toBe("Empty transcript");
  });

  it("keeps the legacy fallbacks for takes without a transcript", () => {
    expect(sessionTitle(take({ status: "failed", text: null }))).toBe("Saved. Retry available");
  });
});

describe("attemptProvenanceLabel", () => {
  it("joins model, protocol, and time provenance", () => {
    expect(
      attemptProvenanceLabel(
        {
          text: "a",
          segments: [],
          model: "parakeet",
          protocol: "starling",
          savedAt: "2026-09-20T10:00:00.000Z",
        },
        when,
      ),
    ).toBe("parakeet · starling · @2026-09-20T10:00:00.000Z");
  });

  it("falls back to a plain label for legacy attempts without provenance", () => {
    expect(attemptProvenanceLabel({ text: "a", segments: [] }, when)).toBe("earlier attempt");
  });
});

describe("transcriptExportText", () => {
  it("exports the raw transcript alone, intact", () => {
    expect(transcriptExportText(take())).toBe("raw words\n");
  });

  it("appends every earlier attempt under its own provenance separator, oldest first (B04)", () => {
    const session = take({
      history: [
        {
          text: "first pass",
          segments: [],
          model: "parakeet",
          protocol: "starling",
          savedAt: "2026-09-20T10:00:00.000Z",
        },
        {
          text: "",
          segments: [],
          model: "other",
          protocol: "openai",
          savedAt: "2026-09-20T11:00:00.000Z",
        },
      ],
    });

    expect(transcriptExportText(session)).toBe(
      [
        "raw words",
        "--- EARLIER ATTEMPT — parakeet, starling, 2026-09-20T10:00:00.000Z ---",
        "first pass",
        "--- EARLIER ATTEMPT — other, openai, 2026-09-20T11:00:00.000Z ---",
        "(empty transcript)",
      ].join("\n\n") + "\n",
    );
  });

  it("keeps the refined copy last, behind its separator", () => {
    const session = take({
      history: [{ text: "first pass", segments: [], savedAt: "2026-09-20T10:00:00.000Z" }],
      refined: { text: "Refined words.", model: "llama3.1", createdAt: 1_760_000_000_000 },
    });

    expect(transcriptExportText(session)).toBe(
      [
        "raw words",
        "--- EARLIER ATTEMPT — 2026-09-20T10:00:00.000Z ---",
        "first pass",
        "--- REFINED TRANSCRIPT — llama3.1, 2025-10-09T08:53:20.000Z ---",
        "Refined words.",
      ].join("\n\n") + "\n",
    );
  });

  it("stamps a refined copy as model, when", () => {
    expect(
      refinedTranscriptStamp(
        { text: "x", model: "llama3.1", createdAt: 5 },
        "2026-01-01T00:00:00.000Z",
      ),
    ).toBe("llama3.1, 2026-01-01T00:00:00.000Z");
  });
});
