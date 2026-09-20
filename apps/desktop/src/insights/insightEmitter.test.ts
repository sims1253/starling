import { describe, expect, it } from "vite-plus/test";

import { encodeWav16k } from "@starling/dictation";
import { aggregate } from "./insightMetrics";
import {
  InsightRecorder,
  TOKENIZER_ID,
  segmentWordCounts,
  transformationChanges,
  wavCaptureStats,
  wordMultisetDifference,
} from "./insightEmitter";
import { MemoryInsightEventStore, insightEventProblems } from "./insightEvents";

/**
 * Emitter coverage (E29): every emitted event conforms to the frozen schema —
 * which is the structural privacy contract, so no transcript text, selection
 * or path can ride along — and the recorder's sequence derivation gives the
 * aggregate the oracle's semantics: retries replace, deletes dominate, and
 * generated words stay separate from speech.
 */

const T0 = Date.parse("2026-09-20T12:00:00Z");

function recorder() {
  const store = new MemoryInsightEventStore();
  let tick = 0;
  let serial = 0;

  return {
    store,
    recorder: new InsightRecorder(store, {
      now: () => new Date(T0 + tick++ * 1000),
      uuid: () => `00000000-0000-4000-8000-${String(serial++).padStart(12, "0")}`,
    }),
  };
}

describe("declared tokenizer", () => {
  it("counts word-like segments as lexical and all tokens as raw", () => {
    const counts = segmentWordCounts("Hello, world — don't guess 3.14!");

    expect(counts.lexical).toBe(5); // Hello world don't guess 3.14
    expect(counts.raw).toBeGreaterThanOrEqual(counts.lexical); // punctuation clusters included
    expect(segmentWordCounts("")).toEqual({ lexical: 0, raw: 0 });
    expect(segmentWordCounts("   \n\t ")).toEqual({ lexical: 0, raw: 0 });
  });

  it("computes multiset additions and removals", () => {
    expect(wordMultisetDifference("one two three", "one two three")).toEqual({
      additions: 0,
      removals: 0,
    });
    expect(wordMultisetDifference("one two three", "one too three")).toEqual({
      additions: 1,
      removals: 1,
    });
    expect(wordMultisetDifference("a a b", "a")).toEqual({ additions: 0, removals: 2 });
    expect(wordMultisetDifference("a", "a a b")).toEqual({ additions: 2, removals: 0 });
  });

  it("maps a revision diff to declared change kinds", () => {
    // A casing/punctuation fix reads as one style change, nothing structural.
    expect(transformationChanges("hello world", "Hello, world!")).toEqual({
      change_counts: { structural: 0, user: 0, dictionary: 0, snippet: 0, style: 1 },
      generated_words: 1,
    });
    // A net addition of words reads as structural, not style.
    expect(transformationChanges("hello world", "hello there world")).toEqual({
      change_counts: { structural: 1, user: 0, dictionary: 0, snippet: 0, style: 0 },
      generated_words: 1,
    });
  });

  it("reads sample frames from a canonical WAV", async () => {
    const samples = new Float32Array(16_000);
    // A fresh Uint8Array copy gives an ArrayBuffer-backed view, which Blob
    // accepts (IndexedDB can hand back SharedArrayBuffer-backed buffers).
    const bytes = new Uint8Array(encodeWav16k({ samples, sampleRate: 16_000, channels: 1 }));
    const wav = new Blob([bytes]);

    await expect(wavCaptureStats(wav)).resolves.toEqual({
      sampleCount: 16_000,
      sampleRate: 16_000,
    });
  });
});

describe("InsightRecorder", () => {
  it("emits schema-valid events for the full lifecycle", async () => {
    const { recorder: insights } = recorder();

    await insights.captureFinalized({
      captureId: "take-1",
      sampleCount: 640_000,
      sampleRate: 16_000,
      completeAudio: true,
    });
    await insights.recognitionSelected({
      captureId: "take-1",
      transcriptText: "one two three four five",
      postStopReadyMs: 1200,
    });
    await insights.transformationCompleted({
      captureId: "take-1",
      rawText: "one two three four five",
      revisedText: "One, two, three, four, five!",
    });
    await insights.deliveryRecorded({
      captureId: "take-1",
      status: "confirmed",
      outputText: "One, two, three, four, five!",
      revisedText: "One, two, three, four, five!",
      baselineText: "one two three four five",
    });

    const snapshot = insights.snapshot();

    expect(snapshot).toHaveLength(4);

    for (const event of snapshot) {
      expect(insightEventProblems(event), event.event_id).toEqual([]);
    }

    const result = aggregate(snapshot, 50);

    expect(result.unique_takes).toBe(1);
    expect(result.recognized_words).toBe(5);
    expect(result.recognized_words_per_captured_minute).toBeCloseTo((5 * 60) / 40, 6);
    // Only "one" -> "One" changed word-for-word; commas are not word-like
    // segments, so this reads as one style change and one generated word.
    expect(result.change_counts.style).toBe(1);
    expect(result.delivery_counts.confirmed).toBe(1);
    expect(result.generated_words_by_status.confirmed).toBe(1);
    // 5 words at 50 wpm = 6s typing estimate; 40s capture + 1.2s wait: the
    // negative result is reported, never clamped to zero.
    expect(result.typing_time_comparison_seconds).toBeCloseTo(6 - 40 - 1.2, 6);
  });

  it("marks every emitted recognition with the declared tokenizer", async () => {
    const { recorder: insights } = recorder();

    await insights.captureFinalized({
      captureId: "take-t",
      sampleCount: 1,
      sampleRate: 16_000,
      completeAudio: true,
    });
    await insights.recognitionSelected({
      captureId: "take-t",
      transcriptText: "words",
      postStopReadyMs: null,
    });

    const selected = insights.snapshot().find((event) => event.type === "recognition_selected");

    expect(selected).toMatchObject({ tokenizer: TOKENIZER_ID, post_stop_ready_ms: null });
  });

  it("replaces a retry's word count instead of adding it", async () => {
    const { recorder: insights } = recorder();

    await insights.captureFinalized({
      captureId: "take-r",
      sampleCount: 160_000,
      sampleRate: 16_000,
      completeAudio: true,
    });
    await insights.recognitionSelected({
      captureId: "take-r",
      transcriptText: "one two three four",
      postStopReadyMs: 500,
    });
    await insights.recognitionSelected({
      captureId: "take-r",
      transcriptText: "one two three four five",
      postStopReadyMs: null,
    });

    const result = aggregate(insights.snapshot());

    expect(result.recognized_words).toBe(5); // not 9
    // The unknown wait on the retry suppresses the time proxy entirely.
    expect(result.typing_time_comparison_seconds).toBeNull();
  });

  it("counts each refinement pass as its own revision", async () => {
    const { recorder: insights } = recorder();

    await insights.captureFinalized({
      captureId: "take-p",
      sampleCount: 1,
      sampleRate: 16_000,
      completeAudio: true,
    });
    await insights.transformationCompleted({
      captureId: "take-p",
      rawText: "a b",
      revisedText: "a c",
    });
    await insights.transformationCompleted({
      captureId: "take-p",
      rawText: "a b",
      revisedText: "A c",
    });

    const result = aggregate(insights.snapshot());

    expect(result.transformation_counts.model_authoring).toBe(2);
    // Pass one: "b"->"c" is one style change. Pass two: "a b"->"A c" swaps
    // both words ("a" also gained a capital), so two more.
    expect(result.change_counts.style).toBe(3);
  });

  it("records export deliveries as submitted, never confirmed", async () => {
    const { recorder: insights } = recorder();

    await insights.captureFinalized({
      captureId: "take-d",
      sampleCount: 1,
      sampleRate: 16_000,
      completeAudio: true,
    });
    await insights.deliveryRecorded({
      captureId: "take-d",
      status: "submitted_unconfirmed",
      outputText: "some words",
    });
    await insights.deliveryRecorded({
      captureId: "take-d",
      status: "failed",
      outputText: "some words",
    });

    const result = aggregate(insights.snapshot());

    expect(result.delivery_counts.submitted_unconfirmed).toBe(1);
    expect(result.delivery_counts.failed).toBe(1);
    expect(result.delivery_counts.confirmed).toBe(0);
    expect(result.generated_words_by_status.submitted_unconfirmed).toBe(0);
  });

  it("dominates stale replays after a delete", async () => {
    const { recorder: insights } = recorder();

    await insights.captureFinalized({
      captureId: "take-x",
      sampleCount: 160_000,
      sampleRate: 16_000,
      completeAudio: true,
    });
    await insights.recognitionSelected({
      captureId: "take-x",
      transcriptText: "one two",
      postStopReadyMs: 100,
    });

    const staleReplay = insights.snapshot().find((event) => event.type === "recognition_selected");

    expect(staleReplay).toBeDefined();

    await insights.captureDeleted("take-x");

    // Even a new post-deletion recognition event cannot resurrect the
    // capture's totals: the tombstone dominates regardless of order.
    await insights.recognitionSelected({
      captureId: "take-x",
      transcriptText: "one two three",
      postStopReadyMs: 100,
    });

    const result = aggregate(insights.snapshot());

    expect(result.unique_takes).toBe(0);
    expect(result.recognized_words).toBe(0);

    // The second deletion of the same capture is an idempotent no-op.
    await insights.captureDeleted("take-x");

    const tombstones = insights.snapshot().filter((event) => event.type === "capture_deleted");

    expect(tombstones).toHaveLength(1);
  });

  it("conflicts loudly when the same finalized capture is re-stamped", async () => {
    const { recorder: insights } = recorder();

    await insights.captureFinalized({
      captureId: "take-c",
      sampleCount: 1,
      sampleRate: 16_000,
      completeAudio: true,
    });
    // Same capture, different payload (a later occurred_at from the ticking
    // clock): a conflict, not a harmless replay.
    await expect(
      insights.captureFinalized({
        captureId: "take-c",
        sampleCount: 1,
        sampleRate: 16_000,
        completeAudio: true,
      }),
    ).rejects.toThrow(/conflicting payloads/);
  });

  it("replays an identical capture finalization as a no-op", async () => {
    const store = new MemoryInsightEventStore();

    const fixed = new InsightRecorder(store, {
      now: () => new Date(T0),
      uuid: () => "uuid",
    });

    await fixed.captureFinalized({
      captureId: "take-same",
      sampleCount: 1,
      sampleRate: 16_000,
      completeAudio: true,
    });
    await fixed.captureFinalized({
      captureId: "take-same",
      sampleCount: 1,
      sampleRate: 16_000,
      completeAudio: true,
    });

    expect(fixed.snapshot()).toHaveLength(1);
  });

  it("resets to an empty population", async () => {
    const { recorder: insights } = recorder();

    await insights.captureFinalized({
      captureId: "take-z",
      sampleCount: 1,
      sampleRate: 16_000,
      completeAudio: true,
    });
    await insights.reset();

    expect(insights.snapshot()).toHaveLength(0);
    expect(aggregate(insights.snapshot()).unique_takes).toBe(0);
  });

  it("loads a persisted log back into its mirror", async () => {
    const store = new MemoryInsightEventStore();
    const writer = new InsightRecorder(store, { now: () => new Date(T0), uuid: () => "u" });

    await writer.captureFinalized({
      captureId: "take-l",
      sampleCount: 1,
      sampleRate: 16_000,
      completeAudio: true,
    });

    const reader = new InsightRecorder(store, { now: () => new Date(T0), uuid: () => "u" });

    await reader.load();

    expect(reader.snapshot()).toHaveLength(1);
    expect(aggregate(reader.snapshot()).unique_takes).toBe(1);
  });
});
