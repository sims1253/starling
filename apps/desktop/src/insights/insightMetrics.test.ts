import { describe, expect, it } from "vite-plus/test";

import basicSession from "../../../../packages/contracts/insight-events/fixtures/basic-session.json";
import deletionPropagation from "../../../../packages/contracts/insight-events/fixtures/deletion-propagation.json";
import generatedOutput from "../../../../packages/contracts/insight-events/fixtures/generated-output.json";
import negativeProxy from "../../../../packages/contracts/insight-events/fixtures/negative-proxy.json";
import syncReplay from "../../../../packages/contracts/insight-events/fixtures/sync-replay.json";
import timezoneShift from "../../../../packages/contracts/insight-events/fixtures/timezone-shift.json";
import type { InsightEvent, RecognitionSelectedEvent } from "./insightEvents";
import { activityByDay, aggregate, localWallTime } from "./insightMetrics";

/**
 * Port of the E28 contract tests (`tests/test_insight_events.py`, metric and
 * timezone layers): every number below is frozen by the Python oracle — the
 * fixtures must keep producing exactly these aggregates through the
 * TypeScript port.
 */

function events(value: unknown): readonly InsightEvent[] {
  // SAFETY: fixture JSON from the frozen contract; the aggregate below
  // re-validates every numeric field it reads.
  return value as readonly InsightEvent[];
}

const BASIC = events(basicSession);

function selection(index: number): RecognitionSelectedEvent {
  // SAFETY: BASIC[index] is a recognition_selected fixture event.
  return BASIC[index] as RecognitionSelectedEvent;
}

function eventAt(index: number): InsightEvent {
  const event = BASIC[index];

  if (event === undefined) throw new Error("basic-session fixture shape changed");

  return event;
}

describe("aggregate: frozen metric semantics", () => {
  it("counts the basic session exactly as the oracle does", () => {
    const result = aggregate(BASIC, 50);

    expect(result.formula_version).toBe(1);
    expect(result.unique_takes).toBe(2);
    expect(result.selected_recognitions).toBe(2);
    expect(result.recognized_words).toBe(110); // latest selection only
    expect(result.raw_recognized_words).toBe(114);
    expect(result.captured_seconds).toBeCloseTo(50.0, 6);
    expect(result.eligible_capture_seconds).toBeCloseTo(50.0, 6);
    expect(result.recognized_words_per_captured_minute).toBeCloseTo(132.0, 6);
    expect(result.typing_time_comparison_seconds).toBeCloseTo(75.0, 6);
    expect(result.delivery_counts.confirmed).toBe(1);
    expect(result.delivery_counts.submitted_unconfirmed).toBe(1);
    expect(result.output_words_by_status.confirmed).toBe(120);
    expect(result.generated_words_by_status.confirmed).toBe(20);
    expect(result.output_words_by_status.submitted_unconfirmed).toBe(200);
    expect(result.generated_words_by_status.submitted_unconfirmed).toBe(190);
  });

  it("reports a weighted total, never a per-take average", () => {
    // take-1: 100 words / 40s = 150 wpm; take-2: 10 / 10s = 60 wpm.
    // The contract reports the weighted total 110*60/50 = 132, never the
    // mean of per-take rates (105).
    expect(aggregate(BASIC).recognized_words_per_captured_minute).toBeCloseTo(132.0, 6);
  });

  it("does not double-count sync replays or reordering", () => {
    expect(aggregate(events(syncReplay))).toEqual(aggregate(BASIC));
    expect(aggregate([...BASIC].reverse())).toEqual(aggregate(BASIC));
  });

  it("replaces rather than adds on retry", () => {
    // take-1 has two selections (90 then 100 lexical words); only 100 counts.
    expect(aggregate(BASIC).recognized_words).toBe(110);
  });

  it("removes attributable events on deletion and beats stale replay", () => {
    // take-3 (30 words, 5s, one confirmed delivery) is tombstoned; the file
    // also replays take-3's recognition AFTER the tombstone. Totals must
    // equal the untouched basic session.
    expect(aggregate(events(deletionPropagation))).toEqual(aggregate(BASIC));

    const tombstoneFirst: InsightEvent[] = [
      {
        schema_version: 1,
        event_id: "del-first",
        capture_id: "take-1",
        occurred_at: "2026-09-20T13:00:00Z",
        type: "capture_deleted",
      },
      ...BASIC,
    ];
    const result = aggregate(tombstoneFirst);

    expect(result.unique_takes).toBe(1);
    expect(result.recognized_words).toBe(10);
  });

  it("rejects a conflicting event id", () => {
    const conflict = { ...selection(1), lexical_words: 20 };

    expect(() => aggregate([...BASIC, conflict])).toThrow(/conflicting payload/);
  });

  it("rejects a conflicting selection sequence", () => {
    const conflict = { ...selection(1), event_id: "conflict", lexical_words: 20 };

    expect(() => aggregate([...BASIC, conflict])).toThrow(/sequence/);
  });

  it("refuses structurally impossible counts", () => {
    const lexicalOverRaw = {
      ...selection(1),
      event_id: "impossible-1",
      attempt_id: "attempt-x",
      selection_seq: 3,
      lexical_words: 500,
    };

    expect(() => aggregate([...BASIC, lexicalOverRaw])).toThrow(/Lexical/);

    const generatedOverOutput = {
      ...eventAt(4),
      event_id: "impossible-2",
      delivery_id: "delivery-x",
      generated_words: 999,
    };

    expect(() => aggregate([...BASIC, generatedOverOutput])).toThrow(/Generated/);

    const zeroRate = {
      ...eventAt(0),
      event_id: "impossible-3",
      capture_id: "take-x",
      sample_rate: 0,
    };

    expect(() => aggregate([...BASIC, zeroRate])).toThrow(/positive/);
  });

  it("never lets non-finite fields degrade into numbers", () => {
    const nanWords = {
      ...selection(1),
      event_id: "nan-words",
      attempt_id: "attempt-nan",
      selection_seq: 3,
      lexical_words: Number.NaN,
    };

    expect(() => aggregate([...BASIC, nanWords])).toThrow(/lexical_words must be finite/);

    const infWait = {
      ...selection(1),
      event_id: "inf-wait",
      attempt_id: "attempt-inf",
      selection_seq: 4,
      post_stop_ready_ms: Number.POSITIVE_INFINITY,
    };

    expect(() => aggregate([...BASIC, infWait])).toThrow(/post_stop_ready_ms must be finite/);

    const nanRate = {
      ...eventAt(0),
      event_id: "nan-rate",
      capture_id: "take-nan",
      sample_rate: Number.NaN,
    };

    expect(() => aggregate([...BASIC, nanRate])).toThrow(/sample_rate must be finite/);

    const nanCount = {
      ...eventAt(0),
      event_id: "nan-count",
      capture_id: "take-nan-2",
      sample_count: Number.NaN,
    };

    expect(() => aggregate([...BASIC, nanCount])).toThrow(/sample_count must be finite/);
  });

  it("refuses a bad typing baseline", () => {
    for (const value of [0, -1, Number.NaN]) {
      expect(() => aggregate(BASIC, value)).toThrow();
    }
  });

  it("disables the rate, not corrupts it, on incomparable tokenizers", () => {
    const mixed = BASIC.map((event) =>
      event.event_id === "a3" ? { ...event, tokenizer: "uax29-de-v1" } : event,
    );
    const result = aggregate(mixed);

    expect(result.recognized_words_per_captured_minute).toBeNull();
    expect(result.recognized_words).toBe(110);
  });

  it("means no proxy without a typing baseline", () => {
    expect(aggregate(BASIC).typing_time_comparison_seconds).toBeNull();
  });

  it("suppresses the proxy rather than guessing on an unknown wait", () => {
    const unknownWaits = BASIC.map((event) =>
      event.type === "recognition_selected" ? { ...event, post_stop_ready_ms: null } : event,
    );

    expect(aggregate(unknownWaits, 50).typing_time_comparison_seconds).toBeNull();
  });

  it("reports a negative time saved instead of clamping it", () => {
    // 100 words at 50 typing-wpm = 120s estimate; 40s capture + 1000s
    // post-stop wait => -920s. The fixture freezes that this is displayed.
    const result = aggregate(events(negativeProxy), 50);

    expect(result.typing_time_comparison_seconds).toBeCloseTo(-920.0, 6);
    expect(result.incomplete_captures).toBe(1);
  });

  it("keeps generated and snippet output separate from speech", () => {
    const result = aggregate(events(generatedOutput), 50);

    expect(result.recognized_words).toBe(110); // speech only, never 255
    expect(result.generated_words_by_status.confirmed).toBe(145); // 60 model + 85 snippet
    expect(result.output_words_by_status.confirmed).toBe(240);
    expect(result.recognized_words_per_captured_minute).toBeCloseTo(132.0, 6);
    // rev-g1 was refined twice: seq2 {style: 2} replaces seq1 {style: 5};
    // the snippet pass adds {snippet: 1}.
    expect(result.change_counts).toEqual({
      structural: 0,
      user: 0,
      dictionary: 0,
      snippet: 1,
      style: 2,
    });
    expect(result.transformation_counts.model_authoring).toBe(1);
    expect(result.transformation_counts.snippet_expansion).toBe(1);
    // take-g2 has an unknown wait => no proxy despite a baseline.
    expect(result.typing_time_comparison_seconds).toBeNull();
  });

  it("never counts a submission as a confirmed delivery", () => {
    const result = aggregate(BASIC);

    expect(result.delivery_counts.confirmed).toBe(1);
    expect(result.delivery_counts.submitted_unconfirmed).toBe(1);
    expect([result.output_words_by_status.confirmed, result.output_words_by_status.submitted_unconfirmed]).toEqual([
      120,
      200,
    ]);
  });

  it("treats orphan events as non-takes", () => {
    const orphan: InsightEvent = {
      schema_version: 1,
      event_id: "orphan",
      capture_id: "ghost",
      occurred_at: "2026-09-20T12:00:00Z",
      type: "recognition_selected",
      attempt_id: "ghost-1",
      selection_seq: 1,
      lexical_words: 99,
      raw_words: 99,
      tokenizer: "uax29-en-v1",
      post_stop_ready_ms: 10,
    };
    const result = aggregate([orphan]);

    expect(result.unique_takes).toBe(0);
    expect(result.recognized_words).toBe(0);
  });

  it("rejects a duplicate capture finalization", () => {
    const twin = { ...eventAt(0), event_id: "c1-twin" };

    expect(() => aggregate([...BASIC, twin])).toThrow(/canonical/);
  });
});

describe("timezone, reset, export", () => {
  it("groups the activity calendar by the reporting timezone", () => {
    const shifted = events(timezoneShift);
    const berlin = activityByDay(shifted, "Europe/Berlin");
    const utc = activityByDay(shifted, "UTC");

    // 23:30Z on Oct 24 is already Oct 25 in Berlin.
    expect(berlin.get("2026-10-25")).toEqual({ takes: 3, captured_seconds: 15 });
    expect(utc.get("2026-10-24")?.takes).toBe(1);
    expect(utc.get("2026-10-25")?.takes).toBe(2);
  });

  it("is not a naive wall clock across the DST fallback", () => {
    // During the Oct 25 2026 Berlin fallback, 00:30Z and 01:30Z both read
    // 02:30 local wall time but carry different UTC offsets (+2 then +1).
    const pre = localWallTime("2026-10-25T00:30:00Z", "Europe/Berlin");
    const post = localWallTime("2026-10-25T01:30:00Z", "Europe/Berlin");

    expect([pre.hour, pre.minute, pre.second]).toEqual([2, 30, 0]);
    expect([post.hour, post.minute, post.second]).toEqual([2, 30, 0]);
    expect(pre.utc_offset_hours).toBeCloseTo(2.0, 6);
    expect(post.utc_offset_hours).toBeCloseTo(1.0, 6);
  });

  it("propagates deletion into the activity calendar", () => {
    const days = activityByDay(events(deletionPropagation), "Europe/Berlin");
    let takes = 0;

    for (const day of days.values()) takes += day.takes;

    expect(takes).toBe(2); // take-3 removed
  });

  it("refuses an unknown timezone", () => {
    expect(() => activityByDay(events(timezoneShift), "Mars/Olympus_Mons")).toThrow(/timezone/);
  });

  it("yields a zero state on reset and round-trips exports as JSON", () => {
    const empty = aggregate([]);

    expect(empty.unique_takes).toBe(0);
    expect(empty.recognized_words).toBe(0);
    expect(empty.recognized_words_per_captured_minute).toBeNull();
    expect(empty.typing_time_comparison_seconds).toBeNull();

    // Export contract: every aggregate is plain JSON.
    const payloads = [empty, aggregate(BASIC, 50), aggregate(events(generatedOutput))];

    for (const payload of payloads) {
      expect(JSON.parse(JSON.stringify(payload))).toEqual(payload);
    }
  });
});
