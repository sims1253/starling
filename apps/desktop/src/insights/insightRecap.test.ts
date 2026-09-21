import { Schema } from "effect";
import { describe, expect, it } from "vite-plus/test";

import basicSession from "../../../../packages/contracts/insight-events/fixtures/basic-session.json";
import { InsightEventSchema, type InsightEvent } from "./insightEvents";
import { weeklyRecap } from "./insightRecap";

/**
 * Weekly recap coverage (E29 phase 2): two consecutive seven-day windows of
 * equal instant length (matched periods — a DST transition cannot shorten
 * one side), deltas stated as counts with a direction and never as praise,
 * a quiet week reported as not-enough-data, and the frozen contract
 * fixtures recap exactly as the metrics layer computes them.
 */

const events = Schema.decodeUnknownSync(Schema.Array(InsightEventSchema), {
  onExcessProperty: "error",
});

const BASIC = events(basicSession);

const TZ = "Europe/Berlin";

describe("weeklyRecap", () => {
  it("recaps the frozen basic-session fixtures inside a fresh seven-day window", () => {
    // The fixture takes happened on 2026-09-20; viewed from 2026-09-21 noon
    // they are this week, and the week before has none.
    const result = weeklyRecap(BASIC, { timezone: TZ, now: Date.parse("2026-09-21T12:00:00Z") });

    expect(result.ok).toBe(true);

    if (result.ok) {
      expect(result.recap.thisWeek).toEqual({ words: 110, takes: 2, capturedSeconds: 50 });
      expect(result.recap.previousWeek).toBeNull();
      expect(result.recap.lines).toEqual([
        "This week: 110 words recognized from speech · 2 takes · 0.8 min captured",
        "No takes in the week before — this is the first week with takes in range.",
      ]);
    }
  });

  it("compares matched periods with a stated direction and no judgment", () => {
    const lastWeek = take("prev", "2026-09-12T10:00:00Z", 280);

    const thisWeek = [
      ...take("now1", "2026-09-20T10:00:00Z", 200),
      ...take("now2", "2026-09-20T11:00:00Z", 112),
    ];

    const result = weeklyRecap([...lastWeek, ...thisWeek], {
      timezone: TZ,
      now: Date.parse("2026-09-21T12:00:00Z"),
    });

    expect(result.ok).toBe(true);

    if (result.ok) {
      expect(result.recap.previousWeek).toEqual({ words: 280, takes: 1, capturedSeconds: 10 });
      expect(result.recap.changeWords).toBe(32);
      expect(
        result.recap.lines.some((line) => line === "Words this week were 32 more than last week."),
      ).toBe(true);
    }
  });

  it("reports fewer words without any failure language", () => {
    const result = weeklyRecap(
      [...take("prev", "2026-09-12T10:00:00Z", 280), ...take("now", "2026-09-20T10:00:00Z", 100)],
      { timezone: TZ, now: Date.parse("2026-09-21T12:00:00Z") },
    );

    expect(result.ok).toBe(true);

    if (result.ok) {
      expect(result.recap.changeWords).toBe(-180);
      expect(result.recap.lines.join(" ")).toContain("180 fewer than last week");
      expect(result.recap.lines.join(" ")).not.toMatch(/fail|behind|bad|slack/i);
    }
  });

  it("treats a week without takes as not-enough-data, not zero effort", () => {
    const result = weeklyRecap([...take("prev", "2026-09-12T10:00:00Z", 280)], {
      timezone: TZ,
      now: Date.parse("2026-09-21T12:00:00Z"),
    });

    expect(result).toEqual({
      ok: false,
      reason: "no takes in the last seven days — a quiet week is stated, not scored",
    });
  });

  it("keeps both windows seven instant-days across the Berlin DST fallback", () => {
    // Now is after the Oct 25 2026 fallback; window membership is decided
    // by instants, so wall-clock drift across the fallback cannot move a
    // take between windows or shorten either side.
    const now = Date.parse("2026-10-26T12:00:00Z");

    const result = weeklyRecap(
      [
        ...take("fresh", "2026-10-25T12:00:00Z", 50), // this week (before the fallback hour)
        ...take("edge", "2026-10-18T12:00:00Z", 50), // exactly 8 days before now: previous window
        ...take("old", "2026-10-11T12:00:00Z", 60), // exactly 15 days before now: outside both
      ],
      { timezone: TZ, now },
    );

    expect(result.ok).toBe(true);

    if (result.ok) {
      expect(result.recap.thisWeek.takes).toBe(1);
      expect(result.recap.previousWeek?.takes).toBe(1);
      expect(result.recap.previousWeek?.words).toBe(50);
    }

    // Viewed from one day earlier, the "edge" take is this week instead —
    // instant windows, not local wall-clock ones, decide.
    const shifted = weeklyRecap([...take("edge", "2026-10-18T12:00:00Z", 50)], {
      timezone: TZ,
      now: Date.parse("2026-10-25T12:00:00Z"),
    });

    expect(shifted.ok).toBe(true);

    if (shifted.ok) expect(shifted.recap.thisWeek.takes).toBe(1);
  });
});

function take(captureId: string, occurredAt: string, words: number): readonly InsightEvent[] {
  return [
    {
      schema_version: 1,
      event_id: `cf-${captureId}`,
      capture_id: captureId,
      occurred_at: occurredAt,
      type: "capture_finalized",
      sample_count: 160_000,
      sample_rate: 16_000,
      complete_audio: true,
      mode_id: "faithful",
      reporting_timezone: TZ,
    },
    {
      schema_version: 1,
      event_id: `rs-${captureId}`,
      capture_id: captureId,
      occurred_at: occurredAt,
      type: "recognition_selected",
      attempt_id: `att-${captureId}`,
      selection_seq: 1,
      lexical_words: words,
      raw_words: words,
      tokenizer: "uax29-intl-v1",
      post_stop_ready_ms: null,
    },
  ];
}
