import { describe, expect, it } from "vite-plus/test";

import { DEFAULT_INSIGHT_CONSENT } from "./insightConsent";
import type { CaptureDeletedEvent, InsightEvent, RecognitionSelectedEvent } from "./insightEvents";
import {
  dayTotals,
  milestonePanel,
  milestones,
  streaks,
  weeklyGoalProgress,
} from "./insightMilestones";

/**
 * Milestone/goal/streak coverage (E29 phase 2): everything here derives
 * from the numerical event log under the contract's own rules (retries
 * replace, tombstones remove), milestone evidence cites its counts, goal
 * progress is stated without judgment, the streak's current run survives a
 * take-less today, and the whole surface stays off until opted in.
 */

const RATE = 16_000;

function take(
  captureId: string,
  occurredAt: string,
  words: number,
  options: { readonly seconds?: number } = {},
): readonly InsightEvent[] {
  return [
    {
      schema_version: 1,
      event_id: `cf-${captureId}`,
      capture_id: captureId,
      occurred_at: occurredAt,
      type: "capture_finalized",
      sample_count: Math.round((options.seconds ?? 10) * RATE),
      sample_rate: RATE,
      complete_audio: true,
      mode_id: "faithful",
      reporting_timezone: "Europe/Berlin",
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

function flat(groups: readonly (readonly InsightEvent[])[]): readonly InsightEvent[] {
  return groups.flat();
}

/** A deletion tombstone for `captureId` — the deletion guarantee under test. */
function deleted(captureId: string, occurredAt: string): CaptureDeletedEvent {
  return {
    schema_version: 1,
    event_id: `cd-${captureId}`,
    capture_id: captureId,
    occurred_at: occurredAt,
    type: "capture_deleted",
  };
}

describe("dayTotals", () => {
  it("buckets by local day and applies the contract's replace-on-retry rule", () => {
    const retry: RecognitionSelectedEvent = {
      schema_version: 1,
      event_id: "rs-t3-2",
      capture_id: "t3",
      occurred_at: "2026-09-19T09:05:00Z",
      type: "recognition_selected",
      attempt_id: "att-t3-2",
      selection_seq: 2,
      lexical_words: 50,
      raw_words: 50,
      tokenizer: "uax29-intl-v1",
      post_stop_ready_ms: null,
    };

    const events = [
      ...take("t1", "2026-09-18T10:00:00Z", 60),
      ...take("t2", "2026-09-18T15:00:00Z", 40),
      // t3's first selection (30 words) is superseded by a higher sequence.
      ...take("t3", "2026-09-19T09:00:00Z", 30),
      retry,
    ];

    const days = dayTotals(events, "Europe/Berlin");

    expect(days).toEqual([
      { day: "2026-09-18", takes: 2, recognizedWords: 100, capturedSeconds: 20 },
      { day: "2026-09-19", takes: 1, recognizedWords: 50, capturedSeconds: 10 },
    ]);
  });

  it("drops tombstoned captures and orphans entirely", () => {
    const tombstone: CaptureDeletedEvent = {
      schema_version: 1,
      event_id: "cd-t2",
      capture_id: "t2",
      occurred_at: "2026-09-18T12:00:00Z",
      type: "capture_deleted",
    };

    const events = [
      ...take("t1", "2026-09-18T10:00:00Z", 60),
      ...take("t2", "2026-09-18T11:00:00Z", 40),
      tombstone,
    ];

    const days = dayTotals(events, "Europe/Berlin");

    expect(days).toEqual([
      { day: "2026-09-18", takes: 1, recognizedWords: 60, capturedSeconds: 10 },
    ]);
  });

  it("separates local days across a midnight boundary", () => {
    const events = flat([
      take("t1", "2026-09-18T21:30:00Z", 10), // 23:30 Berlin, Sep 18
      take("t2", "2026-09-18T22:30:00Z", 20), // 00:30 Berlin, Sep 19
    ]);

    expect(dayTotals(events, "Europe/Berlin").map((day) => day.day)).toEqual([
      "2026-09-18",
      "2026-09-19",
    ]);
  });
});

describe("milestones", () => {
  it("celebrates the first take and the seventh distinct day", () => {
    const days = ["01", "02", "03", "04", "05", "06", "07"];
    const events = flat(days.map((day, index) => take(`t${index}`, `2026-09-${day}T10:00:00Z`, 5)));

    const reached = milestones(events, { timezone: "Europe/Berlin" });

    expect(reached.map((milestone) => milestone.id)).toEqual(["first-take", "first-week"]);
    expect(reached[0]?.achievedOnDay).toBe("2026-09-01");
    expect(reached[1]?.achievedOnDay).toBe("2026-09-07");
    expect(reached[1]?.evidence).toBe("The seventh distinct day with at least one take.");
  });

  it("does not count a day whose only takes were deleted toward the seventh distinct day", () => {
    // Seven distinct days, then every take on the seventh is deleted: only
    // six live days remain, so the first-week milestone must not fire.
    const days = ["01", "02", "03", "04", "05", "06", "07"];
    const events = flat([
      ...days.map((day, index) => take(`t${index}`, `2026-09-${day}T10:00:00Z`, 5)),
      deleted("t6", "2026-09-08T10:00:00Z"),
    ]);

    const reached = milestones(events, { timezone: "Europe/Berlin" });

    expect(reached.map((milestone) => milestone.id)).toEqual(["first-take"]);
  });

  it("crosses word and minute totals on the day they cross", () => {
    const events = flat([
      take("t1", "2026-09-01T10:00:00Z", 600, { seconds: 1_200 }),
      take("t2", "2026-09-02T10:00:00Z", 600, { seconds: 1_200 }), // 1200 words, 40 min
      take("t3", "2026-09-03T10:00:00Z", 10, { seconds: 1_500 }), // 1210 words, 65 min
    ]);

    const reached = milestones(events, { timezone: "Europe/Berlin" });

    expect(reached.find((milestone) => milestone.id === "words-1000")?.achievedOnDay).toBe(
      "2026-09-02",
    );
    expect(reached.find((milestone) => milestone.id === "minutes-60")?.achievedOnDay).toBe(
      "2026-09-03",
    );
  });

  it("yields nothing for an empty log", () => {
    expect(milestones([], { timezone: "Europe/Berlin" })).toEqual([]);
  });
});

describe("weeklyGoalProgress", () => {
  const NOW = Date.parse("2026-09-21T12:00:00Z");

  it("counts only the trailing seven days and states progress as counts", () => {
    const events = flat([
      take("t1", "2026-09-20T10:00:00Z", 312),
      take("t2", "2026-09-10T10:00:00Z", 400), // outside the window
    ]);

    const progress = weeklyGoalProgress(events, { goalWordsPerWeek: 500, now: NOW });

    expect(progress.thisWeekWords).toBe(312);
    expect(progress.achieved).toBe(false);
    expect(progress.description).toBe("312 of 500 words recognized from speech in the last 7 days");
  });

  it("marks a reached goal without any evaluative language", () => {
    const events = flat([take("t1", "2026-09-20T10:00:00Z", 600)]);
    const progress = weeklyGoalProgress(events, { goalWordsPerWeek: 500, now: NOW });

    expect(progress.achieved).toBe(true);
    expect(progress.description).not.toMatch(/bad|fail|behind|great|excellent/i);
  });
});

describe("streaks", () => {
  const TZ = "Europe/Berlin";

  it("counts the current run ending today or yesterday", () => {
    // Three consecutive days ending yesterday, viewed at today noon.
    const events = flat([
      take("t1", "2026-09-18T10:00:00Z", 5),
      take("t2", "2026-09-19T10:00:00Z", 5),
      take("t3", "2026-09-20T10:00:00Z", 5),
    ]);

    const view = streaks(events, { timezone: TZ, now: Date.parse("2026-09-21T12:00:00Z") });

    expect(view.currentDays).toBe(3);
    expect(view.longestDays).toBe(3);
  });

  it("does not break the run just because today has no take yet", () => {
    const events = flat([
      take("t1", "2026-09-20T08:00:00Z", 5),
      take("t2", "2026-09-21T07:00:00Z", 5), // today, Berlin
    ]);

    const atEvening = streaks(events, { timezone: TZ, now: Date.parse("2026-09-21T18:00:00Z") });

    expect(atEvening.currentDays).toBe(2);

    const nextDay = streaks(events, { timezone: TZ, now: Date.parse("2026-09-22T09:00:00Z") });

    expect(nextDay.currentDays).toBe(2); // the run through yesterday simply shrank, it did not fail
  });

  it("reports the longest run over a gap", () => {
    const events = flat([
      take("t1", "2026-09-14T10:00:00Z", 5),
      take("t2", "2026-09-15T10:00:00Z", 5),
      take("t3", "2026-09-20T10:00:00Z", 5),
    ]);

    const view = streaks(events, { timezone: TZ, now: Date.parse("2026-09-21T12:00:00Z") });

    expect(view.longestDays).toBe(2);
    expect(view.currentDays).toBe(1);
  });

  it("drops a day whose only takes were deleted from the run", () => {
    // The deletion guarantee these surfaces rest on: a day with no live
    // takes must not keep a streak alive. Here today's only take is
    // deleted, so the run ends yesterday — the deleted day never counted.
    const events = flat([
      take("t1", "2026-09-20T10:00:00Z", 5),
      take("t2", "2026-09-21T07:00:00Z", 5), // 09:00 Berlin, today
      deleted("t2", "2026-09-21T09:00:00Z"),
    ]);

    const view = streaks(events, { timezone: TZ, now: Date.parse("2026-09-21T12:00:00Z") });

    expect(view.currentDays).toBe(1); // only the 20th survives
    expect(view.longestDays).toBe(1);
  });
});

describe("milestonePanel consent gating", () => {
  const events = flat([take("t1", "2026-09-20T10:00:00Z", 312)]);
  const NOW = Date.parse("2026-09-21T12:00:00Z");

  it("stays entirely off without the opt-in", () => {
    const panel = milestonePanel(events, DEFAULT_INSIGHT_CONSENT, {
      timezone: "Europe/Berlin",
      now: NOW,
    });

    expect(panel).toEqual({ enabled: false, milestones: [], goal: null, streak: null });
  });

  it("shows milestones when enabled, and the goal/streak only with their own opt-ins", () => {
    const base = { ...DEFAULT_INSIGHT_CONSENT, milestones: true };
    const onlyMilestones = milestonePanel(events, base, { timezone: "Europe/Berlin", now: NOW });

    expect(onlyMilestones.enabled).toBe(true);
    expect(onlyMilestones.milestones.length).toBeGreaterThan(0);
    expect(onlyMilestones.goal).toBeNull();
    expect(onlyMilestones.streak).toBeNull();

    const withGoal = milestonePanel(
      events,
      { ...base, weeklyGoalWords: 500 },
      {
        timezone: "Europe/Berlin",
        now: NOW,
      },
    );

    expect(withGoal.goal?.goalWordsPerWeek).toBe(500);

    const withStreak = milestonePanel(
      events,
      { ...base, streaks: true },
      {
        timezone: "Europe/Berlin",
        now: NOW,
      },
    );

    expect(withStreak.streak?.currentDays).toBe(1);
  });
});
