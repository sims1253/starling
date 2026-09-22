import { isCaptureFinalized, type InsightEvent } from "./insightEvents";
import { aggregate, localDayKey, parseInsightTimestamp } from "./insightMetrics";
import { plural, pluralWord } from "./insightFormat";
import type { InsightConsent } from "./insightConsent";

/**
 * Milestones, the optional weekly goal and the optional streak (E29 phase
 * 2), all derived from the numerical E28 events — no content-derived data
 * anywhere in this module. The framing rules come straight from
 * INSIGHTS.md: milestones celebrate use without misleading precision, a
 * goal frames progress and never penalizes a quieter week, the streak game
 * is opt-in with rest days explicitly not failures, and every claim cites
 * the counts it rests on. All of it is off until the user turns it on.
 */

export interface Milestone {
  /** Stable id, so UI keys and tests survive reworded labels. */
  readonly id: string;
  readonly label: string;
  /** Local day key (declared reporting timezone) the milestone was reached. */
  readonly achievedOnDay: string;
  /** The grounded claim — dates and counts, nothing evaluative. */
  readonly evidence: string;
}

export interface StreakOptions {
  /** Injectable clock for deterministic tests (epoch ms). */
  readonly now?: number;
}

export interface GoalOptions {
  /** Injectable clock for deterministic tests (epoch ms). */
  readonly now?: number;
}

export interface MilestoneOptions extends GoalOptions, StreakOptions {
  readonly timezone: string;
}

const DAY_MS = 24 * 60 * 60 * 1000;

/** Recognized-word totals that earn a milestone, smallest first. */
const WORD_MILESTONES = [1_000, 10_000] as const;

/** Captured-minute totals that earn a milestone, smallest first. */
const MINUTE_MILESTONES = [60, 600] as const;

/**
 * Labels for the captured-minute milestones the pluralized form would
 * mangle; every other threshold reads as its plain minute count.
 */
const MINUTE_MILESTONE_LABELS = new Map<number, string>([[60, "An hour of captured audio"]]);

/** The streak's own line: the live run, or the honest absence of one. */
function streakDescription(currentDays: number, longestDays: number): string {
  if (currentDays === 0) {
    return `No consecutive take days right now (longest ever: ${longestDays})`;
  }

  return `${currentDays} consecutive ${pluralWord(currentDays, "day")} with at least one take (longest: ${longestDays})`;
}

/**
 * The log's events grouped by capture id — the store dedupes by event_id,
 * and every event kind carries the capture it is attributable to.
 */
function eventsByCapture(events: readonly InsightEvent[]): Map<string, InsightEvent[]> {
  const grouped = new Map<string, InsightEvent[]>();

  for (const event of events) {
    const items = grouped.get(event.capture_id);

    if (items === undefined) grouped.set(event.capture_id, [event]);
    else items.push(event);
  }

  return grouped;
}

interface DayTotals {
  readonly day: string;
  readonly takes: number;
  readonly recognizedWords: number;
  readonly capturedSeconds: number;
}

/**
 * Per-local-day totals over attributable captures. Each capture's numbers
 * come from the metric contract's own `aggregate` run on exactly that
 * capture's events — so tombstones, canonical-finalization and
 * highest-sequence selection apply with their frozen semantics instead of a
 * second implementation drifting from the oracle — and the day is the
 * local day of the capture's finalization instant. A capture the contract
 * counts as zero takes (tombstoned — deleted) contributes nothing and,
 * crucially, never keeps its day alive: a day whose only takes were deleted
 * drops out entirely, so streaks and milestones stop counting it exactly
 * like the metric panels do.
 */
export function dayTotals(events: readonly InsightEvent[], timezone: string): readonly DayTotals[] {
  const byDay = new Map<string, { takes: number; words: number; seconds: number }>();

  for (const items of eventsByCapture(events).values()) {
    const capture = items.find(isCaptureFinalized);

    // Orphan metadata is not evidence of a captured take (the contract's rule).
    if (capture === undefined) continue;

    const totals = aggregate(items);

    // A tombstoned capture aggregates to zero takes; it must not create or
    // extend a day bucket — deletion removes the take's contribution from
    // these surfaces too, not merely zeroes its numbers.
    if (totals.unique_takes === 0) continue;

    const day = localDayKey(parseInsightTimestamp(capture.occurred_at), timezone);
    const bucket = byDay.get(day) ?? { takes: 0, words: 0, seconds: 0 };

    bucket.takes += totals.unique_takes;
    bucket.words += totals.recognized_words;
    bucket.seconds += totals.captured_seconds;
    byDay.set(day, bucket);
  }

  return [...byDay.entries()]
    .map(([day, bucket]) => ({
      day,
      takes: bucket.takes,
      recognizedWords: bucket.words,
      capturedSeconds: bucket.seconds,
    }))
    .sort((left, right) => (left.day < right.day ? -1 : 1));
}

/**
 * The milestone list, oldest first. Deterministic and bounded: the first
 * take, the seventh distinct take-day, and the crossing days of the
 * cumulative word and captured-minute totals — each with the counts that
 * grounded it, and nothing about productivity or improvement.
 */
export function milestones(
  events: readonly InsightEvent[],
  options: MilestoneOptions,
): readonly Milestone[] {
  const days = dayTotals(events, options.timezone);
  const reached: Milestone[] = [];

  if (days.length === 0) return reached;

  const first = days[0];

  if (first === undefined) return reached;

  reached.push({
    id: "first-take",
    label: "First voice note",
    achievedOnDay: first.day,
    evidence: `The earliest local day with a retained take (${plural(first.takes, "take")}).`,
  });

  // "First week of voice notes": the seventh distinct day with a take.
  if (days.length >= 7) {
    const seventh = days[6];

    if (seventh !== undefined) {
      reached.push({
        id: "first-week",
        label: "First week of voice notes",
        achievedOnDay: seventh.day,
        evidence: "The seventh distinct day with at least one take.",
      });
    }
  }

  let words = 0;
  let seconds = 0;

  for (const day of days) {
    words += day.recognizedWords;
    seconds += day.capturedSeconds;

    for (const threshold of WORD_MILESTONES) {
      if (words >= threshold && !reached.some((item) => item.id === `words-${threshold}`)) {
        reached.push({
          id: `words-${threshold}`,
          label: `${threshold.toLocaleString()} words recognized from speech`,
          achievedOnDay: day.day,
          evidence: `Cumulative recognized words (selected final transcripts) crossed ${threshold.toLocaleString()} on this day.`,
        });
      }
    }

    for (const threshold of MINUTE_MILESTONES) {
      const minuteThreshold = threshold * 60;

      if (
        seconds >= minuteThreshold &&
        !reached.some((item) => item.id === `minutes-${threshold}`)
      ) {
        reached.push({
          id: `minutes-${threshold}`,
          label: MINUTE_MILESTONE_LABELS.get(threshold) ?? `${threshold} minutes of captured audio`,
          achievedOnDay: day.day,
          evidence: `Cumulative captured seconds (silence included) crossed ${minuteThreshold} on this day.`,
        });
      }
    }
  }

  return reached;
}

export interface GoalProgress {
  readonly goalWordsPerWeek: number;
  readonly thisWeekWords: number;
  readonly achieved: boolean;
  /** Progress stated as counts — never a shortfall judgment. */
  readonly description: string;
}

/**
 * Progress toward the user-chosen weekly word goal over the trailing
 * seven-day window (a matched period of whole days, DST-safe because the
 * window is defined in instants while the labels are local days). A quiet
 * week reads as "0 of N words", not as failure.
 */
export function weeklyGoalProgress(
  events: readonly InsightEvent[],
  options: { readonly goalWordsPerWeek: number } & GoalOptions,
): GoalProgress {
  const now = options.now ?? Date.now();

  const within = events.filter(
    (event) => parseInsightTimestamp(event.occurred_at) >= now - 7 * DAY_MS,
  );

  const words = aggregate(within).recognized_words;

  return {
    goalWordsPerWeek: options.goalWordsPerWeek,
    thisWeekWords: words,
    achieved: words >= options.goalWordsPerWeek,
    description: `${words} of ${options.goalWordsPerWeek.toLocaleString()} words recognized from speech in the last 7 days`,
  };
}

export interface StreakView {
  /** Consecutive active days ending today or yesterday. */
  readonly currentDays: number;
  readonly longestDays: number;
  readonly description: string;
}

/**
 * The opt-in streak: consecutive local days with at least one take. Today
 * counts when it has a take and does not break the run when it does not
 * yet — the day is not over — so `currentDays` runs through yesterday.
 */
export function streaks(events: readonly InsightEvent[], options: MilestoneOptions): StreakView {
  const days = new Set(dayTotals(events, options.timezone).map((day) => day.day));
  const now = options.now ?? Date.now();
  const today = localDayKey(now, options.timezone);

  // Walk back from today; a take-less today simply does not extend the run.
  let currentDays = 0;
  let cursor = today;

  for (;;) {
    if (days.has(cursor)) {
      currentDays += 1;
    } else if (cursor !== today) {
      break;
    }

    cursor = previousDayKey(cursor);
  }

  // Longest run over the sorted distinct day keys.
  const sorted = [...days].sort();
  let longestDays = 0;
  let run = 0;
  let previous: string | undefined;

  for (const day of sorted) {
    run = previous !== undefined && day === nextDayKey(previous) ? run + 1 : 1;
    longestDays = Math.max(longestDays, run);
    previous = day;
  }

  return {
    currentDays,
    longestDays,
    description: streakDescription(currentDays, longestDays),
  };
}

/** Day-key arithmetic in UTC component space; keys are already local days. */
function dayKeyToUtc(day: string): number {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(day);

  if (match === null) throw new Error(`not a local day key: ${day}`);

  return Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
}

function utcToDayKey(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}

function previousDayKey(day: string): string {
  return utcToDayKey(dayKeyToUtc(day) - DAY_MS);
}

function nextDayKey(day: string): string {
  return utcToDayKey(dayKeyToUtc(day) + DAY_MS);
}

export interface MilestonePanel {
  /** False unless the user opted in — milestones are never a default. */
  readonly enabled: boolean;
  readonly milestones: readonly Milestone[];
  readonly goal: GoalProgress | null;
  /** Null unless the user opted into the streak game. */
  readonly streak: StreakView | null;
}

/**
 * The whole optional celebratory surface under its consents: milestones and
 * the goal exist only when enabled, the streak only with its own opt-in —
 * the streak is an independent consent, so it renders whenever its toggle
 * is on, milestones or not — and the goal additionally requires the user
 * to have chosen one.
 */
export function milestonePanel(
  events: readonly InsightEvent[],
  consent: InsightConsent,
  options: MilestoneOptions,
): MilestonePanel {
  const streak = consent.streaks ? streaks(events, options) : null;

  if (!consent.milestones) {
    return { enabled: false, milestones: [], goal: null, streak };
  }

  return {
    enabled: true,
    milestones: milestones(events, options),
    goal:
      consent.weeklyGoalWords !== null
        ? weeklyGoalProgress(events, {
            goalWordsPerWeek: consent.weeklyGoalWords,
            now: options.now,
          })
        : null,
    streak,
  };
}
