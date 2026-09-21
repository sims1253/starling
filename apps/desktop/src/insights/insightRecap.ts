import type { InsightEvent } from "./insightEvents";
import { aggregate, parseInsightTimestamp } from "./insightMetrics";

/**
 * The weekly recap (E29 phase 2), derived entirely from the existing
 * numerical metrics — offline, on the local event log, with no new event
 * kind and no content-derived data. The comparison window rule is
 * INSIGHTS.md's matched-period discipline: this week and last week are two
 * consecutive seven-day windows of the same instant length (so a DST
 * transition cannot make one window shorter), the delta is stated as a
 * count with its direction and never as praise or a shortfall, and a week
 * without takes is "not enough data", not zero effort.
 */

const DAY_MS = 24 * 60 * 60 * 1000;

const WEEK_DAYS = 7;

export interface RecapWeek {
  readonly words: number;
  readonly takes: number;
  readonly capturedSeconds: number;
}

export interface WeeklyRecap {
  /** Local day keys of this week's window (declared reporting timezone). */
  readonly startDay: string;
  readonly endDay: string;
  readonly thisWeek: RecapWeek;
  /** Null when the previous window has no takes at all — nothing to compare. */
  readonly previousWeek: RecapWeek | null;
  /** Signed word delta (this − previous); null when there is no comparison. */
  readonly changeWords: number | null;
  /** The recap's grounded lines, ready to render. */
  readonly lines: readonly string[];
}

export type WeeklyRecapResult =
  | { readonly ok: true; readonly recap: WeeklyRecap }
  | { readonly ok: false; readonly reason: string };

export interface RecapOptions {
  readonly timezone: string;
  /** Injectable clock for deterministic tests (epoch ms). */
  readonly now?: number;
}

function weekTotals(events: readonly InsightEvent[], sinceMs: number, untilMs: number): RecapWeek {
  const within = events.filter((event) => {
    const at = parseInsightTimestamp(event.occurred_at);

    return at >= sinceMs && at < untilMs;
  });

  const totals = aggregate(within);

  return {
    words: totals.recognized_words,
    takes: totals.unique_takes,
    capturedSeconds: totals.captured_seconds,
  };
}

function formatMinutes(seconds: number): string {
  return `${(seconds / 60).toFixed(1)} min`;
}

function localDayKey(ms: number, timezone: string): string {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: timezone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(ms);
}

/**
 * Compute the weekly recap. Throws nothing the caller must handle beyond
 * the event contract's own structural errors; a week without takes is the
 * honest not-enough-data result, and the comparison line appears only when
 * last week actually had takes.
 */
export function weeklyRecap(
  events: readonly InsightEvent[],
  options: RecapOptions,
): WeeklyRecapResult {
  const now = options.now ?? Date.now();
  const thisSince = now - WEEK_DAYS * DAY_MS;
  const previousSince = now - 2 * WEEK_DAYS * DAY_MS;

  const thisWeek = weekTotals(events, thisSince, now);
  const previousWeekTotals = weekTotals(events, previousSince, thisSince);

  if (thisWeek.takes === 0) {
    return {
      ok: false,
      reason: "no takes in the last seven days — a quiet week is stated, not scored",
    };
  }

  const previousWeek = previousWeekTotals.takes > 0 ? previousWeekTotals : null;

  const changeWords = previousWeek === null ? null : thisWeek.words - previousWeek.words;

  const lines = [
    `This week: ${thisWeek.words} words recognized from speech · ${thisWeek.takes} takes · ${formatMinutes(thisWeek.capturedSeconds)} captured`,
  ];

  if (previousWeek === null) {
    lines.push("No takes in the week before — this is the first week with takes in range.");
  } else {
    const delta = thisWeek.words - previousWeek.words;

    const direction =
      delta === 0 ? "the same as" : delta > 0 ? `${delta} more than` : `${-delta} fewer than`;

    lines.push(
      `Last week: ${previousWeek.words} words · ${previousWeek.takes} takes · ${formatMinutes(previousWeek.capturedSeconds)} captured`,
    );
    lines.push(`Words this week were ${direction} last week.`);
  }

  return {
    ok: true,
    recap: {
      startDay: localDayKey(thisSince, options.timezone),
      endDay: localDayKey(now - 1, options.timezone),
      thisWeek,
      previousWeek,
      changeWords,
      lines: Object.freeze(lines),
    },
  };
}
