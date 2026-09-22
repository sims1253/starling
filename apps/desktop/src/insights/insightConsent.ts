import { Option, Schema } from "effect";

/**
 * Insights consent (E29 phase 2): the independent settings INSIGHTS.md
 * demands for everything phase 2 adds. Every flag defaults to off — none of
 * these surfaces exists for the user until they say so, and turning one off
 * never touches the others or the core dictation flow.
 *
 * The two content-derived analyses (recurring phrases, vocabulary patterns)
 * are separate grants because they are the only surfaces that need
 * transcript *contents*; everything else in Insights counts numbers the
 * E28 events already carry. Milestones, streaks and the weekly goal are
 * number-derived but optional by design — "rest days are not failures" and
 * the streak game must be a choice, never a default pressure.
 */

/** The persisted shape, decoded at the storage boundary. */
const StoredConsentSchema = Schema.Struct({
  recurringPhrases: Schema.Boolean,
  vocabularyPatterns: Schema.Boolean,
  milestones: Schema.Boolean,
  streaks: Schema.Boolean,
  weeklyGoalWords: Schema.NullOr(Schema.Number),
});

const decodeStoredConsent = Schema.decodeUnknownOption(StoredConsentSchema, {
  onExcessProperty: "ignore",
});

/** Storage surface, so tests can inject a map and the app passes localStorage. */
export interface ConsentStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

const CONSENT_KEY = "starling:insights:consent";

export interface InsightConsent {
  /** Content-derived recurring-phrase analysis. Off until explicitly granted. */
  readonly recurringPhrases: boolean;
  /** Content-derived vocabulary-pattern analysis. Off until explicitly granted. */
  readonly vocabularyPatterns: boolean;
  /** Milestone celebrations. Off by default. */
  readonly milestones: boolean;
  /** The streak counter. Off by default — the streak game is optional. */
  readonly streaks: boolean;
  /**
   * The user-chosen weekly word goal, or null for none. A goal frames
   * progress, never a shortfall: quiet weeks are not penalized anywhere.
   */
  readonly weeklyGoalWords: number | null;
}

export const DEFAULT_INSIGHT_CONSENT: InsightConsent = Object.freeze({
  recurringPhrases: false,
  vocabularyPatterns: false,
  milestones: false,
  streaks: false,
  weeklyGoalWords: null,
});

export const WEEKLY_GOAL_MIN = 1;

export const WEEKLY_GOAL_MAX = 1_000_000;

/**
 * Read the consent state; anything unreadable, absent or malformed is the
 * default (everything off) rather than an error — consent never blocks the
 * app, and a damaged value must not read as granted.
 */
export function readInsightConsent(storage: ConsentStorage): InsightConsent {
  const raw = storage.getItem(CONSENT_KEY);

  if (raw === null) return DEFAULT_INSIGHT_CONSENT;

  // The stored value is untrusted input like any other storage record; the
  // schema is the boundary, and anything it does not accept — missing
  // fields, wrong types, a non-object — is the all-off default rather than
  // an error. A damaged value must never read as granted.
  let parsed: unknown;

  try {
    parsed = JSON.parse(raw);
  } catch {
    return DEFAULT_INSIGHT_CONSENT;
  }

  const decoded = decodeStoredConsent(parsed);

  if (Option.isNone(decoded)) return DEFAULT_INSIGHT_CONSENT;

  const stored = decoded.value;

  return Object.freeze({
    recurringPhrases: stored.recurringPhrases,
    vocabularyPatterns: stored.vocabularyPatterns,
    milestones: stored.milestones,
    streaks: stored.streaks,
    weeklyGoalWords: readableGoal(stored.weeklyGoalWords),
  });
}

function readableGoal(value: number | null): number | null {
  if (value === null || !Number.isInteger(value)) return null;

  return value >= WEEKLY_GOAL_MIN && value <= WEEKLY_GOAL_MAX ? value : null;
}

/**
 * Parse a weekly-goal draft from the input field: a whole number inside the
 * declared bounds, or null. The draft is committed on blur, not per
 * keystroke, so half-typed values ("1e" on the way to "1e3" is out of bounds
 * anyway, "-" on the way to nothing) parse to null instead of rewriting the
 * field under the user's cursor — and the bounds this states are the same
 * ones `readableGoal` enforces on stored values.
 */
export function parseWeeklyGoal(value: string): number | null {
  const trimmed = value.trim();

  if (!/^\d+$/.test(trimmed)) return null;

  const parsed = Number.parseInt(trimmed, 10);

  return parsed >= WEEKLY_GOAL_MIN && parsed <= WEEKLY_GOAL_MAX ? parsed : null;
}

/** Persist the consent state as one whole object — grants never half-land. */
export function writeInsightConsent(storage: ConsentStorage, consent: InsightConsent): void {
  storage.setItem(
    CONSENT_KEY,
    JSON.stringify({
      recurringPhrases: consent.recurringPhrases,
      vocabularyPatterns: consent.vocabularyPatterns,
      milestones: consent.milestones,
      streaks: consent.streaks,
      weeklyGoalWords: consent.weeklyGoalWords,
    }),
  );
}

/** Which content-derived kinds a consent state allows to be *retained*. */
export function consentedTermKinds(consent: InsightConsent): ReadonlySet<"terms" | "phrases"> {
  const kinds = new Set<"terms" | "phrases">();

  if (consent.vocabularyPatterns) kinds.add("terms");

  if (consent.recurringPhrases) kinds.add("phrases");

  return kinds;
}
