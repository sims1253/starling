import { Predicate } from "effect";

/**
 * The voice-card exclusion list's storage boundary (E29 phase 2). The stored
 * value is untrusted input like any other local-settings record, so the read
 * is a schema-shaped boundary the consent module would recognize, not a bare
 * JSON.parse: only strings that could be card labels survive, and only a
 * bounded number of them. A corrupted or hostile value (thousands of
 * entries, free text, nested objects) degrades to what fits the contract,
 * never to unbounded in-memory state. The write degrades the other way
 * round: when storage refuses (quota, privacy mode), the exclusion stays
 * active for the session and the caller says so — a settings write must
 * never crash the interaction it was part of.
 */

/** Storage surface, so tests can inject a map and the app passes localStorage. */
export interface ExclusionStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

const EXCLUSIONS_KEY = "starling:insights:exclusions";

/**
 * A label is one card's phrase or term: at most three whitespace-free
 * tokens of at most 64 code units each joined by single spaces — the same
 * shape the term record's schema admits, so anything else (double spaces,
 * tabs, newlines, longer runs) is refused at this boundary, not merely
 * bounded.
 */
const LABEL_PATTERN = /^(?:[^\p{White_Space}\p{C}]{1,64})(?: [^\p{White_Space}\p{C}]{1,64}){0,2}$/u;

/** The longest label the boundary admits: three 64-code-unit tokens. */
export const MAX_EXCLUSION_LABEL_LENGTH = 3 * 64 + 2;

/** Years of excluded labels; more is damage, not preference. */
export const MAX_EXCLUSIONS = 256;

/** A stored entry is a label only if it fits the shape the cards can show. */
export function isExclusionLabel(value: string): boolean {
  return value.length <= MAX_EXCLUSION_LABEL_LENGTH && LABEL_PATTERN.test(value);
}

/** Read the exclusion set; anything unreadable or over the bounds shrinks. */
export function readExclusions(storage: ExclusionStorage): ReadonlySet<string> {
  let parsed: unknown;

  try {
    parsed = JSON.parse(storage.getItem(EXCLUSIONS_KEY) ?? "[]");
  } catch {
    return new Set();
  }

  if (!Array.isArray(parsed)) return new Set();

  return new Set(
    parsed.filter(Predicate.isString).filter(isExclusionLabel).slice(0, MAX_EXCLUSIONS),
  );
}

/**
 * Persist the whole exclusion set. False means storage refused the write and
 * the exclusions are session-only — the caller surfaces that, because a
 * silent refusal reads as "never shown again" while actually lasting until
 * the app closes.
 */
export function writeExclusions(storage: ExclusionStorage, labels: ReadonlySet<string>): boolean {
  try {
    storage.setItem(EXCLUSIONS_KEY, JSON.stringify([...labels]));

    return true;
  } catch {
    return false;
  }
}
