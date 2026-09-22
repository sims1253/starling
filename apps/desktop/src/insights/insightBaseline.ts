/**
 * The typing baseline's storage boundary (E29 phase 2), the same discipline
 * the exclusion list and consent records use: the stored value is untrusted
 * input like any other local-settings record, so the read is bounded and
 * validated rather than a bare getItem — only a draft that could be part of
 * a whole-number WPM survives, and anything else (free text, a pasted novel)
 * degrades to the unset baseline, never to unbounded in-memory state. The
 * write degrades the other way round: when storage refuses (quota, privacy
 * mode) or the value is too large to be a draft at all, the baseline holds
 * for this session only and the caller says so — a settings write must
 * never crash the interaction it was part of.
 */

/** Storage surface, so tests can inject a map and the app passes localStorage. */
export interface BaselineStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

const TYPING_BASELINE_KEY = "starling:insights:typingWpm";

/**
 * The longest draft the boundary admits. A real WPM is at most a few digits;
 * the bound exists so a pasted novel cannot occupy the settings entry.
 */
export const MAX_BASELINE_DRAFT_LENGTH = 16;

/**
 * A stored draft is admissible only when it could be part of a whole-number
 * WPM: digits with an optional leading minus (a mid-typing "-3"), including
 * the empty string. Anything else reads as unset — the field's own
 * validation already explains what a baseline needs to be.
 */
const BASELINE_DRAFT_PATTERN = /^-?\d{0,9}$/;

/** Whether a draft could persist as a baseline: the shape and the bound. */
export function isBaselineDraft(value: string): boolean {
  return value.length <= MAX_BASELINE_DRAFT_LENGTH && BASELINE_DRAFT_PATTERN.test(value);
}

/** Read the stored draft; anything unreadable or out of shape reads unset. */
export function readTypingBaseline(storage: BaselineStorage): string {
  const stored = storage.getItem(TYPING_BASELINE_KEY);

  return stored !== null && isBaselineDraft(stored) ? stored : "";
}

/**
 * Persist the draft. False means it did not land — storage refused, or the
 * value cannot be a baseline draft at all — and the baseline is session-only;
 * the caller surfaces that, because a silent refusal reads as "saved" until
 * the app closes.
 */
export function writeTypingBaseline(storage: BaselineStorage, value: string): boolean {
  if (!isBaselineDraft(value)) return false;

  try {
    storage.setItem(TYPING_BASELINE_KEY, value);

    return true;
  } catch {
    return false;
  }
}
