import { isRecognitionSelected, type InsightEvent } from "./insightEvents";
import { parseInsightTimestamp } from "./insightMetrics";
import type { InsightTermRecord } from "./insightTerms";

/**
 * The "Your voice" cards (E29 phase 2): recurring phrases and recurring
 * vocabulary, derived exclusively from the consented term aggregates — never
 * from transcripts, which this module never sees. Grounding is the whole
 * product discipline here (INSIGHTS.md): a card may only claim what its
 * counts support, every description cites those counts, and nothing
 * infers traits, moods or productivity. A playful label is not a
 * psychological assessment, so none is attempted.
 */

export type VoiceCardKind = "recurring_phrase" | "vocabulary_term";

export interface VoicePatternCard {
  readonly kind: VoiceCardKind;
  /** The phrase or term itself — content-derived, shown only under its grant. */
  readonly label: string;
  /** Distinct takes in the window whose transcript contained the label. */
  readonly takes: number;
  /** Total occurrences across those takes. */
  readonly occurrences: number;
  /** Takes in the window at all — the denominator a card's claim is made against. */
  readonly windowTakes: number;
  readonly windowDays: number;
  /** The grounded claim, stated only in terms of the counts above. */
  readonly description: string;
}

export interface VoiceCardOptions {
  /** Window length in days; only records inside it ground cards. */
  readonly windowDays?: number;
  /** Distinct-take threshold; below this a label is not "recurring". */
  readonly minTakes?: number;
  /** How many cards per kind, most recurring first. */
  readonly limit?: number;
  /** User exclusions: labels never shown and never counted into descriptions. */
  readonly exclude?: ReadonlySet<string>;
  /** Injectable clock for deterministic tests (epoch ms). */
  readonly now?: number;
  /**
   * Vocabulary cards count only terms at least this long (code units);
   * shorter tokens are overwhelmingly function words. Declared filter,
   * stated in the card's basis line.
   */
  readonly minTermLength?: number;
}

const DEFAULT_WINDOW_DAYS = 14;

const DEFAULT_MIN_TAKES = 2;

const DEFAULT_LIMIT = 5;

const DEFAULT_MIN_TERM_LENGTH = 4;

interface WindowedTotals {
  readonly takes: number;
  readonly byLabel: Map<string, { takes: number; occurrences: number }>;
}

/**
 * Aggregate labels over the records inside the time window. A record counts
 * a label once per take no matter how often it repeats inside that take, so
 * "3 of 8 takes" stays a statement about takes, and the occurrence total is
 * carried separately and never presented as takes.
 */
function windowTotals(
  records: readonly InsightTermRecord[],
  labelsOf: (
    record: InsightTermRecord,
  ) => readonly { readonly text: string; readonly count: number }[],
  options: Required<Pick<VoiceCardOptions, "windowDays" | "now">>,
): WindowedTotals {
  const since = options.now - options.windowDays * 24 * 60 * 60 * 1000;
  const byLabel = new Map<string, { takes: number; occurrences: number }>();
  let takes = 0;

  for (const record of records) {
    if (parseInsightTimestamp(record.occurred_at) < since) continue;

    takes += 1;

    for (const entry of labelsOf(record)) {
      const tally = byLabel.get(entry.text) ?? { takes: 0, occurrences: 0 };

      tally.takes += 1;
      tally.occurrences += entry.count;
      byLabel.set(entry.text, tally);
    }
  }

  return { takes, byLabel };
}

function buildCards(
  kind: VoiceCardKind,
  totals: WindowedTotals,
  options: Required<VoiceCardOptions>,
): readonly VoicePatternCard[] {
  const cards: VoicePatternCard[] = [];

  for (const [label, tally] of totals.byLabel) {
    if (options.exclude.has(label)) continue;

    if (tally.takes < options.minTakes) continue;

    if (kind === "vocabulary_term" && label.length < options.minTermLength) continue;

    cards.push({
      kind,
      label,
      takes: tally.takes,
      occurrences: tally.occurrences,
      windowTakes: totals.takes,
      windowDays: options.windowDays,
      description: groundedDescription(kind, label, tally, totals.takes, options),
    });
  }

  // Most takes first, then most occurrences, then label — a total order, so
  // the card list is deterministic for tests and screenshots alike.
  cards.sort(
    (left, right) =>
      right.takes - left.takes ||
      right.occurrences - left.occurrences ||
      (left.label < right.label ? -1 : left.label > right.label ? 1 : 0),
  );

  return cards.slice(0, options.limit);
}

function groundedDescription(
  kind: VoiceCardKind,
  label: string,
  tally: { takes: number; occurrences: number },
  windowTakes: number,
  options: Required<VoiceCardOptions>,
): string {
  const repeated = tally.occurrences > tally.takes ? `, ${tally.occurrences} times` : "";

  const basis =
    kind === "vocabulary_term"
      ? `; terms under ${options.minTermLength} characters are not counted`
      : "";

  return `"${label}" appeared in ${tally.takes} of ${windowTakes} analyzed takes${repeated} in the last ${options.windowDays} days${basis}`;
}

/** Recurring-phrase cards: two- and three-word phrases across takes. */
export function recurringPhraseCards(
  records: readonly InsightTermRecord[],
  options: VoiceCardOptions = {},
): readonly VoicePatternCard[] {
  const resolved = resolveOptions(options);

  return buildCards(
    "recurring_phrase",
    windowTotals(records, (record) => record.phrases, resolved),
    resolved,
  );
}

/** Vocabulary-pattern cards: recurring word-like terms across takes. */
export function vocabularyPatternCards(
  records: readonly InsightTermRecord[],
  options: VoiceCardOptions = {},
): readonly VoicePatternCard[] {
  const resolved = resolveOptions(options);

  return buildCards(
    "vocabulary_term",
    windowTotals(records, (record) => record.terms, resolved),
    resolved,
  );
}

function resolveOptions(options: VoiceCardOptions): Required<VoiceCardOptions> {
  return {
    windowDays: options.windowDays ?? DEFAULT_WINDOW_DAYS,
    minTakes: options.minTakes ?? DEFAULT_MIN_TAKES,
    limit: options.limit ?? DEFAULT_LIMIT,
    exclude: options.exclude ?? new Set<string>(),
    now: options.now ?? Date.now(),
    minTermLength: options.minTermLength ?? DEFAULT_MIN_TERM_LENGTH,
  };
}

/**
 * Both card kinds under their independent consents: a kind whose grant is
 * off yields no cards even when its aggregates exist (and after a
 * withdrawal plus purge, none exist to yield).
 *
 * Two denominators, both stated: `windowTakes` is the distinct captures
 * with a selected recognition inside the window (a retranscription is the
 * same take, never counted twice), and `analyzedTakes` is how many of those
 * have retained aggregates — takes recorded while a grant was off have
 * none. Cards cite `analyzedTakes` ("appeared in 3 of 5 analyzed takes")
 * because a take with no aggregates is an unknown, not a negative: claiming
 * "3 of 8 takes" would assert the phrase is absent from takes nobody
 * analyzed. The UI shows both numbers so the coverage is visible.
 */
export interface VoicePanel {
  readonly phraseCards: readonly VoicePatternCard[];
  readonly vocabularyCards: readonly VoicePatternCard[];
  /** Distinct takes with a selected recognition inside the window. */
  readonly windowTakes: number;
  /** Takes in the window whose aggregates exist under the current grants. */
  readonly analyzedTakes: number;
}

export function voicePanel(
  events: readonly InsightEvent[],
  records: readonly InsightTermRecord[],
  consent: { readonly recurringPhrases: boolean; readonly vocabularyPatterns: boolean },
  options: VoiceCardOptions = {},
): VoicePanel {
  const resolved = resolveOptions(options);

  const since = resolved.now - resolved.windowDays * 24 * 60 * 60 * 1000;

  const windowCaptures = new Set<string>();

  for (const event of events) {
    if (isRecognitionSelected(event) && parseInsightTimestamp(event.occurred_at) >= since) {
      windowCaptures.add(event.capture_id);
    }
  }

  const analyzedTakes = records.filter((record) => windowCaptures.has(record.capture_id)).length;

  return {
    phraseCards: consent.recurringPhrases ? recurringPhraseCards(records, resolved) : [],
    vocabularyCards: consent.vocabularyPatterns ? vocabularyPatternCards(records, resolved) : [],
    windowTakes: windowCaptures.size,
    analyzedTakes,
  };
}
