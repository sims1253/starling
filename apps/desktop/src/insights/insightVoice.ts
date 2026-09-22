import { isCaptureDeleted, isRecognitionSelected, type InsightEvent } from "./insightEvents";
import { parseInsightTimestamp } from "./insightMetrics";
import { analyzedKinds, type InsightTermKind, type InsightTermRecord } from "./insightTerms";

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
  /**
   * Captures the event log says were deleted. Their records count nowhere —
   * not in the per-kind denominators, not in the labels — even while the
   * term store's own deletion is still in flight or has failed with a
   * notice; the event tombstone dominates, as it does in every other
   * surface. `voicePanel` derives this from the events it is given.
   */
  readonly excludeCaptures?: ReadonlySet<string>;
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
 *
 * The `takes` denominator is per-kind and provenance-based: only records
 * `analyzedKinds` — the one shared predicate, exported from insightTerms.ts
 * so every denominator derives from it — marks analyzed for THIS kind count.
 * For a stamped record the `derived_kinds` stamp is the truth (a kind
 * stamped with an empty label list is a real zero: analyzed, nothing
 * found); an unstamped record falls back to label evidence. A take recorded
 * under a narrower grant — or emptied by a withdrawal purge — is an unknown
 * for these cards, not a negative, so it must not inflate the denominator;
 * a take that analyzed and found nothing of the kind is a real zero and
 * must.
 */
function windowTotals(
  records: readonly InsightTermRecord[],
  kind: InsightTermKind,
  options: Required<Pick<VoiceCardOptions, "windowDays" | "now" | "excludeCaptures">>,
): WindowedTotals {
  const since = options.now - options.windowDays * 24 * 60 * 60 * 1000;
  const byLabel = new Map<string, { takes: number; occurrences: number }>();
  let takes = 0;

  for (const record of records) {
    if (parseInsightTimestamp(record.occurred_at) < since) continue;

    // The event log's tombstone dominates the records the term store has
    // not caught up on: a deleted capture counts nowhere, so the per-kind
    // denominator and windowTakes cannot disagree about it.
    if (options.excludeCaptures.has(record.capture_id)) continue;

    // Only analyzed-for-this-kind records contribute — to the denominator
    // AND to the labels. Counting labels from an unanalyzed record could
    // push a card's numerator past its own denominator ("N of M" with
    // N > M); an unanalyzed take is an unknown for both.
    if (!analyzedKinds(record)[kind]) continue;

    takes += 1;

    for (const entry of kind === "terms" ? record.terms : record.phrases) {
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

  return buildCards("recurring_phrase", windowTotals(records, "phrases", resolved), resolved);
}

/** Vocabulary-pattern cards: recurring word-like terms across takes. */
export function vocabularyPatternCards(
  records: readonly InsightTermRecord[],
  options: VoiceCardOptions = {},
): readonly VoicePatternCard[] {
  const resolved = resolveOptions(options);

  return buildCards("vocabulary_term", windowTotals(records, "terms", resolved), resolved);
}

function resolveOptions(options: VoiceCardOptions): Required<VoiceCardOptions> {
  return {
    windowDays: options.windowDays ?? DEFAULT_WINDOW_DAYS,
    minTakes: options.minTakes ?? DEFAULT_MIN_TAKES,
    limit: options.limit ?? DEFAULT_LIMIT,
    exclude: options.exclude ?? new Set<string>(),
    excludeCaptures: options.excludeCaptures ?? new Set<string>(),
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
 * same take, never counted twice, and a deleted take drops out — the
 * tombstone removes it here exactly like the metric contract removes it
 * from every other surface), and `analyzedTakes` is how many of those
 * retained aggregates under the current grants. Cards cite their own
 * per-kind analyzed count ("appeared in 3 of 5 analyzed takes" where 5
 * counts only takes stamped as analyzed for that card's kind — a take
 * that analyzed and found nothing counts, a purged, never-granted or
 * legacy-unstamped take is an unknown) because an unknown is not a
 * negative: claiming "3 of 8 takes" would assert the phrase is absent
 * from takes nobody analyzed. The UI shows both numbers so the coverage
 * is visible.
 */
export interface VoicePanel {
  readonly phraseCards: readonly VoicePatternCard[];
  readonly vocabularyCards: readonly VoicePatternCard[];
  /** Distinct, non-deleted takes with a selected recognition in the window. */
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
  // The metric contract's tombstone rule, computed first so the resolved
  // options can carry it to the card denominators too: any capture_deleted
  // removes the capture's events regardless of arrival order, and its term
  // record counts nowhere even while the term store's own deletion is
  // still catching up.
  const tombstonedCaptures = new Set<string>();

  for (const event of events) {
    if (isCaptureDeleted(event)) tombstonedCaptures.add(event.capture_id);
  }

  const resolved = resolveOptions({
    ...options,
    excludeCaptures: options.excludeCaptures ?? tombstonedCaptures,
  });

  const since = resolved.now - resolved.windowDays * 24 * 60 * 60 * 1000;

  const windowCaptures = new Set<string>();

  for (const event of events) {
    if (
      isRecognitionSelected(event) &&
      parseInsightTimestamp(event.occurred_at) >= since &&
      !tombstonedCaptures.has(event.capture_id)
    ) {
      windowCaptures.add(event.capture_id);
    }
  }

  // Any-kind analyzed count: a take counts when the shared predicate says
  // it was analyzed for at least one kind — stamped non-empty, or unstamped
  // with surviving labels (legacy evidence) — so the panel denominator and
  // the per-kind card denominators can never disagree about provenance.
  const analyzedTakes = records.filter((record) => {
    if (!windowCaptures.has(record.capture_id)) return false;

    const kinds = analyzedKinds(record);

    return kinds.terms || kinds.phrases;
  }).length;

  return {
    phraseCards: consent.recurringPhrases ? recurringPhraseCards(records, resolved) : [],
    vocabularyCards: consent.vocabularyPatterns ? vocabularyPatternCards(records, resolved) : [],
    windowTakes: windowCaptures.size,
    analyzedTakes,
  };
}
