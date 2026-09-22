import type { VoicePatternCard } from "./insightVoice";
import { formatMinutes } from "./insightFormat";

/**
 * Share cards (E29 phase 2): a local artifact the user previews before
 * anything leaves their hands. The discipline is fixed by INSIGHTS.md: the
 * default card carries aggregate words, minutes and a milestone only — no
 * app names, no document names, no actual phrases — and content-derived
 * fields join solely by an explicit, per-card opt-in. Copying and saving
 * are the only actions; there is deliberately no share-to-site call,
 * auto-post hook or network path anywhere in this module, and the card
 * records its date range and definitions so a reader can see what the
 * numbers mean.
 */

/** The fields a card can carry. Aggregate-only fields are defaultable. */
export const SHARE_CARD_FIELDS = ["words", "minutes", "takes", "milestone", "topPhrase"] as const;

export type ShareCardField = (typeof SHARE_CARD_FIELDS)[number];

/** The fields a content-free default card carries (INSIGHTS.md's default). */
export const DEFAULT_SHARE_INCLUDES: ReadonlySet<ShareCardField> = new Set([
  "words",
  "minutes",
  "milestone",
]);

/** Fields that can only appear through an explicit per-card opt-in. */
export const CONTENT_DERIVED_SHARE_FIELDS: ReadonlySet<ShareCardField> = new Set(["topPhrase"]);

export interface ShareCardInput {
  /** Local day keys of the card's inclusive date range. */
  readonly rangeStartDay: string;
  readonly rangeEndDay: string;
  /** Recognized speech words in the range (never generated output words). */
  readonly words: number;
  /** Captured minutes in the range, silence included. */
  readonly minutes: number;
  readonly takes: number;
  /** A milestone label worth celebrating, when one was reached. */
  readonly milestone?: string;
  /**
   * The top recurring-phrase card, when the user's consent allows one to
   * exist. It is still withheld unless "topPhrase" is explicitly included.
   */
  readonly topPhrase?: VoicePatternCard;
}

export interface ShareCardOptions {
  /**
   * The explicit include set. Absent means the default (aggregate-only);
   * a content-derived field is ignored unless present here.
   */
  readonly include?: ReadonlySet<ShareCardField>;
}

export interface ShareCard {
  readonly title: string;
  readonly rangeStartDay: string;
  readonly rangeEndDay: string;
  /** The card's body lines, ready to render — only included fields appear. */
  readonly lines: readonly string[];
  /**
   * Field names withheld by the redaction default. Preview-only metadata:
   * the preview shows it so the user knows what exists and is not leaving,
   * but it never travels with the copied or saved text — the artifact must
   * not disclose that content-derived data exists and was withheld.
   */
  readonly withheld: readonly ShareCardField[];
  /** The definitions footnote every card carries (INSIGHTS.md). */
  readonly definitions: string;
}

const DEFINITIONS =
  "Words are recognized from speech (selected final transcripts only); minutes are captured audio with silence included. Computed locally by Starling.";

export const SHARE_CARD_TITLE = "What voice made possible";

// The redaction default is structural, not conventional: no content-derived
// field may ever sit in the default include set, and this invariant fires
// at module load if one is added there — the "explicit opt-in" gate stays
// meaningful because the default can never grant what only an explicit
// include can.
for (const field of DEFAULT_SHARE_INCLUDES) {
  if (CONTENT_DERIVED_SHARE_FIELDS.has(field)) {
    throw new Error(`the default share includes must never carry ${field}`);
  }
}

/**
 * Build a share card. Redaction is the default and the structure enforces
 * it: a field reaches a line only through the include set, the default
 * include set is structurally barred from content-derived fields (above),
 * and nothing else from the input (no take ids, no app data, no free text)
 * has any path into the output at all.
 */
export function buildShareCard(input: ShareCardInput, options: ShareCardOptions = {}): ShareCard {
  const include = options.include ?? DEFAULT_SHARE_INCLUDES;
  const lines: string[] = [];

  if (include.has("words")) lines.push(`${input.words} words recognized from speech`);

  if (include.has("minutes")) lines.push(`${formatMinutes(input.minutes)} captured`);

  if (include.has("takes")) lines.push(`${input.takes} takes`);

  if (include.has("milestone") && input.milestone !== undefined) {
    lines.push(`Milestone: ${input.milestone}`);
  }

  if (include.has("topPhrase") && input.topPhrase !== undefined) {
    lines.push(`Most repeated phrase: "${input.topPhrase.label}" (${input.topPhrase.takes} takes)`);
  }

  const withheld = SHARE_CARD_FIELDS.filter(
    (field) => !include.has(field) && hasValue(input, field),
  );

  return {
    title: SHARE_CARD_TITLE,
    rangeStartDay: input.rangeStartDay,
    rangeEndDay: input.rangeEndDay,
    lines: Object.freeze(lines),
    withheld: Object.freeze(withheld),
    definitions: DEFINITIONS,
  };
}

function hasValue(input: ShareCardInput, field: ShareCardField): boolean {
  switch (field) {
    case "words":
    case "minutes":
    case "takes":
      return true;
    case "milestone":
      return input.milestone !== undefined;
    case "topPhrase":
      return input.topPhrase !== undefined;
  }
}

/** A milestone the card could cite, when it was reached. */
export interface ShareMilestone {
  readonly label: string;
  /** Local day key the milestone was achieved on. */
  readonly achievedOnDay: string;
}

/**
 * The latest milestone achieved inside the card's date range, when any. The
 * card states a seven-day range, so every claim must rest on that range:
 * a milestone from months ago is not the card's to state inside "this
 * week", and neither is one dated after the range's last day (clock skew
 * or a future-dated day key) — the card cites only what the range covers.
 * The list is oldest-first, so the last in-range entry is the latest.
 */
export function milestoneInRange(
  milestones: readonly ShareMilestone[],
  rangeStartDay: string,
  rangeEndDay: string,
): string | undefined {
  let inRange: ShareMilestone | undefined;

  for (const milestone of milestones) {
    if (milestone.achievedOnDay >= rangeStartDay && milestone.achievedOnDay <= rangeEndDay) {
      inRange = milestone;
    }
  }

  return inRange?.label;
}

/**
 * Render the card as plain text — the preview the user sees and the bytes
 * a copy or save action places on the clipboard or disk. The withheld list
 * is preview-only metadata and deliberately has no line here: what the
 * artifact discloses is exactly the fields the user included, nothing about
 * the ones they did not. Pure and local: no network primitive is reachable
 * from this module by construction.
 */
export function renderShareCard(card: ShareCard): string {
  const parts = [
    card.title,
    `${card.rangeStartDay} – ${card.rangeEndDay}`,
    ...card.lines,
    card.definitions,
  ];

  return parts.join("\n");
}
