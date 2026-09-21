import { describe, expect, it } from "vite-plus/test";

import { recurringPhraseCards } from "./insightVoice";
import {
  DEFAULT_SHARE_INCLUDES,
  SHARE_CARD_TITLE,
  buildShareCard,
  milestoneInRange,
  renderShareCard,
  type ShareCardField,
} from "./insightShare";

/**
 * Share card coverage (E29 phase 2): the default card carries aggregate
 * words/minutes/milestone only; a content-derived field is withheld unless
 * explicitly included on this exact card; the rendered text states its
 * date range and definitions; and rendering is a pure local function —
 * the module has no clipboard, network or post API to call.
 */

const TOP_PHRASE = recurringPhraseCards(
  [
    {
      schema_version: 1,
      capture_id: "t1",
      occurred_at: "2026-09-20T10:00:00Z",
      tokenizer: "uax29-intl-v1",
      terms: [],
      phrases: [{ text: "deploy the server", count: 3 }],
    },
    {
      schema_version: 1,
      capture_id: "t2",
      occurred_at: "2026-09-20T11:00:00Z",
      tokenizer: "uax29-intl-v1",
      terms: [],
      phrases: [{ text: "deploy the server", count: 1 }],
    },
  ],
  { now: Date.parse("2026-09-21T12:00:00Z") },
).find((card) => card.label === "deploy the server");

const INPUT = {
  rangeStartDay: "2026-09-14",
  rangeEndDay: "2026-09-21",
  words: 312,
  minutes: 4.2,
  takes: 6,
  milestone: "First week of voice notes",
  topPhrase: TOP_PHRASE,
};

describe("buildShareCard", () => {
  it("defaults to the aggregate words/minutes/milestone only", () => {
    const card = buildShareCard(INPUT);

    expect(card.title).toBe(SHARE_CARD_TITLE);
    expect(card.lines).toEqual([
      "312 words recognized from speech",
      "4.2 min captured",
      "Milestone: First week of voice notes",
    ]);
    // takes exists in the input but was not included; the phrase was withheld.
    expect(card.withheld).toContain("takes");
    expect(card.withheld).toContain("topPhrase");
    expect(card.lines.every((line) => !line.includes("deploy"))).toBe(true);
  });

  it("includes the content-derived phrase only on an explicit opt-in", () => {
    const include = new Set<ShareCardField>([...DEFAULT_SHARE_INCLUDES, "topPhrase"]);
    const card = buildShareCard(INPUT, { include });

    const phraseLine = card.lines.find((line) => line.startsWith("Most repeated phrase"));

    expect(phraseLine).toBe('Most repeated phrase: "deploy the server" (2 takes)');
    expect(card.withheld).not.toContain("topPhrase");
  });

  it("never emits a milestone line for a milestone that does not exist", () => {
    const card = buildShareCard({ ...INPUT, milestone: undefined });

    expect(card.lines.some((line) => line.startsWith("Milestone"))).toBe(false);
    expect(card.withheld).not.toContain("milestone");
  });

  it("omits the phrase even when explicitly named but absent from the input", () => {
    const include = new Set<ShareCardField>([...DEFAULT_SHARE_INCLUDES, "topPhrase"]);
    const card = buildShareCard({ ...INPUT, topPhrase: undefined }, { include });

    expect(card.lines.some((line) => line.includes("Most repeated"))).toBe(false);
  });
});

describe("renderShareCard", () => {
  it("renders title, date range, lines and definitions — nothing about withholdings", () => {
    const text = renderShareCard(buildShareCard(INPUT));

    expect(text.startsWith(SHARE_CARD_TITLE)).toBe(true);
    expect(text).toContain("2026-09-14 – 2026-09-21");
    expect(text).toContain("312 words recognized from speech");
    expect(text).toContain("silence included");
    expect(text).not.toContain("deploy the server");
    // The withheld list is preview-only metadata: the copied/saved artifact
    // must not disclose that content-derived data exists and was withheld.
    expect(text).not.toContain("Withheld");
  });

  it("keeps the withheld list on the card for the preview to show", () => {
    const card = buildShareCard(INPUT);

    expect(card.withheld).toContain("takes");
    expect(card.withheld).toContain("topPhrase");
  });

  it("is deterministic for the same card", () => {
    const card = buildShareCard(INPUT);

    expect(renderShareCard(card)).toBe(renderShareCard(card));
  });
});

describe("milestoneInRange", () => {
  const milestones = [
    { label: "First voice note", achievedOnDay: "2026-08-02" },
    { label: "1,000 words recognized from speech", achievedOnDay: "2026-09-16" },
  ];

  it("cites the latest milestone achieved inside the card's range", () => {
    expect(milestoneInRange(milestones, "2026-09-14")).toBe("1,000 words recognized from speech");
  });

  it("states no milestone when every one predates the range", () => {
    // A milestone from months ago must not ride on a card labeled "this week".
    expect(milestoneInRange(milestones, "2026-09-20")).toBeUndefined();
    expect(milestoneInRange([], "2026-09-14")).toBeUndefined();
  });
});
