import { describe, expect, it } from "vite-plus/test";

import { TOKENIZER_ID } from "./insightEmitter";
import { DEFAULT_INSIGHT_CONSENT } from "./insightConsent";
import { captureTerms, type InsightTermKind, type InsightTermRecord } from "./insightTerms";
import type { CaptureDeletedEvent, InsightEvent, RecognitionSelectedEvent } from "./insightEvents";
import {
  recurringPhraseCards,
  vocabularyPatternCards,
  voicePanel,
  type VoicePatternCard,
} from "./insightVoice";

/**
 * "Your voice" card coverage (E29 phase 2): descriptions are grounded —
 * they cite the takes and occurrences counts they rest on and nothing
 * else — each kind is gated by its own consent, exclusions remove labels,
 * windows bound the population, and deleting a take's aggregates removes
 * its contribution from every card. Per-kind denominators rest on the
 * record's derived_kinds stamp: an analyzed-but-empty take counts, a
 * purged, never-granted or legacy-unstamped take is an unknown.
 */

const NOW = Date.parse("2026-09-21T12:00:00Z");

const HOUR_MS = 60 * 60 * 1000;

function recordOf(
  captureId: string,
  text: string,
  options: {
    readonly hoursAgo?: number;
    readonly kinds?: ReadonlySet<InsightTermKind>;
  } = {},
): InsightTermRecord {
  return {
    schema_version: 1,
    capture_id: captureId,
    occurred_at: new Date(NOW - (options.hoursAgo ?? 1) * HOUR_MS).toISOString(),
    tokenizer: TOKENIZER_ID,
    ...captureTerms(text, { kinds: options.kinds }),
  };
}

function recognitionOf(captureId: string, hoursAgo = 1): RecognitionSelectedEvent {
  return {
    schema_version: 1,
    event_id: `rs-${captureId}`,
    capture_id: captureId,
    occurred_at: new Date(NOW - hoursAgo * HOUR_MS).toISOString(),
    type: "recognition_selected",
    attempt_id: `att-${captureId}`,
    selection_seq: 1,
    lexical_words: 5,
    raw_words: 5,
    tokenizer: TOKENIZER_ID,
    post_stop_ready_ms: null,
  };
}

function deletionOf(captureId: string): CaptureDeletedEvent {
  return {
    schema_version: 1,
    event_id: `cd-${captureId}`,
    capture_id: captureId,
    occurred_at: new Date(NOW - 30 * 60 * 1000).toISOString(),
    type: "capture_deleted",
  };
}

function labelsOf(cards: readonly VoicePatternCard[]): readonly string[] {
  return cards.map((card) => card.label);
}

describe("recurringPhraseCards", () => {
  it("grounds every claim in the counts it cites", () => {
    const records = [
      recordOf("take-1", "deploy the server and deploy the server again"),
      recordOf("take-2", "please deploy the server once more"),
    ];

    const card = recurringPhraseCards(records, { now: NOW }).find(
      (candidate) => candidate.label === "deploy the server",
    );

    expect(card).toBeDefined();
    expect(card?.takes).toBe(2);
    expect(card?.occurrences).toBe(3);
    expect(card?.windowTakes).toBe(2);
    expect(card?.description).toBe(
      `"deploy the server" appeared in 2 of 2 analyzed takes, 3 times in the last 14 days`,
    );
  });

  it("counts only takes stamped as analyzed for this kind in the denominator", () => {
    // take-3 was written under a narrower grant (terms only, per its stamp):
    // it is an unknown for phrase cards, not a negative — the denominator
    // must not count it.
    const records = [
      recordOf("take-1", "deploy the server and deploy the server again"),
      recordOf("take-2", "please deploy the server once more"),
      recordOf("take-3", "unrelated solitary vocabulary", { kinds: new Set(["terms"]) }),
    ];

    const card = recurringPhraseCards(records, { now: NOW }).find(
      (candidate) => candidate.label === "deploy the server",
    );

    expect(card?.windowTakes).toBe(2);
    expect(card?.description).toBe(
      `"deploy the server" appeared in 2 of 2 analyzed takes, 3 times in the last 14 days`,
    );
  });

  it("counts an analyzed take that yielded zero of the kind", () => {
    // take-3 stamped phrases but the transcript produced no phrases: a real
    // zero, not an unknown — excluding it would understate the denominator
    // and overstate recurrence.
    const records = [
      recordOf("take-1", "deploy the server and deploy the server again"),
      recordOf("take-2", "please deploy the server once more"),
      { ...recordOf("take-3", "a"), phrases: [] },
    ];

    const card = recurringPhraseCards(records, { now: NOW }).find(
      (candidate) => candidate.label === "deploy the server",
    );

    expect(card?.windowTakes).toBe(3);
    expect(card?.description).toContain("2 of 3 analyzed takes");
  });

  it("counts a legacy unstamped record only through its label evidence", () => {
    // A record written before derived_kinds existed decodes fine but says
    // nothing about what it analyzed. Labels of a kind present are the
    // evidence it was analyzed for that kind (the writer derives only
    // granted kinds); absent labels stay an unknown — no guessing.
    const { derived_kinds: _stamp, ...legacy } = recordOf("take-3", "deploy the server");
    const { derived_kinds: _noStamp, ...emptyLegacy } = recordOf("take-4", "quiet");

    const records = [
      recordOf("take-1", "deploy the server and deploy the server again"),
      recordOf("take-2", "please deploy the server once more"),
      legacy,
      emptyLegacy,
    ];

    const card = recurringPhraseCards(records, { now: NOW }).find(
      (candidate) => candidate.label === "deploy the server",
    );

    expect(card?.windowTakes).toBe(3);
    expect(card?.takes).toBeLessThanOrEqual(card?.windowTakes ?? 0);
  });

  it("requires the phrase to recur across takes, not within one", () => {
    const records = [recordOf("take-1", "very unique phrase very unique phrase")];

    expect(recurringPhraseCards(records, { now: NOW })).toEqual([]);
  });

  it("bounds the population to the declared window", () => {
    const records = [
      recordOf("take-1", "deploy the server"),
      recordOf("take-2", "deploy the server", { hoursAgo: 24 * 20 }), // outside 14 days
    ];

    const cards = recurringPhraseCards(records, { windowDays: 14, now: NOW });

    expect(cards).toEqual([]);
  });

  it("drops excluded labels entirely", () => {
    const records = [
      recordOf("take-1", "deploy the server"),
      recordOf("take-2", "deploy the server"),
      recordOf("take-3", "rotate the keys"),
      recordOf("take-4", "rotate the keys"),
    ];

    const cards = recurringPhraseCards(records, {
      now: NOW,
      exclude: new Set(["deploy the server", "deploy the", "the server"]),
    });

    // The deploy phrases are gone; the rotate phrases remain in their
    // deterministic order.
    expect(labelsOf(cards)).toEqual(["rotate the", "rotate the keys", "the keys"]);
  });

  it("orders deterministically (takes, then occurrences, then label) and limits", () => {
    const records = [
      recordOf("t1", "alpha beta gamma"),
      recordOf("t2", "alpha beta gamma"),
      recordOf("t3", "alpha beta gamma"),
      recordOf("t4", "delta epsilon zeta"),
      recordOf("t5", "delta epsilon zeta"),
    ];

    expect(labelsOf(recurringPhraseCards(records, { now: NOW }))).toEqual([
      "alpha beta",
      "alpha beta gamma",
      "beta gamma",
      "delta epsilon",
      "delta epsilon zeta",
    ]);
    expect(labelsOf(recurringPhraseCards(records, { now: NOW, limit: 1 }))).toEqual(["alpha beta"]);
  });
});

describe("vocabularyPatternCards", () => {
  it("applies the declared minimum term length", () => {
    const records = [
      recordOf("take-1", "run the api gateway"),
      recordOf("take-2", "run the api gateway"),
    ];

    const cards = vocabularyPatternCards(records, { now: NOW });

    expect(labelsOf(cards)).toContain("gateway");
    expect(labelsOf(cards)).not.toContain("api");
    expect(labelsOf(cards)).not.toContain("run");
    expect(cards[0]?.description).toContain("terms under 4 characters are not counted");
  });

  it("loses a term when deletion removes its second take", () => {
    const records = [
      recordOf("take-1", "kubernetes rollout"),
      recordOf("take-2", "kubernetes rollout"),
    ];

    expect(labelsOf(vocabularyPatternCards(records, { now: NOW }))).toContain("kubernetes");

    // take-2 was deleted; its aggregates are gone from the store, and the
    // term no longer recurs across takes.
    const afterDeletion = records.filter((record) => record.capture_id !== "take-2");

    expect(vocabularyPatternCards(afterDeletion, { now: NOW })).toEqual([]);
  });
});

describe("voicePanel consent gating", () => {
  const records = [
    recordOf("take-1", "deploy the server"),
    recordOf("take-2", "deploy the server"),
  ];

  const events: readonly InsightEvent[] = [
    recognitionOf("take-1"),
    recognitionOf("take-2"),
    recognitionOf("take-9", 24 * 20), // outside the window
  ];

  it("yields no cards of a kind whose grant is off", () => {
    const panel = voicePanel(events, records, DEFAULT_INSIGHT_CONSENT, { now: NOW });

    expect(panel.phraseCards).toEqual([]);
    expect(panel.vocabularyCards).toEqual([]);
    expect(panel.windowTakes).toBe(2);
  });

  it("enables each kind independently", () => {
    const phrasesOnly = voicePanel(
      events,
      records,
      { ...DEFAULT_INSIGHT_CONSENT, recurringPhrases: true },
      { now: NOW },
    );

    expect(phrasesOnly.phraseCards.length).toBeGreaterThan(0);
    expect(phrasesOnly.vocabularyCards).toEqual([]);

    const vocabularyOnly = voicePanel(
      events,
      records,
      { ...DEFAULT_INSIGHT_CONSENT, vocabularyPatterns: true },
      { now: NOW },
    );

    expect(vocabularyOnly.phraseCards).toEqual([]);
    expect(vocabularyOnly.vocabularyCards.length).toBeGreaterThan(0);
  });
});

describe("voicePanel denominators", () => {
  it("drops deleted takes from windowTakes like every other surface", () => {
    const records = [
      recordOf("take-1", "deploy the server"),
      recordOf("take-2", "rotate the keys"),
      recordOf("take-9", "deploy the server"),
    ];

    const deleted: readonly InsightEvent[] = [
      recognitionOf("take-1"),
      recognitionOf("take-2"),
      recognitionOf("take-9"),
      deletionOf("take-9"),
    ];

    const live: readonly InsightEvent[] = [
      recognitionOf("take-1"),
      recognitionOf("take-2"),
      recognitionOf("take-9"),
    ];

    // The tombstone keeps take-9's recognition event in the log; the metric
    // contract drops the capture everywhere, and so must this denominator —
    // a deleted take cannot survive in a number.
    expect(voicePanel(live, records, DEFAULT_INSIGHT_CONSENT, { now: NOW }).windowTakes).toBe(3);
    expect(voicePanel(deleted, records, DEFAULT_INSIGHT_CONSENT, { now: NOW }).windowTakes).toBe(2);

    // The dropped capture cannot be "analyzed" either.
    expect(voicePanel(deleted, records, DEFAULT_INSIGHT_CONSENT, { now: NOW }).analyzedTakes).toBe(
      2,
    );
  });

  it("counts analyzedTakes by the derived_kinds stamp, not by label lists", () => {
    const events: readonly InsightEvent[] = [
      recognitionOf("take-1"),
      recognitionOf("take-2"),
      recognitionOf("take-3"),
    ];

    const stamped = [
      recordOf("take-1", "deploy the server"),
      // Analyzed under both grants but the transcript yielded nothing: a
      // real zero for both kinds, still an analyzed take.
      { ...recordOf("take-3", "solo"), terms: [], phrases: [] },
    ];

    expect(voicePanel(events, stamped, DEFAULT_INSIGHT_CONSENT, { now: NOW }).analyzedTakes).toBe(
      2,
    );

    // A legacy unstamped record counts only through label evidence:
    // one holding aggregates joins the coverage, an empty one does not.
    const { derived_kinds: _stamp, ...legacy } = recordOf("take-2", "rotate the keys");
    const { derived_kinds: _noStamp, ...emptyLegacy } = recordOf("take-4", "quiet");

    expect(
      voicePanel(events, [...stamped, legacy], DEFAULT_INSIGHT_CONSENT, { now: NOW }).analyzedTakes,
    ).toBe(3);

    expect(
      voicePanel(events, [...stamped, emptyLegacy], DEFAULT_INSIGHT_CONSENT, { now: NOW })
        .analyzedTakes,
    ).toBe(2);
  });
});
