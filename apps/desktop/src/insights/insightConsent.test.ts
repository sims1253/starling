import { describe, expect, it } from "vite-plus/test";

import {
  DEFAULT_INSIGHT_CONSENT,
  consentedTermKinds,
  readInsightConsent,
  writeInsightConsent,
  type ConsentStorage,
} from "./insightConsent";

/**
 * Consent coverage (E29 phase 2): every grant defaults to off, storage
 * round-trips whole objects only, and anything unreadable or malformed is
 * the default — a damaged value must never read as granted. The two
 * content-derived analyses stay independent: neither implies the other.
 */

function storage(initial: Record<string, string> = {}): ConsentStorage {
  const entries = new Map(Object.entries(initial));

  return {
    getItem: (key) => entries.get(key) ?? null,
    setItem: (key, value) => {
      entries.set(key, value);
    },
    removeItem: (key) => {
      entries.delete(key);
    },
  };
}

describe("readInsightConsent", () => {
  it("defaults every grant to off and the goal to none", () => {
    expect(readInsightConsent(storage())).toEqual(DEFAULT_INSIGHT_CONSENT);
    expect(DEFAULT_INSIGHT_CONSENT).toEqual({
      recurringPhrases: false,
      vocabularyPatterns: false,
      milestones: false,
      streaks: false,
      weeklyGoalWords: null,
    });
  });

  it("round-trips a whole granted state", () => {
    const held = storage();

    const granted = {
      recurringPhrases: true,
      vocabularyPatterns: false,
      milestones: true,
      streaks: false,
      weeklyGoalWords: 500,
    };

    writeInsightConsent(held, granted);

    expect(readInsightConsent(held)).toEqual(granted);
  });

  it("treats malformed or partial stored values as all-off", () => {
    expect(readInsightConsent(storage({ "starling:insights:consent": "{not json" }))).toEqual(
      DEFAULT_INSIGHT_CONSENT,
    );
    expect(readInsightConsent(storage({ "starling:insights:consent": "42" }))).toEqual(
      DEFAULT_INSIGHT_CONSENT,
    );
    // A string is not a grant; a missing field is not a grant.
    expect(
      readInsightConsent(
        storage({ "starling:insights:consent": JSON.stringify({ recurringPhrases: "yes" }) }),
      ),
    ).toEqual(DEFAULT_INSIGHT_CONSENT);
  });

  it("refuses out-of-range goals instead of clamping them", () => {
    const whole = (goal: number): string =>
      JSON.stringify({
        recurringPhrases: false,
        vocabularyPatterns: false,
        milestones: false,
        streaks: false,
        weeklyGoalWords: goal,
      });

    for (const bad of [0, -5, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
      const held = storage({ "starling:insights:consent": whole(bad) });

      expect(readInsightConsent(held).weeklyGoalWords).toBeNull();
    }

    const held = storage({ "starling:insights:consent": whole(250) });

    expect(readInsightConsent(held).weeklyGoalWords).toBe(250);
  });
});

describe("consentedTermKinds", () => {
  it("maps each content-derived grant to its own kind", () => {
    expect(consentedTermKinds(DEFAULT_INSIGHT_CONSENT).size).toBe(0);
    expect([...consentedTermKinds({ ...DEFAULT_INSIGHT_CONSENT, recurringPhrases: true })]).toEqual(
      ["phrases"],
    );
    expect([
      ...consentedTermKinds({ ...DEFAULT_INSIGHT_CONSENT, vocabularyPatterns: true }),
    ]).toEqual(["terms"]);
    expect(
      [
        ...consentedTermKinds({
          ...DEFAULT_INSIGHT_CONSENT,
          recurringPhrases: true,
          vocabularyPatterns: true,
        }),
      ].sort(),
    ).toEqual(["phrases", "terms"]);
  });
});
