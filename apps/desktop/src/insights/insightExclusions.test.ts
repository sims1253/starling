import { describe, expect, it } from "vite-plus/test";

import {
  MAX_EXCLUSIONS,
  MAX_EXCLUSION_LABEL_LENGTH,
  readExclusions,
  writeExclusions,
  type ExclusionStorage,
} from "./insightExclusions";

/**
 * Exclusion-list storage coverage (E29 phase 2): the stored value is
 * untrusted input with a bounded, validated read — a corrupted or hostile
 * entry can never grow the in-memory set without limit — and a refused
 * write reports itself instead of crashing the interaction that asked for
 * the exclusion.
 */

function storage(initial: Record<string, string> = {}): ExclusionStorage {
  const entries = new Map(Object.entries(initial));

  return {
    getItem: (key) => entries.get(key) ?? null,
    setItem: (key, value) => {
      entries.set(key, value);
    },
  };
}

describe("readExclusions", () => {
  it("round-trips a written set", () => {
    const held = storage();

    writeExclusions(held, new Set(["deploy the server", "alpha"]));

    expect(readExclusions(held)).toEqual(new Set(["deploy the server", "alpha"]));
  });

  it("treats unreadable, missing and non-array values as no exclusions", () => {
    expect(readExclusions(storage())).toEqual(new Set());

    expect(readExclusions(storage({ "starling:insights:exclusions": "not json" }))).toEqual(
      new Set(),
    );

    expect(readExclusions(storage({ "starling:insights:exclusions": '{"a":"b"}' }))).toEqual(
      new Set(),
    );
  });

  it("keeps only strings that look like labels and caps the count", () => {
    const labels = Array.from({ length: MAX_EXCLUSIONS + 50 }, (_, index) => `label${index}`);

    const hostile = JSON.stringify([
      "a legitimate label",
      ...labels,
      42,
      null,
      { nested: "object" },
      "",
      "x".repeat(MAX_EXCLUSION_LABEL_LENGTH + 1),
    ]);

    const read = readExclusions(storage({ "starling:insights:exclusions": hostile }));

    expect(read.size).toBe(MAX_EXCLUSIONS);
    expect(read.has("a legitimate label")).toBe(true);
    expect(read.has("")).toBe(false);
    expect([...read].some((label) => label.length > MAX_EXCLUSION_LABEL_LENGTH)).toBe(false);
  });

  it("refuses labels that are not the one-to-three-token shape", () => {
    // Double spaces, tabs and newlines survive a length bound but not the
    // boundary: the comment's shape contract is enforced, not just stated.
    const malformed = JSON.stringify([
      "double  spaced",
      "tab\tseparated",
      "line\nbroken",
      "trailing ",
      "four word phrase here",
    ]);

    const read = readExclusions(storage({ "starling:insights:exclusions": malformed }));

    expect(read.size).toBe(0);
  });
});

describe("writeExclusions", () => {
  it("reports a refused write instead of throwing through the caller", () => {
    const refusing: ExclusionStorage = {
      getItem: () => null,
      setItem: () => {
        throw new Error("quota exceeded");
      },
    };

    expect(writeExclusions(refusing, new Set(["alpha"]))).toBe(false);
  });

  it("reports a successful write", () => {
    expect(writeExclusions(storage(), new Set(["alpha"]))).toBe(true);
  });
});
