import { describe, expect, it } from "vite-plus/test";

import {
  MAX_BASELINE_DRAFT_LENGTH,
  readTypingBaseline,
  writeTypingBaseline,
  type BaselineStorage,
} from "./insightBaseline";

/**
 * Typing-baseline storage coverage (E29 phase 2): the stored value is
 * untrusted input with a bounded, validated read — free text or a pasted
 * novel can never seed the field or occupy the settings entry without
 * limit — and a refused write reports itself instead of crashing the
 * interaction that asked for the baseline.
 */

function storage(initial: Record<string, string> = {}): BaselineStorage {
  const entries = new Map(Object.entries(initial));

  return {
    getItem: (key) => entries.get(key) ?? null,
    setItem: (key, value) => {
      entries.set(key, value);
    },
  };
}

describe("readTypingBaseline", () => {
  it("round-trips a written draft", () => {
    const held = storage();

    writeTypingBaseline(held, "150");

    expect(readTypingBaseline(held)).toBe("150");
  });

  it("reads unset for missing, free-text and oversized values", () => {
    expect(readTypingBaseline(storage())).toBe("");
    expect(readTypingBaseline(storage({ "starling:insights:typingWpm": "fast" }))).toBe("");
    expect(readTypingBaseline(storage({ "starling:insights:typingWpm": "9".repeat(50) }))).toBe("");
  });

  it("admits mid-typing drafts, including the empty and minus-only ones", () => {
    expect(readTypingBaseline(storage({ "starling:insights:typingWpm": "" }))).toBe("");
    expect(readTypingBaseline(storage({ "starling:insights:typingWpm": "-" }))).toBe("-");
    expect(readTypingBaseline(storage({ "starling:insights:typingWpm": "-3" }))).toBe("-3");
  });
});

describe("writeTypingBaseline", () => {
  it("reports a refused write instead of throwing through the caller", () => {
    const refusing: BaselineStorage = {
      getItem: () => null,
      setItem: () => {
        throw new Error("quota exceeded");
      },
    };

    expect(writeTypingBaseline(refusing, "150")).toBe(false);
  });

  it("refuses drafts beyond the bound so the entry cannot grow unbounded", () => {
    const held = storage();

    expect(writeTypingBaseline(held, "x".repeat(MAX_BASELINE_DRAFT_LENGTH + 1))).toBe(false);
    expect(readTypingBaseline(held)).toBe("");
  });

  it("reports a successful write", () => {
    expect(writeTypingBaseline(storage(), "150")).toBe(true);
  });
});
