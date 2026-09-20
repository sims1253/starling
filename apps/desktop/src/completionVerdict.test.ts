import { describe, expect, it } from "vite-plus/test";

import {
  completionRejectionMessage,
  completionVerdict,
  type CompletionChoice,
} from "./completionVerdict";

function choice(overrides: Partial<CompletionChoice> = {}): CompletionChoice {
  return { content: "Refined text.", finishReason: "stop", refusal: undefined, ...overrides };
}

describe("completionVerdict", () => {
  it("accepts a completion that finished on the model's own stop", () => {
    expect(completionVerdict(choice())).toEqual({
      accepted: true,
      content: "Refined text.",
    });
  });

  it("accepts a compatible local response that omits the termination field", () => {
    // Documented compatibility policy: local servers (llama.cpp, LM Studio,
    // older Ollama builds) may omit finish_reason entirely. Omission is not
    // evidence of truncation, so the completion is trusted like a stop.
    for (const finishReason of [undefined, null]) {
      expect(completionVerdict(choice({ finishReason }))).toEqual({
        accepted: true,
        content: "Refined text.",
      });
    }
  });

  it("rejects finish_reason length as truncated, whatever the content holds", () => {
    const verdict = completionVerdict(
      choice({ content: "Only a prefix of the", finishReason: "length" }),
    );

    expect(verdict.accepted).toBe(false);

    if (!verdict.accepted) expect(verdict.rejection.reason).toBe("truncated");
  });

  it("rejects finish_reason content_filter as filtered", () => {
    const verdict = completionVerdict(choice({ content: "", finishReason: "content_filter" }));

    expect(verdict.accepted).toBe(false);

    if (!verdict.accepted) expect(verdict.rejection.reason).toBe("content-filter");
  });

  it("rejects a non-empty model refusal even with finish_reason stop", () => {
    const verdict = completionVerdict(choice({ refusal: "I cannot edit this transcript." }));

    expect(verdict.accepted).toBe(false);

    if (!verdict.accepted) {
      expect(verdict.rejection.reason).toBe("refusal");

      if (verdict.rejection.reason === "refusal")
        expect(verdict.rejection.refusal).toBe("I cannot edit this transcript.");
    }
  });

  it("rejects empty or whitespace-only content when nothing explains it", () => {
    for (const content of ["", "   \n\t  "]) {
      const verdict = completionVerdict(choice({ content }));

      expect(verdict.accepted).toBe(false);

      if (!verdict.accepted) expect(verdict.rejection.reason).toBe("empty");
    }
  });

  it("rejects an unrecognized finish_reason instead of guessing it means stop", () => {
    const verdict = completionVerdict(choice({ finishReason: "eos" }));

    expect(verdict.accepted).toBe(false);

    if (!verdict.accepted) {
      expect(verdict.rejection.reason).toBe("unrecognized-finish");

      if (verdict.rejection.reason === "unrecognized-finish")
        expect(verdict.rejection.finishReason).toBe("eos");
    }
  });
});

describe("completionRejectionMessage", () => {
  it("gives truncation an actionable recovery path and names what stays intact", () => {
    const message = completionRejectionMessage({
      reason: "truncated",
      finishReason: "length",
    });

    expect(message).toContain("output limit");
    expect(message).toContain('finish_reason "length"');
    expect(message).toContain("num_predict");
    expect(message).toContain("shorter take");
    expect(message).toContain("previous refined text is unchanged");
  });

  it("says what was filtered and that nothing was replaced", () => {
    const message = completionRejectionMessage({
      reason: "content-filter",
      finishReason: "content_filter",
    });

    expect(message).toContain("content_filter");
    expect(message).toContain("previous refined text is unchanged");
  });

  it("quotes the model's own refusal", () => {
    const message = completionRejectionMessage({
      reason: "refusal",
      refusal: "I cannot edit this transcript.",
    });

    expect(message).toContain("refused");
    expect(message).toContain("I cannot edit this transcript.");
  });

  it("names an unrecognized finish_reason so the provider can be identified", () => {
    const message = completionRejectionMessage({
      reason: "unrecognized-finish",
      finishReason: "eos",
    });

    expect(message).toContain('"eos"');
    expect(message).toContain('"stop"');
  });

  it("keeps the established empty-transcript wording", () => {
    expect(completionRejectionMessage({ reason: "empty" })).toBe(
      "The refinement server returned an empty refined transcript.",
    );
  });
});
