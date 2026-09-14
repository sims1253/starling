import assert from "node:assert/strict";
import { describe, it } from "vite-plus/test";

import { analyzeTranscript, applySuggestedEdit } from "../src/fidelity.js";

describe("fidelity-sensitive transcripts", () => {
  it("preserves a numbered list that starts after one", () => {
    const raw = "5. capture audio\n6. transcribe it";
    const analysis = analyzeTranscript(raw);
    assert.equal(analysis.rawText, raw);
    assert.equal(analysis.warnings[0]?.code, "non-one-list-start");
    assert.equal(analysis.suggestedEdits.length, 0);
  });

  it("does not replace uncommon expected vocabulary", () => {
    const correct = analyzeTranscript("Add auth to the proxy.", { expectedTerms: ["auth"] });
    assert.equal(correct.rawText, "Add auth to the proxy.");
    assert.equal(
      correct.warnings.some((warning) => warning.code === "expected-term-missing"),
      false,
    );

    const suspect = analyzeTranscript("Add off to the proxy.", { expectedTerms: ["auth"] });
    assert.equal(suspect.rawText, "Add off to the proxy.");
    assert.equal(
      suspect.warnings.find((warning) => warning.code === "expected-term-missing")?.term,
      "auth",
    );
  });

  it("flags a correction sound but does not guess the intended edit", () => {
    const raw = "I want the color to be orange, err, yellow.";
    const analysis = analyzeTranscript(raw);
    assert.equal(analysis.rawText, raw);
    assert.equal(
      analysis.warnings.some((warning) => warning.code === "possible-self-correction"),
      true,
    );
    assert.deepEqual(analysis.suggestedEdits, []);
  });

  it("preserves every meaning-bearing negation", () => {
    const raw = "I'd prefer to never merge this, and I haven't approved it.";
    const analysis = analyzeTranscript(raw);
    assert.equal(analysis.rawText, raw);
    assert.equal(
      analysis.warnings.filter((warning) => warning.code === "negation-present").length,
      2,
    );
  });

  it("preserves the discourse word like", () => {
    const raw = "This was, like, much easier to review.";
    const analysis = analyzeTranscript(raw);
    assert.equal(analysis.rawText, raw);
    assert.equal(
      analysis.warnings.some((warning) => warning.code === "discourse-word-preserved"),
      true,
    );
  });

  it("treats A and agreed as content rather than empty noise", () => {
    for (const raw of ["A", "agreed."]) {
      const analysis = analyzeTranscript(raw);
      assert.equal(analysis.rawText, raw);
      assert.equal(
        analysis.warnings.some((warning) => warning.code === "short-answer"),
        true,
      );
    }
  });

  it("warns on measured coverage gaps without fabricating missing text", () => {
    const raw = "First answer. Second answer.";

    const analysis = analyzeTranscript(raw, {
      recordingDurationSeconds: 600,
      coveredDurationSeconds: 240,
    });

    assert.equal(analysis.rawText, raw);
    assert.equal(
      analysis.warnings.some((warning) => warning.code === "possible-audio-gap"),
      true,
    );
  });

  it("keeps normalizer edits separate until explicitly applied", () => {
    const raw = "hello world";

    const edit = {
      start: 0,
      end: 5,
      replacement: "Hello",
      reason: "sentence casing",
      source: "normalizer" as const,
    };

    const analysis = analyzeTranscript(raw, { suggestedEdits: [edit] });
    assert.equal(analysis.rawText, raw);
    const selectedEdit = analysis.suggestedEdits[0];
    assert.ok(selectedEdit);
    assert.equal(applySuggestedEdit(analysis.rawText, selectedEdit), "Hello world");
  });
});
