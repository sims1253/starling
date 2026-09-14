export type FidelityWarningCode =
  | "non-one-list-start"
  | "possible-self-correction"
  | "negation-present"
  | "expected-term-missing"
  | "short-answer"
  | "discourse-word-preserved"
  | "possible-audio-gap";

export interface TextRange {
  readonly start: number;
  readonly end: number;
}

export interface FidelityWarning {
  readonly code: FidelityWarningCode;
  readonly severity: "info" | "review";
  readonly message: string;
  readonly range?: TextRange;
  readonly term?: string;
}

/** An advisory change. Applying it is always an explicit UI/user action. */
export interface SuggestedEdit {
  readonly start: number;
  readonly end: number;
  readonly replacement: string;
  readonly reason: string;
  readonly source: "user" | "normalizer";
}

export interface TranscriptAnalysisOptions {
  /** Domain words that should have appeared in this utterance. */
  readonly expectedTerms?: readonly string[];
  /** Edits supplied by a user or separate normalizer, never applied here. */
  readonly suggestedEdits?: readonly SuggestedEdit[];
  /** Optional coverage data from timestamp-bearing server responses. */
  readonly recordingDurationSeconds?: number;
  readonly coveredDurationSeconds?: number;
}

export interface TranscriptAnalysis {
  /** The exact string supplied to analyzeTranscript, including whitespace. */
  readonly rawText: string;
  readonly suggestedEdits: readonly SuggestedEdit[];
  readonly warnings: readonly FidelityWarning[];
}

function frozenRange(start: number, end: number): TextRange {
  return Object.freeze({ start, end });
}

function pushMatches(
  text: string,
  expression: RegExp,
  warning: (match: RegExpExecArray) => FidelityWarning,
  target: FidelityWarning[],
): void {
  expression.lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = expression.exec(text)) !== null) target.push(Object.freeze(warning(match)));
}

function normalizeTerm(value: string): string {
  return value.normalize("NFKC").toLocaleLowerCase();
}

function validateSuggestions(
  rawText: string,
  values: readonly SuggestedEdit[],
): readonly SuggestedEdit[] {
  const frozen = values.map((edit) => {
    if (
      !Number.isInteger(edit.start) ||
      !Number.isInteger(edit.end) ||
      edit.start < 0 ||
      edit.end < edit.start ||
      edit.end > rawText.length
    ) {
      throw new RangeError("suggested edit range must be within rawText");
    }

    if (edit.source !== "user" && edit.source !== "normalizer") {
      throw new TypeError("suggested edit source must be user or normalizer");
    }

    return Object.freeze({ ...edit });
  });

  return Object.freeze(frozen);
}

/**
 * Analyze fidelity-sensitive text without changing it.
 *
 * The warnings identify spans worth preserving or reviewing. They cannot tell
 * whether the recognizer heard the audio correctly, and they deliberately do
 * not guess replacements for uncommon words, negations, or corrections.
 */
export function analyzeTranscript(
  rawText: string,
  options: TranscriptAnalysisOptions = {},
): TranscriptAnalysis {
  const warnings: FidelityWarning[] = [];

  const firstListItem = /(?:^|\n)[ \t]*(\d+)[.)](?=\s)/m.exec(rawText);

  if (firstListItem?.[1] && Number(firstListItem[1]) !== 1) {
    const numberStart = firstListItem.index + firstListItem[0].indexOf(firstListItem[1]);
    warnings.push(
      Object.freeze({
        code: "non-one-list-start",
        severity: "info",
        message: `List starts at ${firstListItem[1]}. Keep that number.`,
        range: frozenRange(numberStart, numberStart + firstListItem[1].length),
      }),
    );
  }

  pushMatches(
    rawText,
    /\b(?:er+|uh+|um+)\b/giu,
    (match) => ({
      code: "possible-self-correction",
      severity: "review",
      message: "Possible spoken correction. Check the nearby words before editing.",
      range: frozenRange(match.index, match.index + match[0].length),
    }),
    warnings,
  );

  pushMatches(
    rawText,
    /\b(?:not|never|no|neither|nor|cannot|can't|won't|wouldn't|shouldn't|isn't|aren't|haven't|hasn't|hadn't|don't|doesn't|didn't)\b/giu,
    (match) => ({
      code: "negation-present",
      severity: "info",
      message: "Keep negations when editing; removing them can change the meaning.",
      range: frozenRange(match.index, match.index + match[0].length),
    }),
    warnings,
  );

  pushMatches(
    rawText,
    /\blike\b/giu,
    (match) => ({
      code: "discourse-word-preserved",
      severity: "info",
      message: 'The transcript keeps the word "like".',
      range: frozenRange(match.index, match.index + match[0].length),
    }),
    warnings,
  );

  if (/^\s*(?:[A-Za-z]|agreed|yes|no|okay|ok)\s*[.!]?\s*$/iu.test(rawText)) {
    warnings.push(
      Object.freeze({
        code: "short-answer",
        severity: "info",
        message: "Keep short answers, including a single letter.",
        range: frozenRange(0, rawText.length),
      }),
    );
  }

  const normalizedText = normalizeTerm(rawText);

  for (const term of options.expectedTerms ?? []) {
    if (!term || normalizedText.includes(normalizeTerm(term))) continue;
    warnings.push(
      Object.freeze({
        code: "expected-term-missing",
        severity: "review",
        message: `Expected term "${term}" is missing. Check the saved audio.`,
        term,
      }),
    );
  }

  const recording = options.recordingDurationSeconds;
  const covered = options.coveredDurationSeconds;

  if (
    recording !== undefined &&
    covered !== undefined &&
    Number.isFinite(recording) &&
    Number.isFinite(covered) &&
    recording > 1 &&
    covered >= 0 &&
    covered < recording * 0.8
  ) {
    warnings.push(
      Object.freeze({
        code: "possible-audio-gap",
        severity: "review",
        message: "Timed segments cover less than 80% of the recording. Check the saved audio.",
      }),
    );
  }

  return Object.freeze({
    rawText,
    suggestedEdits: validateSuggestions(rawText, options.suggestedEdits ?? []),
    warnings: Object.freeze(warnings),
  });
}

/** Apply one explicitly selected edit, returning a new string. */
export function applySuggestedEdit(rawText: string, edit: SuggestedEdit): string {
  validateSuggestions(rawText, [edit]);

  return `${rawText.slice(0, edit.start)}${edit.replacement}${rawText.slice(edit.end)}`;
}
