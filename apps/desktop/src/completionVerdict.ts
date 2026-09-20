/**
 * Termination verdict for one chat-completions choice (B09).
 *
 * A non-empty `choices[0].message.content` proves nothing on its own: the
 * content can be a prefix the model stopped writing because it hit its output
 * budget. The OpenAI-compatible termination metadata — `finish_reason` and,
 * for refusals, `message.refusal` — is what says the answer is whole, so the
 * refinement path reads it before accepting anything.
 *
 * Compatibility policy for local providers: a response that omits
 * `finish_reason` entirely (or sends null) is accepted. Local servers —
 * llama.cpp, LM Studio, older Ollama builds — may not send the field, and
 * omission is not evidence of truncation. An explicit value, though, is read
 * strictly: only "stop" proves a complete answer, "length" and
 * "content_filter" are named failures, and any other value is refused as
 * unrecognized rather than guessed at.
 */

/** What the server actually said about one choice, decoded but unjudged. */
export interface CompletionChoice {
  readonly content: string;

  /** Termination metadata; undefined or null when the provider omits it. */
  readonly finishReason: string | null | undefined;

  /** The model's own refusal text when it declined the request. */
  readonly refusal: string | null | undefined;
}

export type CompletionRejection =
  | { readonly reason: "truncated"; readonly finishReason: "length" }
  | { readonly reason: "content-filter"; readonly finishReason: "content_filter" }
  | { readonly reason: "refusal"; readonly refusal: string }
  | { readonly reason: "empty" }
  | { readonly reason: "unrecognized-finish"; readonly finishReason: string };

export type CompletionVerdict =
  | { readonly accepted: true; readonly content: string }
  | { readonly accepted: false; readonly rejection: CompletionRejection };

/**
 * Judge one decoded completion choice. Order matters: a stated refusal wins
 * over everything (the model said no, whatever else it also said), then the
 * explicit finish_reason, and only then the content's own emptiness — so an
 * empty content alongside "length" reads as truncation, not as an empty
 * transcript.
 */
export function completionVerdict(choice: CompletionChoice): CompletionVerdict {
  const refusal = choice.refusal?.trim();

  if (refusal) return { accepted: false, rejection: { reason: "refusal", refusal } };

  const finishReason = choice.finishReason ?? null;

  if (finishReason === "length")
    return { accepted: false, rejection: { reason: "truncated", finishReason } };

  if (finishReason === "content_filter")
    return { accepted: false, rejection: { reason: "content-filter", finishReason } };

  if (finishReason !== null && finishReason !== "stop")
    return {
      accepted: false,
      rejection: { reason: "unrecognized-finish", finishReason },
    };

  if (choice.content.trim().length === 0)
    return { accepted: false, rejection: { reason: "empty" } };

  return { accepted: true, content: choice.content };
}

/**
 * The user-facing sentence for a rejected completion. Every message names
 * what stays intact — the previous refined text — because a rejected
 * refinement never replaces anything the take already holds.
 */
export function completionRejectionMessage(rejection: CompletionRejection): string {
  switch (rejection.reason) {
    case "truncated":
      return [
        'The refinement model hit its output limit, so the result would be truncated (finish_reason "length"). It was discarded and the previous refined text is unchanged.',
        "Raise the model's output limit (for example Ollama's num_predict or the context length in LM Studio) or refine a shorter take, then try again.",
      ].join(" ");
    case "content-filter":
      return [
        'The refinement server filtered the result (finish_reason "content_filter"), so no refined transcript was produced.',
        "The previous refined text is unchanged; the raw transcript was not touched.",
      ].join(" ");
    case "refusal":
      return [
        `The refinement model refused to process the transcript: "${rejection.refusal}"`,
        "The previous refined text is unchanged; the raw transcript was not touched.",
      ].join(" ");
    case "empty":
      return "The refinement server returned an empty refined transcript.";
    case "unrecognized-finish":
      return [
        `The refinement server reported an unrecognized finish_reason "${rejection.finishReason}"; only "stop" proves a complete refinement.`,
        "The result was discarded and the previous refined text is unchanged.",
      ].join(" ");
  }
}
