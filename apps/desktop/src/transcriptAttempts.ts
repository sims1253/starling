import type {
  DictationSession,
  DictationSessionStatus,
  RefinedTranscript,
  TranscriptAttempt,
} from "@starling/dictation";

/**
 * Retranscription decisions for finished takes (B04).
 *
 * A durable recording's audio is never consumed by an attempt, so every
 * settled take — successful, empty, failed, or never attempted — can be sent
 * to the transcription server again, with the same model or another one
 * chosen in settings, without re-recording or reimporting. The only state
 * that forbids a new attempt is one still running; everything else is a
 * recovery or review action away.
 */

/**
 * Whether the drawer offers "Transcribe again" for this take right now. The
 * running check is two-fold: the durable `status` mark ("transcribing") and
 * the in-flight upload set, because a status can go stale — a take whose
 * attempt just failed still reads "transcribing" until the failure write
 * lands — while the upload set is exact for this window.
 */
export function canTranscribeAgain(
  status: DictationSessionStatus,
  attemptInFlight: boolean,
): boolean {
  return !attemptInFlight && status !== "transcribing";
}

/**
 * The retranscription action's caption: "Retry" reads as recovery on a
 * failure, "Transcribe again" as a deliberate re-run on a success (including
 * an empty result the user wants another model's opinion on), and
 * "Transcribe" starts a never-attempted take's first run.
 */
export function transcribeAgainLabel(status: DictationSessionStatus): string {
  if (status === "failed") return "Retry";

  return status === "transcribed" ? "Transcribe again" : "Transcribe";
}

/**
 * The history row's title. An empty-text transcript is a finished, review-
 * needed result — the audio is intact and "Transcribe again" is right there —
 * so it must not wear the "Transcribing…" title of a take still in flight;
 * that mislabel is exactly the recovery dead end B04 removes.
 */
export function sessionTitle(session: DictationSession): string {
  if (session.transcript) return session.transcript.text || "Empty transcript";

  return session.status === "failed" ? "Saved. Retry available" : "Transcribing…";
}

/**
 * One earlier attempt's "model · protocol · when" provenance line for the
 * drawer. Attempts stored before provenance existed (or by a caller without
 * a known model) fall back to a plain label instead of a broken half-stamp.
 */
export function attemptProvenanceLabel(
  attempt: TranscriptAttempt,
  formatWhen: (iso: string) => string,
): string {
  const parts: string[] = [];

  if (attempt.model) parts.push(attempt.model);

  if (attempt.protocol) parts.push(attempt.protocol);

  if (attempt.savedAt) parts.push(formatWhen(attempt.savedAt));

  return parts.length > 0 ? parts.join(" · ") : "earlier attempt";
}

/** The "model, when" attribution stamped on every refined copy. */
export function refinedTranscriptStamp(refined: RefinedTranscript, when: string) {
  return `${refined.model}, ${when}`;
}

/** The separator line that appends a refined copy to a text export. */
function refinedTranscriptSeparator(refined: RefinedTranscript, iso: string) {
  return `--- REFINED TRANSCRIPT — ${refinedTranscriptStamp(refined, iso)} ---`;
}

/** The separator line that appends one earlier attempt to a text export. */
function attemptSeparator(attempt: TranscriptAttempt) {
  const parts: string[] = [];

  if (attempt.model) parts.push(attempt.model);

  if (attempt.protocol) parts.push(attempt.protocol);

  if (attempt.savedAt) parts.push(attempt.savedAt);

  const stamp = parts.join(", ");

  return stamp ? `--- EARLIER ATTEMPT — ${stamp} ---` : "--- EARLIER ATTEMPT ---";
}

/** One attempt's export body: its text, or an explicit empty marker. */
function attemptBody(attempt: TranscriptAttempt) {
  return attempt.text || "(empty transcript)";
}

/**
 * Compose the take's text export: the current raw transcript first and
 * intact, then every earlier recognition attempt under its own provenance
 * separator (oldest first, the order they were attempted), then the refined
 * copy last — so transcript versions, like the audio, stay associated with
 * the same session and leave together.
 */
export function transcriptExportText(session: DictationSession): string {
  const blocks: string[] = [session.transcript?.text || "(empty transcript)"];

  for (const attempt of session.transcriptHistory ?? []) {
    blocks.push(attemptSeparator(attempt), attemptBody(attempt));
  }

  if (session.refined) {
    blocks.push(
      refinedTranscriptSeparator(
        session.refined,
        new Date(session.refined.createdAt).toISOString(),
      ),
      session.refined.text,
    );
  }

  return `${blocks.join("\n\n")}\n`;
}
