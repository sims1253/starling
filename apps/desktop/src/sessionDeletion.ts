import type { DictationSession } from "@starling/dictation";

/**
 * Explicit confirmation for deleting a saved recording (B05).
 *
 * The trash button used to call store.delete() directly: one click — mouse
 * or keyboard — permanently erased a perfectly good take, its audio, and
 * every transcript version, with no confirmation and no undo, while the
 * damaged-record path at least asked first. Deletion now goes through this
 * guard: the button only opens a confirmation dialog, the destructive write
 * happens solely on its explicit confirm, and a take whose transcription or
 * refinement is still in flight cannot even be proposed for deletion.
 */

export const DELETE_WHILE_TRANSCRIBING =
  "This take is still transcribing. Wait for it to finish, then delete it.";

export const DELETE_WHILE_REFINING =
  "This take is still refining. Wait for it to finish, then delete it.";

/** What the requesting window knows about the take's in-flight work. */
export interface DeleteTakeActivity {
  /** True while a transcription attempt on this take is running. */
  readonly transcribing: boolean;
  /** True while a refinement (standalone or threaded) on this take is running. */
  readonly refining: boolean;
}

export type SessionDeletionIntent =
  | { readonly kind: "blocked"; readonly message: string }
  | { readonly kind: "confirm"; readonly id: string };

/**
 * The dialog's open state plus the exactly-once handoff of the confirmed
 * id: `confirm()` returns the take to delete a single time, so a stuck
 * double activation cannot re-delete, and `cancel()` always leaves every
 * recording untouched.
 */
export class SessionDeleteDialog {
  private pendingId: string | undefined;

  /** The take awaiting its explicit confirm, when the dialog is open. */
  requested(): string | undefined {
    return this.pendingId;
  }

  /**
   * Open the dialog for one take — unless its work is still in flight, in
   * which case nothing opens and the blocked intent carries the reason the
   * caller should surface.
   */
  request(id: string, activity: DeleteTakeActivity): SessionDeletionIntent {
    if (activity.transcribing) return { kind: "blocked", message: DELETE_WHILE_TRANSCRIBING };

    if (activity.refining) return { kind: "blocked", message: DELETE_WHILE_REFINING };

    this.pendingId = id;

    return { kind: "confirm", id };
  }

  /** Close the dialog without deleting anything. */
  cancel(): void {
    this.pendingId = undefined;
  }

  /** Close the dialog and hand out the id to delete — exactly once. */
  confirm(): string | undefined {
    const id = this.pendingId;

    this.pendingId = undefined;

    return id;
  }
}

/**
 * What the confirmation tells the user is at stake: the audio file and every
 * transcript version saved beside it — the current result, earlier
 * recognition attempts, and any refined copy — so the scope of the loss is
 * explicit before the one irreversible step.
 */
export function deletionWarning(session: DictationSession): string {
  const removed: string[] = ["the audio"];

  if (session.transcript !== undefined || (session.transcriptHistory?.length ?? 0) > 0) {
    removed.push("every saved transcript version");
  }

  if (session.refined !== undefined) removed.push("the refined copy");

  return `Delete this saved recording? This permanently removes ${removed.join(", ")}. There is no undo.`;
}
