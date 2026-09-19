import type { DictationSession } from "@starling/dictation";

/**
 * Pure thread logic for multi-turn refinement (#117). A thread is just a
 * `threadId` label on sessions, assigned only by an explicit "Refine in
 * thread" press; these helpers read that label out of the session listing so
 * the UI never derives thread state from anything but stored sessions. Every
 * function is pure over its input — no store, no React, no globals except
 * the id generator's crypto fallback.
 */

/**
 * Fresh thread identity, following the session-id convention: a UUID when
 * the host provides crypto.randomUUID, otherwise a time-and-random base36
 * fallback. Thread ids are opaque labels; only the first 8 characters are
 * shown in the UI.
 */
export function newThreadId(): string {
  return (
    globalThis.crypto?.randomUUID?.() ??
    `thread-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
  );
}

/**
 * The reading order of one thread's turns. createdAt is the history pane's
 * ordering key; a thread reads oldest-first (turn 1 → N), so this is the
 * pane's order reversed. The id tiebreak keeps takes that share a timestamp
 * deterministic regardless of the input array's order.
 */
function compareTakeOrder(left: DictationSession, right: DictationSession): number {
  const byCreated = left.createdAt.localeCompare(right.createdAt);

  return byCreated !== 0 ? byCreated : left.id.localeCompare(right.id);
}

/**
 * Freshness order for picking the active thread: updatedAt descending, then
 * createdAt and id descending as tiebreaks. updatedAt leads because pressing
 * "Refine in thread" bumps it — the thread acted on most recently is the one
 * the next unthreaded take should join.
 */
function compareFreshness(left: DictationSession, right: DictationSession): number {
  const byUpdated = left.updatedAt.localeCompare(right.updatedAt);

  if (byUpdated !== 0) return byUpdated;

  const byCreated = left.createdAt.localeCompare(right.createdAt);

  if (byCreated !== 0) return byCreated;

  return left.id.localeCompare(right.id);
}

/**
 * The thread of the most recently updated session that has one, or undefined
 * when no session is threaded. Derived purely from stored sessions, so it
 * survives reloads and stays correct across windows.
 */
export function activeThreadId(sessions: readonly DictationSession[]): string | undefined {
  let winner: DictationSession | undefined;

  for (const session of sessions) {
    if (session.threadId === undefined) continue;

    if (winner === undefined || compareFreshness(session, winner) > 0) winner = session;
  }

  return winner?.threadId;
}

/**
 * One thread's turns in reading order: members sorted by the history pane's
 * ordering key, oldest first. Sessions outside the thread are ignored.
 */
export function threadTurns(
  sessions: readonly DictationSession[],
  threadId: string,
): readonly DictationSession[] {
  // filter() copies before sort(), so the caller's array is never reordered.
  return Object.freeze(
    sessions.filter((session) => session.threadId === threadId).sort(compareTakeOrder),
  );
}

/**
 * The refinement context for a take about to be refined inside its thread:
 * the refined text of the most recent earlier turn that HAS a refined copy.
 * undefined when there is none — the thread head, or nobody refined yet — in
 * which case a threaded refine behaves like a standalone one.
 *
 * "Earlier" follows the thread's reading order (see compareTakeOrder), the
 * same ordering the history pane's newest-first listing reverses. A
 * `beforeTakeId` that is not (yet) a member — its assignment still in
 * flight — counts every current member as earlier, which is exactly the
 * join-an-existing-thread case the drawer computes before assigning.
 */
export function threadContext(
  sessions: readonly DictationSession[],
  threadId: string,
  beforeTakeId: string,
): string | undefined {
  const turns = threadTurns(sessions, threadId);
  const boundary = turns.findIndex((turn) => turn.id === beforeTakeId);
  const earlier = boundary === -1 ? turns : turns.slice(0, boundary);

  // Walk backwards: the nearest refined predecessor is the thread's current
  // text, and members without a refined copy are skipped, not blocking.
  for (let index = earlier.length - 1; index >= 0; index -= 1) {
    const refined = earlier[index]?.refined;

    if (refined !== undefined) return refined.text;
  }

  return undefined;
}
