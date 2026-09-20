import type { DictationSession } from "@starling/dictation";

/**
 * Pure thread logic for multi-turn refinement (#117). A thread is just a
 * `threadId` label plus its `threadJoinedAt` append stamp on sessions,
 * assigned only by an explicit "Refine in thread" press; these helpers read
 * that state out of the session listing so the UI never derives thread state
 * from anything but stored sessions. Every function is pure over its input —
 * no store, no React, no globals except the id generator's crypto fallback.
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
 * The reading order of one thread's turns: the explicit append sequence
 * (`threadJoinedAt`, stamped when the take was assigned to the thread), not
 * the recording clock. A thread is a conversation, and conversations grow by
 * appends — an older recording that joins an existing thread appends as the
 * thread's latest turn, so its refinement context is the thread's current
 * document and the next turn after it sees its edit (B11).
 *
 * Members persisted before the stamp existed have none: they joined before
 * the sequence did, so they read before every stamped join. That makes the
 * stamp comparison a tie exactly when a thread is fully legacy — no member
 * carries a stamp — and the tiebreak below then compares createdAt then id
 * for every pair: precisely the pre-B11 reading order. Existing threads
 * therefore keep the order their data was already sorted by instead of
 * regressing on upgrade, while mixed threads keep stamp-first semantics.
 * createdAt then id also tiebreak stamps shared in the same millisecond —
 * racing windows — so the order is always deterministic regardless of the
 * input array's order.
 */
function compareTakeOrder(left: DictationSession, right: DictationSession): number {
  const leftJoined = left.threadJoinedAt ?? Number.NEGATIVE_INFINITY;
  const rightJoined = right.threadJoinedAt ?? Number.NEGATIVE_INFINITY;

  // A tie here means both sides carry the same stamp, or neither carries
  // one — the fully-legacy case that falls back to the old order below.
  if (leftJoined !== rightJoined) return leftJoined - rightJoined;

  const byCreated = left.createdAt.localeCompare(right.createdAt);

  return byCreated !== 0 ? byCreated : left.id.localeCompare(right.id);
}

/**
 * Freshness order for picking the active thread, written ASCENDING like
 * every other comparator here: a negative number means `left` is older than
 * `right`, a positive number that `left` is newer (updatedAt first, then
 * createdAt, then id as tiebreaks). The caller — activeThreadId — selects
 * the newest by keeping the session for which this reports greater than
 * zero against the current winner. updatedAt leads because pressing
 * "Refine in thread" bumps it: the thread acted on most recently is the one
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
 * One thread's turns in reading order: members sorted by append sequence
 * (see compareTakeOrder), first append to latest. Sessions outside the
 * thread are ignored.
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
 * The refinement base for a take about to be refined inside its thread: the
 * nearest earlier turn that HAS a refined copy, returned as the member
 * itself so the caller can both read its text and record its id — the
 * captured base identity (B11) that keeps a rerun's base stable and
 * explainable. undefined when there is none — the thread head, or nobody
 * refined yet — in which case a threaded refine behaves like a standalone
 * one.
 *
 * "Earlier" follows the thread's reading order (see compareTakeOrder), so a
 * joining take — appended last whatever its recording age — refines against
 * the thread's current document. A `beforeTakeId` that is not (yet) a member
 * — its assignment still in flight — counts every current member as earlier,
 * which is exactly the join-an-existing-thread case the drawer computes
 * before assigning; once the assignment stamps its append, the take is the
 * latest turn and computes the same base, so a retry never sees a different
 * answer than the press that joined.
 */
export function threadContextBase(
  sessions: readonly DictationSession[],
  threadId: string,
  beforeTakeId: string,
): DictationSession | undefined {
  const turns = threadTurns(sessions, threadId);
  const boundary = turns.findIndex((turn) => turn.id === beforeTakeId);
  const earlier = boundary === -1 ? turns : turns.slice(0, boundary);

  // Walk backwards: the nearest refined predecessor is the thread's current
  // text, and members without a usable refined copy are skipped, not
  // blocking. Whitespace-only text carries no thread state — downstream it
  // would trim to no context — so it is skipped like an unrefined member and
  // the walk continues to the previous real text. (Empty text cannot reach
  // here at all: the storage schema treats it as damage and quarantines the
  // record out of the listing.)
  for (let index = earlier.length - 1; index >= 0; index -= 1) {
    const member = earlier[index];
    const refined = member?.refined;

    if (refined !== undefined && refined.text.trim() !== "") return member;
  }

  return undefined;
}

/**
 * The refinement context text for a take about to be refined inside its
 * thread: `threadContextBase`'s member's refined text, or undefined when
 * there is no base.
 */
export function threadContext(
  sessions: readonly DictationSession[],
  threadId: string,
  beforeTakeId: string,
): string | undefined {
  return threadContextBase(sessions, threadId, beforeTakeId)?.refined?.text;
}
