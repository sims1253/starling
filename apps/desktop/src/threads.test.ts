import { describe, expect, it } from "vite-plus/test";

import type { DictationSession } from "@starling/dictation";
import { activeThreadId, newThreadId, threadContextBase, threadTurns } from "./threads";

interface Fixture {
  readonly id: string;
  readonly createdAt: string;
  readonly updatedAt?: string;
  readonly threadId?: string;
  readonly refinedText?: string;
  readonly threadJoinedAt?: number;
}

/**
 * Mutable draft for fixture sessions — the SessionDraft pattern from
 * storage.ts — so optional fields are added only when present and the
 * result is frozen on return like a decoded session.
 */
interface FixtureDraft {
  id: string;
  createdAt: string;
  updatedAt: string;
  status: "transcribed";
  wav: Blob;
  attemptCount: number;
  transcript: { text: string; segments: [] };
  threadId?: string;
  threadJoinedAt?: number;
  refined?: { text: string; model: string; createdAt: number };
}

/** Minimal transcribed session literal — the only fields thread logic reads. */
function take(fields: Fixture): DictationSession {
  const session: FixtureDraft = {
    id: fields.id,
    createdAt: fields.createdAt,
    updatedAt: fields.updatedAt ?? fields.createdAt,
    status: "transcribed",
    wav: new Blob(),
    attemptCount: 1,
    transcript: { text: `raw ${fields.id}`, segments: [] },
  };

  if (fields.threadId !== undefined) session.threadId = fields.threadId;

  if (fields.threadJoinedAt !== undefined) session.threadJoinedAt = fields.threadJoinedAt;

  if (fields.refinedText !== undefined) {
    session.refined = { text: fields.refinedText, model: "llama3.1", createdAt: 1 };
  }

  return Object.freeze(session);
}

describe("newThreadId", () => {
  it("mints unique non-empty ids", () => {
    const first = newThreadId();
    const second = newThreadId();

    expect(first).not.toBe("");
    expect(second).not.toBe("");
    expect(first).not.toBe(second);
  });
});

describe("activeThreadId", () => {
  it("is undefined when nothing is threaded", () => {
    expect(activeThreadId([])).toBeUndefined();

    expect(
      activeThreadId([
        take({ id: "a", createdAt: "2025-09-01T10:00:00.000Z" }),
        take({ id: "b", createdAt: "2025-09-01T10:01:00.000Z" }),
      ]),
    ).toBeUndefined();
  });

  it("follows the most recently updated threaded session", () => {
    expect(
      activeThreadId([
        take({
          id: "older",
          createdAt: "2025-09-01T10:00:00.000Z",
          updatedAt: "2025-09-01T10:00:05.000Z",
          threadId: "thread-old",
        }),
        take({
          id: "newer",
          createdAt: "2025-09-01T09:00:00.000Z",
          updatedAt: "2025-09-01T11:00:00.000Z",
          threadId: "thread-new",
        }),
      ]),
    ).toBe("thread-new");
  });

  it("breaks updatedAt ties deterministically without depending on array order", () => {
    const tied = [
      take({ id: "a", createdAt: "2025-09-01T10:00:00.000Z", threadId: "thread-a" }),
      take({ id: "b", createdAt: "2025-09-01T10:00:00.000Z", threadId: "thread-b" }),
    ];

    // Same updatedAt (both default to createdAt): the greater createdAt wins
    // first, and the greater id breaks a full tie — both orders agree.
    expect(activeThreadId(tied)).toBe("thread-b");
    expect(activeThreadId([...tied].reverse())).toBe("thread-b");

    const fresh = [
      take({
        id: "z",
        createdAt: "2025-09-01T09:00:00.000Z",
        updatedAt: "2025-09-01T10:00:00.000Z",
        threadId: "thread-z",
      }),
      take({
        id: "y",
        createdAt: "2025-09-01T08:00:00.000Z",
        updatedAt: "2025-09-01T10:00:00.000Z",
        threadId: "thread-y",
      }),
    ];

    expect(activeThreadId(fresh)).toBe("thread-z");
    expect(activeThreadId([...fresh].reverse())).toBe("thread-z");
  });

  it("ignores unthreaded sessions even when they were updated last", () => {
    expect(
      activeThreadId([
        take({
          id: "threaded",
          createdAt: "2025-09-01T10:00:00.000Z",
          updatedAt: "2025-09-01T10:00:01.000Z",
          threadId: "thread-live",
        }),
        take({
          id: "unthreaded",
          createdAt: "2025-09-01T11:00:00.000Z",
          updatedAt: "2025-09-01T11:00:05.000Z",
        }),
      ]),
    ).toBe("thread-live");
  });
});

describe("threadTurns", () => {
  it("orders members oldest-first and ignores takes outside the thread", () => {
    const sessions = [
      take({ id: "third", createdAt: "2025-09-01T10:02:00.000Z", threadId: "t1" }),
      take({ id: "outsider", createdAt: "2025-09-01T10:03:00.000Z", threadId: "t2" }),
      take({ id: "first", createdAt: "2025-09-01T10:00:00.000Z", threadId: "t1" }),
      take({ id: "unthreaded", createdAt: "2025-09-01T10:04:00.000Z" }),
      take({ id: "second", createdAt: "2025-09-01T10:01:00.000Z", threadId: "t1" }),
    ];

    expect(threadTurns(sessions, "t1").map((turn) => turn.id)).toEqual([
      "first",
      "second",
      "third",
    ]);
    expect(threadTurns(sessions, "missing").map((turn) => turn.id)).toEqual([]);
  });

  it("breaks shared timestamps by id so the order never depends on input order", () => {
    const tied = [
      take({ id: "b", createdAt: "2025-09-01T10:00:00.000Z", threadId: "t1" }),
      take({ id: "a", createdAt: "2025-09-01T10:00:00.000Z", threadId: "t1" }),
      take({ id: "c", createdAt: "2025-09-01T09:59:59.000Z", threadId: "t1" }),
    ];

    const ids = threadTurns(tied, "t1").map((turn) => turn.id);

    expect(ids).toEqual(["c", "a", "b"]);
    expect(threadTurns([...tied].reverse(), "t1").map((turn) => turn.id)).toEqual(ids);
  });

  it("never reorders the caller's array", () => {
    const sessions = [
      take({ id: "later", createdAt: "2025-09-01T10:01:00.000Z", threadId: "t1" }),
      take({ id: "earlier", createdAt: "2025-09-01T10:00:00.000Z", threadId: "t1" }),
    ];

    void threadTurns(sessions, "t1");

    expect(sessions.map((session) => session.id)).toEqual(["later", "earlier"]);
  });

  it("reads a thread in append order, so an older take that joins later appends last (B11)", () => {
    // B recorded at 10:00 opened the thread; C (08:00) and A (09:00) are both
    // older recordings that joined afterwards, C first. The reading order is
    // the order takes were appended, not the order they were recorded.
    const sessions = [
      take({
        id: "a",
        createdAt: "2025-09-01T09:00:00.000Z",
        threadId: "t1",
        threadJoinedAt: 300,
      }),
      take({
        id: "b",
        createdAt: "2025-09-01T10:00:00.000Z",
        threadId: "t1",
        threadJoinedAt: 100,
      }),
      take({
        id: "c",
        createdAt: "2025-09-01T08:00:00.000Z",
        threadId: "t1",
        threadJoinedAt: 200,
      }),
    ];

    expect(threadTurns(sessions, "t1").map((turn) => turn.id)).toEqual(["b", "c", "a"]);
    expect(threadTurns([...sessions].reverse(), "t1").map((turn) => turn.id)).toEqual([
      "b",
      "c",
      "a",
    ]);
  });

  it("keeps createdAt order for members persisted before the append sequence existed", () => {
    // Legacy members carry no threadJoinedAt: they joined before the field
    // existed, so they sort before every stamped join and keep their old
    // createdAt order among themselves.
    const sessions = [
      take({
        id: "stamped",
        createdAt: "2025-09-01T11:00:00.000Z",
        threadId: "t1",
        threadJoinedAt: 1,
      }),
      take({ id: "legacy-new", createdAt: "2025-09-01T10:00:00.000Z", threadId: "t1" }),
      take({ id: "legacy-old", createdAt: "2025-09-01T09:00:00.000Z", threadId: "t1" }),
    ];

    expect(threadTurns(sessions, "t1").map((turn) => turn.id)).toEqual([
      "legacy-old",
      "legacy-new",
      "stamped",
    ]);
  });

  it("reads a fully-legacy thread in exactly the old createdAt order (R14)", () => {
    // A thread where NO member has a stamp — every join predates the append
    // sequence — must keep the reading order such a thread always had:
    // createdAt then id. The true join sequence is unrecoverable, and the
    // old order is what the user last saw, so a fully-legacy thread must not
    // reorder on upgrade. Here the 09:00 recording joined the thread LAST
    // (after the 10:00 and 11:00 takes) yet still reads first — the pre-B11
    // behavior existing data was sorted by.
    const sessions = [
      take({
        id: "last-join",
        createdAt: "2025-09-01T09:00:00.000Z",
        threadId: "t1",
      }),
      take({
        id: "middle",
        createdAt: "2025-09-01T10:00:00.000Z",
        threadId: "t1",
      }),
      take({
        id: "head",
        createdAt: "2025-09-01T11:00:00.000Z",
        threadId: "t1",
      }),
    ];

    const ids = threadTurns(sessions, "t1").map((turn) => turn.id);

    expect(ids).toEqual(["last-join", "middle", "head"]);
    expect(threadTurns([...sessions].reverse(), "t1").map((turn) => turn.id)).toEqual(ids);
  });

  it("keeps stamp-first semantics once any member carries a stamp (R14)", () => {
    // The same legacy members plus one stamped join: the mixed thread is
    // where the append sequence takes over. The legacy members all predate
    // the sequence, so they read first in createdAt order — including one
    // whose recording is older than the stamped join's — and the stamped
    // member reads as the latest append whatever its recording age.
    const sessions = [
      take({
        id: "stamped-join",
        createdAt: "2025-09-01T08:00:00.000Z",
        threadId: "t1",
        threadJoinedAt: 900,
      }),
      take({
        id: "legacy-newer",
        createdAt: "2025-09-01T11:00:00.000Z",
        threadId: "t1",
      }),
      take({
        id: "legacy-older",
        createdAt: "2025-09-01T10:00:00.000Z",
        threadId: "t1",
      }),
    ];

    expect(threadTurns(sessions, "t1").map((turn) => turn.id)).toEqual([
      "legacy-older",
      "legacy-newer",
      "stamped-join",
    ]);
  });

  it("breaks equal append stamps deterministically without depending on input order", () => {
    // Two windows can append within the same millisecond; the tiebreak is
    // createdAt, then id — the same keys the legacy order uses.
    const tied = [
      take({ id: "n", createdAt: "2025-09-01T10:00:00.000Z", threadId: "t1", threadJoinedAt: 50 }),
      take({ id: "m", createdAt: "2025-09-01T09:00:00.000Z", threadId: "t1", threadJoinedAt: 50 }),
      take({ id: "o", createdAt: "2025-09-01T11:00:00.000Z", threadId: "t1", threadJoinedAt: 40 }),
    ];

    const ids = threadTurns(tied, "t1").map((turn) => turn.id);

    expect(ids).toEqual(["o", "m", "n"]);
    expect(threadTurns([...tied].reverse(), "t1").map((turn) => turn.id)).toEqual(ids);
  });
});

describe("threadContextBase", () => {
  const sessions = [
    take({
      id: "turn-1",
      createdAt: "2025-09-01T10:00:00.000Z",
      threadId: "t1",
      refinedText: "Turn one, refined.",
    }),
    take({ id: "turn-2", createdAt: "2025-09-01T10:01:00.000Z", threadId: "t1" }),
    take({
      id: "turn-3",
      createdAt: "2025-09-01T10:02:00.000Z",
      threadId: "t1",
      refinedText: "Turn three, refined.",
    }),
    take({ id: "turn-4", createdAt: "2025-09-01T10:03:00.000Z", threadId: "t1" }),
    take({
      id: "other-thread",
      createdAt: "2025-09-01T10:02:30.000Z",
      threadId: "t2",
      refinedText: "A different thread entirely.",
    }),
  ];

  it("returns the nearest earlier refined member, whose text is the context", () => {
    expect(threadContextBase(sessions, "t1", "turn-4")?.refined?.text).toBe("Turn three, refined.");
    expect(threadContextBase(sessions, "t1", "turn-3")?.refined?.text).toBe("Turn one, refined.");
    expect(threadContextBase(sessions, "t1", "turn-2")?.refined?.text).toBe("Turn one, refined.");
  });

  it("is undefined for the thread head or when nobody refined yet", () => {
    expect(threadContextBase(sessions, "t1", "turn-1")).toBeUndefined();

    const unrefined = [
      take({ id: "u1", createdAt: "2025-09-01T10:00:00.000Z", threadId: "t1" }),
      take({ id: "u2", createdAt: "2025-09-01T10:01:00.000Z", threadId: "t1" }),
    ];

    expect(threadContextBase(unrefined, "t1", "u2")).toBeUndefined();
    expect(threadContextBase([], "t1", "u2")).toBeUndefined();
    expect(threadContextBase(sessions, "missing", "turn-4")).toBeUndefined();
  });

  it("skips unrefined members instead of stopping at them", () => {
    const mixed = [
      take({
        id: "m1",
        createdAt: "2025-09-01T10:00:00.000Z",
        threadId: "t1",
        refinedText: "Oldest refined.",
      }),
      take({ id: "m2", createdAt: "2025-09-01T10:01:00.000Z", threadId: "t1" }),
      take({ id: "m3", createdAt: "2025-09-01T10:02:00.000Z", threadId: "t1" }),
      take({ id: "m4", createdAt: "2025-09-01T10:03:00.000Z", threadId: "t1" }),
    ];

    // m3 never refined; the context for m4 reaches back to m1.
    expect(threadContextBase(mixed, "t1", "m4")?.refined?.text).toBe("Oldest refined.");
  });

  it("skips whitespace-only refined text instead of treating it as context", () => {
    // The storage schema quarantines EMPTY refined text as damage, so only
    // the whitespace-only shape can reach the walk — it carries no thread
    // state and must be skipped like an unrefined member, not returned (and
    // not allowed to shadow the previous real text).
    const blank = [
      take({
        id: "w1",
        createdAt: "2025-09-01T10:00:00.000Z",
        threadId: "t1",
        refinedText: "Real text.",
      }),
      take({
        id: "w2",
        createdAt: "2025-09-01T10:01:00.000Z",
        threadId: "t1",
        refinedText: "   ",
      }),
      take({ id: "w3", createdAt: "2025-09-01T10:02:00.000Z", threadId: "t1" }),
    ];

    expect(threadContextBase(blank, "t1", "w3")?.refined?.text).toBe("Real text.");

    const onlyBlank = [
      take({
        id: "b1",
        createdAt: "2025-09-01T10:00:00.000Z",
        threadId: "t1",
        refinedText: "\n\t ",
      }),
      take({ id: "b2", createdAt: "2025-09-01T10:01:00.000Z", threadId: "t1" }),
    ];

    expect(threadContextBase(onlyBlank, "t1", "b2")).toBeUndefined();
  });

  it("ignores refined takes that are not members of the thread", () => {
    expect(threadContextBase(sessions, "t1", "turn-3")?.refined?.text).not.toBe(
      "A different thread entirely.",
    );

    // A take outside the thread must never supply or block context.
    expect(threadContextBase(sessions, "t2", "other-thread")).toBeUndefined();
  });

  it("treats every member as earlier when the boundary take is not yet a member", () => {
    // The join case: context is computed for an unthreaded take right before
    // its assignment lands, so its id is absent from the member list.
    expect(threadContextBase(sessions, "t1", "about-to-join")?.refined?.text).toBe(
      "Turn three, refined.",
    );
  });

  it("keeps an appended older take's base stable from join through retry (B11)", () => {
    // A recorded at 09:00, unthreaded. B recorded at 10:00 and is the refined
    // head of thread t1 (appended at 100).
    const head = take({
      id: "b",
      createdAt: "2025-09-01T10:00:00.000Z",
      threadId: "t1",
      refinedText: "B, refined.",
      threadJoinedAt: 100,
    });

    const older = take({ id: "a", createdAt: "2025-09-01T09:00:00.000Z" });

    // First press: the context is computed before the assignment lands, so
    // the joining take counts every member as earlier and picks B.
    expect(threadContextBase([head, older], "t1", "a")?.id).toBe("b");
    expect(threadContextBase([head, older], "t1", "a")?.refined?.text).toBe("B, refined.");

    // The assignment stamps the append sequence (A appended at 200). A retry
    // — after a failure, a cancel, or a reload — computes the same base: A is
    // the thread's latest append, so B is still its nearest predecessor.
    const joined = take({
      id: "a",
      createdAt: "2025-09-01T09:00:00.000Z",
      threadId: "t1",
      threadJoinedAt: 200,
    });

    expect(threadContextBase([head, joined], "t1", "a")?.id).toBe("b");
    expect(threadContextBase([head, joined], "t1", "a")?.refined?.text).toBe("B, refined.");

    // A's completed edit is now the thread's current document: the next turn
    // appended after it (C, at 300) includes that edit, not the stale head.
    const appended = {
      ...joined,
      refined: { text: "A, refined.", model: "llama3.1", createdAt: 1 },
    };

    const next = take({ id: "c", createdAt: "2025-09-01T11:00:00.000Z" });

    expect(threadContextBase([head, appended, next], "t1", "c")?.id).toBe("a");
    expect(threadContextBase([head, appended, next], "t1", "c")?.refined?.text).toBe("A, refined.");

    // Deletion of the appended member falls back to the previous refined
    // member instead of inventing a base.
    expect(threadContextBase([head, next], "t1", "c")?.id).toBe("b");
    expect(threadContextBase([head, next], "t1", "c")?.refined?.text).toBe("B, refined.");
  });

  it("returns the member whose refined text is the base, so the base identity is capturable", () => {
    const identity = [
      take({
        id: "head",
        createdAt: "2025-09-01T10:00:00.000Z",
        threadId: "t1",
        refinedText: "Head, refined.",
        threadJoinedAt: 100,
      }),
      take({
        id: "appended",
        createdAt: "2025-09-01T09:00:00.000Z",
        threadId: "t1",
        refinedText: "Appended, refined.",
        threadJoinedAt: 200,
      }),
    ];

    // The appended take — recorded earlier, joined later — refines against
    // the head; the next member refines against the appended take.
    expect(threadContextBase(identity, "t1", "appended")?.id).toBe("head");
    expect(threadContextBase(identity, "t1", "appended")?.refined?.text).toBe("Head, refined.");
    expect(threadContextBase(identity, "t1", "unjoined")?.id).toBe("appended");

    // No refined predecessor, or no thread: no base to capture.
    expect(threadContextBase(identity, "t1", "head")).toBeUndefined();
    expect(threadContextBase([], "t1", "head")).toBeUndefined();
  });
});
