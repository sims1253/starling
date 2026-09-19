import { describe, expect, it } from "vite-plus/test";

import type { DictationSession } from "@starling/dictation";
import { activeThreadId, newThreadId, threadContext, threadTurns } from "./threads";

interface Fixture {
  readonly id: string;
  readonly createdAt: string;
  readonly updatedAt?: string;
  readonly threadId?: string;
  readonly refinedText?: string;
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
});

describe("threadContext", () => {
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

  it("returns the nearest earlier refined text in the thread", () => {
    expect(threadContext(sessions, "t1", "turn-4")).toBe("Turn three, refined.");
    expect(threadContext(sessions, "t1", "turn-3")).toBe("Turn one, refined.");
    expect(threadContext(sessions, "t1", "turn-2")).toBe("Turn one, refined.");
  });

  it("is undefined for the thread head or when nobody refined yet", () => {
    expect(threadContext(sessions, "t1", "turn-1")).toBeUndefined();

    const unrefined = [
      take({ id: "u1", createdAt: "2025-09-01T10:00:00.000Z", threadId: "t1" }),
      take({ id: "u2", createdAt: "2025-09-01T10:01:00.000Z", threadId: "t1" }),
    ];

    expect(threadContext(unrefined, "t1", "u2")).toBeUndefined();
    expect(threadContext([], "t1", "u2")).toBeUndefined();
    expect(threadContext(sessions, "missing", "turn-4")).toBeUndefined();
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
    expect(threadContext(mixed, "t1", "m4")).toBe("Oldest refined.");
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

    expect(threadContext(blank, "t1", "w3")).toBe("Real text.");

    const onlyBlank = [
      take({
        id: "b1",
        createdAt: "2025-09-01T10:00:00.000Z",
        threadId: "t1",
        refinedText: "\n\t ",
      }),
      take({ id: "b2", createdAt: "2025-09-01T10:01:00.000Z", threadId: "t1" }),
    ];

    expect(threadContext(onlyBlank, "t1", "b2")).toBeUndefined();
  });

  it("ignores refined takes that are not members of the thread", () => {
    expect(threadContext(sessions, "t1", "turn-3")).not.toBe("A different thread entirely.");

    // A take outside the thread must never supply or block context.
    expect(threadContext(sessions, "t2", "other-thread")).toBeUndefined();
  });

  it("treats every member as earlier when the boundary take is not yet a member", () => {
    // The join case: context is computed for an unthreaded take right before
    // its assignment lands, so its id is absent from the member list.
    expect(threadContext(sessions, "t1", "about-to-join")).toBe("Turn three, refined.");
  });
});
