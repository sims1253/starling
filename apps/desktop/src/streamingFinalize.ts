import type { DictationSession, TranscriptionResult } from "@starling/dictation";
import type { StreamingDictation } from "./streamingDictation";

export interface StreamedFinalize {
  /** True when the stream delivered the final transcript itself. */
  readonly streamed: boolean;
  /** Present when the take was persisted and transcribed. */
  readonly session?: DictationSession;
  readonly transcript?: TranscriptionResult;
  /** Present when no durable write happened because the take was discarded. */
  readonly discarded?: boolean;
  /** Why the take fell back to the batch path, for session visibility. */
  readonly streamNote?: string;
  /** True when the caller should save the recorder capture via batch. */
  readonly batchFallback: boolean;
}

/** The durable writes a streamed finalize performs, injected for tests. */
export interface StreamingFinalizeStore {
  saveTranscript(
    id: string,
    transcript: TranscriptionResult,
    options?: { streamed?: boolean; protocol?: string },
  ): Promise<DictationSession>;
  noteStreamError(id: string, message: string): Promise<DictationSession>;
  delete(id: string): Promise<void>;
}

export interface StreamingFinalizeDeps {
  readonly stream: StreamingDictation;
  readonly durationMs: number;
  /** The take generation Stop settled on; false once Discard invalidates it (#160, B03). */
  readonly isCurrentTake: () => boolean;
  readonly parkUnsavedWav: (wav: Blob) => void;
  readonly refresh: () => Promise<void>;
  readonly setSelectedId: (id: string) => void;
  readonly setConnectionReady: () => void;
  readonly transcribe: (session: DictationSession) => Promise<void>;
  /**
   * Invoked once the journal is durably owned by its session, before the
   * transcript work (B03): the caller ends the capture transition here, so a
   * new take can start while transcription is still in flight. Never invoked
   * for a discarded take or a failed durable save.
   */
  readonly onDurableSave?: () => void;
  /**
   * Invoked after a streamed transcript settles durably on its session (E29
   * insight wiring), while the finalize still owns the take: receives the
   * settled session and the streamed transcript. Optional and additive, like
   * onDurableSave; never invoked for a discarded take.
   */
  readonly onStreamedSettled?: (
    session: DictationSession,
    transcript: TranscriptionResult,
  ) => void;
}

function messageFrom(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

/**
 * Remove the provisional row this finalize created — never the state a
 * Discard settled — and resync the list the row may already appear in.
 */
async function dropProvisional(
  store: StreamingFinalizeStore,
  deps: StreamingFinalizeDeps,
  id: string,
): Promise<void> {
  await store.delete(id).catch(() => {});
  await deps.refresh();
}

/**
 * Finalize the streamed take Stop was issued against, honouring a close-guard
 * Discard that lands while the finalize is in flight (#160).
 *
 * The generation is captured by the caller before any await and re-validated
 * around each durable write; a stale finalize performs no session write —
 * no saveTranscript, no noteStreamError, no batch transcribe — cleans up
 * anything provisional its own finish created, undoes its own completed
 * writes (the streamed save or the batch transcription) when the Discard
 * lands mid-write, and never falls through to the batch path. The state
 * the Discard settled is left untouched.
 *
 * Returns batchFallback when the stream was never usable and the caller
 * should save the recorder's own capture via the batch path.
 *
 * `deps.onDurableSave` fires the moment the journal is durably owned by its
 * session — before the commit, the transcript write, or any batch
 * transcription — so the caller can release the recording lifecycle while
 * inference is still in flight (B03).
 */
export async function finishStreamingTake(
  deps: StreamingFinalizeDeps,
  store: StreamingFinalizeStore,
): Promise<StreamedFinalize> {
  const { stream, durationMs, isCurrentTake } = deps;

  if (stream.journaledChunkCount === 0) {
    // The microphone never reached this controller's journal — its session
    // would be a header-only WAV: drop the empty journal and keep the
    // recorder's capture through the batch path (#143).
    await stream.abandon().catch(() => {});

    // A Discard that landed while the journal was dropped must not fall
    // through to the batch path either: the recorder capture is discarded
    // with the take.
    if (!isCurrentTake()) {
      return { streamed: false, discarded: true, batchFallback: false };
    }

    return { streamed: false, batchFallback: true };
  }

  const result = await stream.finish(
    durationMs,
    () => !isCurrentTake(),
    () => deps.onDurableSave?.(),
  );

  if (result.session === undefined) {
    if (!isCurrentTake()) {
      // The journal drained into a Discard: the capture layer already
      // dropped the journal, so nothing was written and there is nothing
      // to undo (#160).
      return { streamed: false, discarded: true, batchFallback: false };
    }

    // The durable save failed; the assembled WAV is the only copy.
    deps.parkUnsavedWav(result.wav);
    throw new Error(
      `Local storage failed: ${messageFrom(result.failure ?? "unknown storage failure")} Keep this window open and download the unsaved WAV to recover it.`,
    );
  }

  if (!isCurrentTake()) {
    // The Discard landed after the capture layer's own probe: remove only
    // the provisional row this finalize created, never touching the state
    // the Discard settled (#160).
    await dropProvisional(store, deps, result.session.id);

    return { streamed: false, discarded: true, batchFallback: false };
  }

  await deps.refresh();

  // A Discard that lands between the save and the transcript write settles
  // the session without the take: skip the write and remove the provisional
  // row this finalize created, instead of transcribing a dropped take.
  if (!isCurrentTake()) {
    await dropProvisional(store, deps, result.session.id);

    return { streamed: false, discarded: true, batchFallback: false };
  }

  if (result.streamed && result.transcript) {
    // The streamed attempt ran on the Starling native protocol by
    // construction — streaming is Starling-only — so its transcript is
    // settled with that provenance (B04).
    await store.saveTranscript(result.session.id, result.transcript, {
      streamed: true,
      protocol: "starling",
    });

    // A Discard that landed during the save settles the session without
    // the take: undo this finalize's completed write instead of surfacing
    // the dropped take.
    if (!isCurrentTake()) {
      await dropProvisional(store, deps, result.session.id);

      return { streamed: false, discarded: true, batchFallback: false };
    }

    deps.onStreamedSettled?.(result.session, result.transcript);
    deps.setSelectedId(result.session.id);
    deps.setConnectionReady();
    await deps.refresh();
  } else {
    if (result.streamNote) {
      await store.noteStreamError(result.session.id, result.streamNote).catch(() => {});
    }

    if (!isCurrentTake()) {
      await dropProvisional(store, deps, result.session.id);

      return { streamed: false, discarded: true, batchFallback: false };
    }

    await deps.transcribe(result.session);

    // A Discard that landed while the batch transcription ran settles the
    // session without the take: undo its completed writes the same way.
    if (!isCurrentTake()) {
      await dropProvisional(store, deps, result.session.id);

      return { streamed: false, discarded: true, batchFallback: false };
    }
  }

  return { streamed: result.streamed, session: result.session, batchFallback: false };
}
