package dev.starling.mobile.transcription

import android.os.Handler
import android.os.Looper
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
import dev.starling.mobile.data.TranscriptSource
import dev.starling.mobile.data.TranscriptionProvenance
import dev.starling.mobile.engine.OnDeviceBackend
import dev.starling.mobile.storage.RecordingStore
import java.io.File
import java.io.IOException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Transcribes only after local audio finalization and persists every outcome.
 * Every attempt, first or retry, transcribes the same saved audio; a success
 * adds a transcript revision and a failure leaves the audio and earlier
 * revisions as they were (#356).
 *
 * Every attempt reads the audio through RecordingStore.withRequestAudio
 * (#342): the same request WAV whether the take is still a WAV or already
 * FLAC, and pinned while the attempt uses it. [onAttemptSettled] runs on
 * the worker after each attempt (the app schedules audio upkeep there).
 *
 * [injectedFailure] is a debug-build test hook (StarlingApplication): when
 * it returns a reason, every attempt fails with it after Stop, as an engine
 * failure would, so device tests can exercise retry and recovery.
 */
class TranscriptionCoordinator(
    private val store: RecordingStore,
    private val onDevice: OnDeviceBackend,
    private val injectedFailure: () -> String? = { null },
    private val onAttemptSettled: () -> Unit = {},
) {
    private val executor: ExecutorService = Executors.newCachedThreadPool { runnable ->
        Thread(runnable, "starling-transcription").apply { isDaemon = true }
    }
    private val mainHandler = Handler(Looper.getMainLooper())
    private val activeIds = ConcurrentHashMap.newKeySet<String>()

    /**
     * Returns whether the request was actually queued. A duplicate call for
     * an id that is already in flight is rejected without a callback, so the
     * caller must not show a pending message before checking this result.
     * [model] is an installed model to use instead of the active one (a
     * retry with another model); it changes no setting.
     */
    fun transcribe(
        id: String,
        model: String? = null,
        callback: (Recording) -> Unit = {},
    ): Boolean {
        // A double tap must not race two responses that overwrite the same
        // durable raw-transcript field.
        if (!activeIds.add(id)) return false
        val queued = runCatching { store.markTranscribing(id) }.getOrElse { exception ->
            activeIds.remove(id)
            callbackFailure(id, exception.message ?: "Unable to queue the recording", callback)
            return false
        }
        executor.execute {
            try {
                val completed = settleOrFail(id) { transcribeAudio(queued, model) }
                mainHandler.post { callback(completed) }
            } finally {
                activeIds.remove(id)
                runCatching(onAttemptSettled)
            }
        }
        return true
    }

    /**
     * Opens a live on-device session for a capture that is about to start,
     * or null when no model is installed; the caller then records exactly
     * as before, and Stop reports the missing model.
     *
     * [onEvent] is invoked on the main thread: [StreamEvent.Live] once audio
     * is accepted, growing [StreamEvent.Partial] transcripts while
     * recording, and [StreamEvent.Interrupted] when the session can no
     * longer be trusted — the recording itself is never affected. Callers
     * forward capture chunks to [StreamSession.onAudio] and must eventually
     * call [finishStreaming] (Stop) or [StreamSession.close]
     * (cancel/failure); nothing else is required of them. [savedAudio] is
     * the file the capture writes (RecordingStore.partialFile); the session
     * reads audio back from it that arrived while the model was still
     * loading.
     */
    fun beginStreaming(
        savedAudio: File? = null,
        onEvent: (StreamEvent) -> Unit = {},
    ): StreamSession? = onDevice.beginStreaming(savedAudio) { event -> mainHandler.post { onEvent(event) } }

    /**
     * Finalizes a live-streamed recording. Call only after the WAV has been
     * committed to the store: finishes the session and stores its final
     * transcript with LIVE_STREAM provenance. Whenever the session failed at
     * any point, the outcome is a batch transcription of the same saved WAV
     * through the ordinary [transcribe] code path, so the recording, retry,
     * and failure story is indistinguishable from the non-streaming flow.
     * The callback is delivered on the main thread, like [transcribe].
     */
    fun finishStreaming(
        session: StreamSession,
        id: String,
        callback: (Recording) -> Unit = {},
    ): Boolean {
        if (!activeIds.add(id)) return false
        val queued = runCatching { store.markTranscribing(id) }.getOrElse { exception ->
            activeIds.remove(id)
            callbackFailure(id, exception.message ?: "Unable to queue the recording", callback)
            return false
        }
        executor.execute {
            try {
                val completed = settleOrFail(id) { when (val outcome = session.finish()) {
                    is CommitOutcome.Final -> if (injectedFailure() != null) {
                        transcribeAudio(queued, null)
                    } else {
                        runCatching {
                            store.markTranscribed(
                                id,
                                outcome.text,
                                TranscriptionProvenance.LIVE_STREAM,
                                TranscriptSource.ON_DEVICE,
                                outcome.model ?: onDevice.activeModelName(),
                            )
                        }.getOrElse {
                            store.markFailed(id, "Transcript was received but could not be saved")
                        }
                    }
                    is CommitOutcome.Fallback ->
                        // The session is unusable; the durable WAV is the
                        // source of truth, so transcribe it like a retry.
                        transcribeAudio(queued, null)
                } }
                mainHandler.post { callback(completed) }
            } finally {
                activeIds.remove(id)
                runCatching(onAttemptSettled)
                // Safeguard: the session this worker was handed must never
                // outlive it, whatever finish() returned — close() is a
                // no-op when the session already cleaned up on its own.
                session.close()
            }
        }
        return true
    }

    /** Whether a transcription request for this id is currently in flight. */
    fun isActive(id: String): Boolean = id in activeIds

    fun shutdown() {
        executor.shutdownNow()
    }

    private fun transcribeAudio(queued: Recording, model: String?): Recording {
        val injected = injectedFailure()
        val result = if (injected != null) {
            InferenceResult.Failure(injected, false)
        } else {
            transcribeStoredAudio(store, queued.id) { audioFile -> onDevice.transcribe(audioFile, model) }
        }
        return when (result) {
            is InferenceResult.Success -> runCatching {
                store.markTranscribed(
                    queued.id,
                    result.rawTranscript,
                    TranscriptionProvenance.BATCH_UPLOAD,
                    TranscriptSource.ON_DEVICE,
                    result.model ?: model ?: onDevice.activeModelName(),
                )
            }.getOrElse {
                store.markFailed(queued.id, "Transcript was received but could not be saved")
            }
            is InferenceResult.Failure -> runCatching {
                store.markFailed(queued.id, result.message)
            }.getOrElse {
                queued.copy(errorMessage = result.message)
            }
        }
    }

    /**
     * Runs one transcription attempt so that the caller's callback always
     * fires: an unexpected exception (unreadable audio, an engine that throws
     * instead of returning a failure) settles the recording as failed and
     * retryable instead of leaving the caller waiting forever.
     */
    private inline fun settleOrFail(id: String, attempt: () -> Recording): Recording =
        try {
            attempt()
        } catch (e: Throwable) {
            // Throwable, not Exception: an OutOfMemoryError while decoding a
            // long recording must still settle the recording and the callback.
            val message = "Transcription failed unexpectedly: ${e.message ?: e::class.java.simpleName}"
            runCatching { store.markFailed(id, message) }
                .recoverCatching { store.get(id).copy(errorMessage = message) }
                .getOrElse {
                    // The store itself is failing; the callback still gets a failed recording.
                    Recording(
                        id = id,
                        createdAtMillis = 0L,
                        wavName = "",
                        status = RecordingStatus.FAILED,
                        errorMessage = message,
                    )
                }
        }

    private fun callbackFailure(id: String, message: String, callback: (Recording) -> Unit) {
        val failed = runCatching { store.markFailed(id, message) }.getOrNull()
        if (failed != null) {
            mainHandler.post { callback(failed) }
        }
    }
}

/**
 * One attempt of [transcribe] on the take's request audio
 * (RecordingStore.withRequestAudio). Missing, removed or damaged audio
 * fails the attempt as not retryable (the row says why); an I/O failure of
 * the engine is retryable like any other transient one.
 */
internal fun transcribeStoredAudio(
    store: RecordingStore,
    id: String,
    transcribe: (File) -> InferenceResult,
): InferenceResult = try {
    store.withRequestAudio(id) { audioFile ->
        try {
            transcribe(audioFile)
        } catch (exception: IOException) {
            InferenceResult.Failure(exception.message ?: "The transcription request failed", true)
        }
    }
} catch (exception: IOException) {
    InferenceResult.Failure(exception.message ?: "The recording audio could not be read", false)
}
