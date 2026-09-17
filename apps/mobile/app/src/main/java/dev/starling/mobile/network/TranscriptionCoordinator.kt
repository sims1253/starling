package dev.starling.mobile.network

import android.os.Handler
import android.os.Looper
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.TranscriptionProvenance
import dev.starling.mobile.engine.OnDeviceBackend
import dev.starling.mobile.storage.RecordingStore
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/** Uploads only after local audio finalization and persists every outcome. */
class TranscriptionCoordinator(
    private val store: RecordingStore,
    private val settings: BackendSettings,
    private val onDevice: OnDeviceBackend,
) {
    private val executor: ExecutorService = Executors.newCachedThreadPool { runnable ->
        Thread(runnable, "starling-transcription").apply { isDaemon = true }
    }
    private val mainHandler = Handler(Looper.getMainLooper())
    private val client = InferenceClient()
    private val streamClient = StreamClient()
    private val activeIds = ConcurrentHashMap.newKeySet<String>()

    /**
     * Returns whether the request was actually queued. A duplicate call for
     * an id that is already in flight is rejected without a callback, so the
     * caller must not show a pending message before checking this result.
     */
    fun transcribe(
        id: String,
        config: BackendConfig = settings.load(),
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
                val completed = transcribeAudio(queued, config)
                mainHandler.post { callback(completed) }
            } finally {
                activeIds.remove(id)
            }
        }
        return true
    }

    /**
     * Opens a live `WS /stream` session for a capture that is about to
     * start, when the configuration supports one: the Starling protocol on a
     * remote server whose endpoint passes the trusted-host policy. Returns
     * null otherwise (OpenAI-shaped endpoints, the on-device engine, or a
     * rejected endpoint) and the caller records exactly as before, without
     * streaming.
     *
     * [onEvent] is invoked on the main thread: [StreamEvent.Live] once audio
     * is accepted, growing [StreamEvent.Partial] transcripts while
     * recording, and [StreamEvent.Interrupted] when the stream can no longer
     * be trusted — the recording itself is never affected. Callers forward
     * capture chunks to [StreamSession.onAudio] and must eventually call
     * [finishStreaming] (Stop) or [StreamSession.close] (cancel/failure);
     * nothing else is required of them.
     */
    fun beginStreaming(
        config: BackendConfig = settings.load(),
        onEvent: (StreamEvent) -> Unit = {},
    ): StreamSession? {
        if (!streamingEligible(config)) return null
        val url = streamUrl(config.endpoint, config.allowTrustedLanHttp) ?: return null
        return streamClient.connect(url) { event ->
            mainHandler.post { onEvent(event) }
        }
    }

    /**
     * Finalizes a live-streamed recording. Call only after the WAV has been
     * committed to the store (durable audio before network use): commits the
     * stream and stores its final transcript with LIVE_STREAM provenance.
     * Whenever the stream failed at any point — connect, mid-stream, or at
     * commit — the outcome is a batch transcription of the same saved WAV
     * through the ordinary [transcribe] code path, so the recording, retry,
     * and failure story is indistinguishable from the non-streaming flow.
     * The callback is delivered on the main thread, like [transcribe].
     */
    fun finishStreaming(
        session: StreamSession,
        id: String,
        config: BackendConfig = settings.load(),
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
                val completed = when (val outcome = session.finish()) {
                    is CommitOutcome.Final -> runCatching {
                        store.markTranscribed(id, outcome.text, TranscriptionProvenance.LIVE_STREAM)
                    }.getOrElse {
                        store.markFailed(id, "Transcript was received but could not be saved")
                    }
                    is CommitOutcome.Fallback ->
                        // The stream is unusable; the durable WAV is the
                        // source of truth, so batch-upload it like a retry.
                        transcribeAudio(queued, config)
                }
                mainHandler.post { callback(completed) }
            } finally {
                activeIds.remove(id)
            }
        }
        return true
    }

    /** Whether a transcription request for this id is currently in flight. */
    fun isActive(id: String): Boolean = id in activeIds

    fun shutdown() {
        executor.shutdownNow()
    }

    private fun transcribeAudio(queued: Recording, config: BackendConfig): Recording {
        val audioFile = store.audioFile(queued)
        val result = if (config.engine == TranscriptionEngine.ON_DEVICE) {
            onDevice.transcribe(audioFile, config)
        } else {
            client.transcribe(audioFile, config)
        }
        return when (result) {
            is InferenceResult.Success -> runCatching {
                store.markTranscribed(queued.id, result.rawTranscript)
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

    private fun callbackFailure(id: String, message: String, callback: (Recording) -> Unit) {
        val failed = runCatching { store.markFailed(id, message) }.getOrNull()
        if (failed != null) {
            mainHandler.post { callback(failed) }
        }
    }

    companion object {
        /**
         * Only the Starling protocol's `WS /stream` is streamed. OpenAI-shaped
         * endpoints have no streaming route, and the on-device engine
         * transcribes from the file by design.
         */
        internal fun streamingEligible(config: BackendConfig): Boolean =
            config.engine == TranscriptionEngine.REMOTE && config.protocol == BackendProtocol.STARLING
    }
}
