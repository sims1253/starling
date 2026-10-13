package dev.starling.mobile.transcription

/** The outcome of one transcription of a saved recording. */
sealed interface InferenceResult {
    /** [model] names what produced the text (the on-device model file); null when the caller knows it. */
    data class Success(val rawTranscript: String, val model: String? = null) : InferenceResult
    data class Failure(val message: String, val retryable: Boolean) : InferenceResult
}

/** One live on-device transcription session handed to the recording UI. */
interface StreamSession {
    /**
     * Whether audio forwarded to [onAudio] is still transcribed live. False
     * once the session was interrupted, finished, or closed; callers use it
     * to stop forwarding chunks, never to stop recording.
     */
    fun acceptsAudio(): Boolean

    /**
     * Forwards one raw PCM16 16 kHz mono chunk (exactly what AudioCapture
     * writes). Safe to call from the capture worker thread; copies the bytes
     * before returning.
     */
    fun onAudio(bytes: ByteArray, count: Int)

    /**
     * Finalizes the buffered audio and blocks until its transcript or a
     * failure. Call on a worker thread, after the WAV has been made durable:
     * [CommitOutcome.Fallback] means the session failed at some point and
     * the saved WAV must be transcribed instead. Idempotent; later calls
     * return the first outcome.
     */
    fun finish(): CommitOutcome

    /** Drops the session without finalizing (recording cancelled or failed). */
    fun close()
}

/**
 * Events of a live session, for the recording UI. Delivered on the
 * session's worker thread; callers that own a main thread must marshal (the
 * transcription coordinator does).
 */
sealed interface StreamEvent {
    /** Audio is being transcribed; partials can follow. */
    data object Live : StreamEvent

    /** A growing partial transcript of the session so far. */
    data class Partial(val text: String) : StreamEvent

    /**
     * Live transcription can no longer be trusted (the model failed to load,
     * a window failed, or — [bufferLimitReached] — the engine fell behind
     * the recording). The recording is unaffected; when it stops, the saved
     * WAV is transcribed instead.
     */
    data class Interrupted(val reason: String, val bufferLimitReached: Boolean) : StreamEvent
}

/**
 * Result of finishing a live session after the recording stopped. A session
 * that failed at any point resolves to [Fallback] so the caller transcribes
 * the durable WAV.
 */
sealed interface CommitOutcome {
    /** The session finalized the buffered audio; verbatim transcript. [model] is the model file that produced it. */
    data class Final(val text: String, val model: String? = null) : CommitOutcome

    /** The session is unusable; transcribe the saved WAV instead. */
    data class Fallback(val reason: String) : CommitOutcome
}
