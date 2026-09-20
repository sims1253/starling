package dev.starling.mobile.network

import org.json.JSONObject

/**
 * One JSON text frame of the `WS /stream` protocol as documented in
 * docs/native-serving.md. Parsing is total and side-effect free: anything
 * malformed or unknown becomes null so a bad or newer frame can never kill a
 * recording; the session just ignores it.
 */
sealed interface StreamMessage {
    /** A growing partial transcript covering the session so far. */
    data class Partial(
        val text: String,
        val startSeconds: Double = 0.0,
        val endSeconds: Double = 0.0,
    ) : StreamMessage

    /** The transcript of all committed audio, produced on commit. */
    data class Final(
        val text: String,
        val durationSeconds: Double = 0.0,
    ) : StreamMessage

    /**
     * A server error frame. [bufferLimitReached] is true for the live-buffer
     * cap frame (`stream buffer limit reached (N s live buffer); audio
     * ignored until reset`): the server then ignores all further audio for
     * the session until a reset, which the client honors by giving up on the
     * stream — audio dropped between cap and reset could never be recovered
     * in the stream's own final, so the durable WAV batch upload takes over.
     */
    data class Error(
        val message: String,
        val bufferLimitReached: Boolean,
    ) : StreamMessage

    /** Reply to a client `{"type":"ping"}`. */
    data object Pong : StreamMessage

    /** Reply to a client `{"type":"reset"}` (not sent by this client). */
    data object ResetAck : StreamMessage

    companion object {
        const val TYPE_PARTIAL = "partial"
        const val TYPE_FINAL = "final"
        const val TYPE_ERROR = "error"
        const val TYPE_PONG = "pong"
        const val TYPE_RESET_ACK = "reset_ack"

        /**
         * Stable prefix of the live-buffer-cap error. The seconds value and
         * wording after it vary with the server's `--max-stream-seconds`.
         */
        const val BUFFER_LIMIT_PREFIX = "stream buffer limit reached"

        /** The bounded-retry busy error a commit can answer with. */
        const val SERVER_BUSY_MESSAGE = "server busy"

        fun parse(payload: String): StreamMessage? {
            val json = runCatching { JSONObject(payload) }.getOrNull() ?: return null
            val type = json.optString("type")
            return when (type) {
                TYPE_PARTIAL -> {
                    // An empty-but-present text is valid (silence so far).
                    if (json.isNull("text")) null else Partial(
                        text = json.optString("text"),
                        startSeconds = json.optDouble("start_s", 0.0),
                        endSeconds = json.optDouble("end_s", 0.0),
                    )
                }
                TYPE_FINAL -> {
                    // The exact server text is kept, empty or not: an empty
                    // transcript is a result to review, not evidence of silence.
                    if (json.isNull("text")) null else Final(
                        text = json.optString("text"),
                        durationSeconds = json.optDouble("duration_s", 0.0),
                    )
                }
                TYPE_ERROR -> {
                    val message = json.optString("message")
                    if (json.isNull("message") || message.isBlank()) null else Error(
                        message = message,
                        bufferLimitReached = message.startsWith(BUFFER_LIMIT_PREFIX),
                    )
                }
                TYPE_PONG -> Pong
                TYPE_RESET_ACK -> ResetAck
                else -> null
            }
        }
    }
}

/**
 * Events of a live streaming session, for the recording UI. Delivered on the
 * stream client's internal threads; callers that own a main thread must
 * marshal (the transcription coordinator does).
 */
sealed interface StreamEvent {
    /** The socket is open and audio is being accepted; partials can follow. */
    data object Live : StreamEvent

    /** A growing partial transcript of the session so far. */
    data class Partial(val text: String) : StreamEvent

    /**
     * The stream can no longer be trusted: connect failed, the connection
     * was lost mid-stream, the server hit its live-buffer cap, or keepalive
     * timed out. The recording is unaffected; when it stops, transcription
     * falls back to the batch upload of the saved WAV.
     */
    data class Interrupted(val reason: String, val bufferLimitReached: Boolean) : StreamEvent
}

/**
 * Result of finishing a live stream after the recording stopped. A session
 * that failed at any point — connect, mid-stream, or at commit — resolves to
 * [Fallback] so the caller retries with the durable WAV.
 */
sealed interface CommitOutcome {
    /** The server finalized the buffered audio; verbatim transcript. */
    data class Final(val text: String) : CommitOutcome

    /** The stream is unusable; transcribe the saved WAV through the batch path. */
    data class Fallback(val reason: String) : CommitOutcome
}
