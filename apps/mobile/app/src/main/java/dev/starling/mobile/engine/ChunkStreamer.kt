package dev.starling.mobile.engine

/**
 * Rolling fixed-window state for on-device live transcription. Port of the
 * native server's ChunkStreamer (cpp/serve/stream_session.cpp, itself a
 * port of src/starling/stream_chunk.py), so on-device partials follow the
 * same window geometry and word stitching as `WS /stream`:
 *
 * - every full [chunkSeconds] window past the finalized boundary is
 *   transcribed once and stitched into the committed words, and the
 *   boundary advances by the window minus [overlapSeconds];
 * - between windows, the live tail (at least [minSeconds] long, at most one
 *   window) is transcribed at most every [partialIntervalSeconds] for a
 *   responsive partial that is never committed;
 * - [flush] finalizes whatever remains on stop.
 *
 * One divergence from the server: a null from the [Transcriber] means the
 * on-device engine failed, not "busy, retry later". The session abandons the
 * stream on the first failure (the saved WAV then goes through the batch
 * path), so [flush] makes a single attempt instead of bounded busy retries.
 *
 * Not thread-safe; the owning session drives it from one worker thread.
 */
class ChunkStreamer(
    sampleRate: Int = SAMPLE_RATE,
    chunkSeconds: Double = CHUNK_SECONDS,
    overlapSeconds: Double = OVERLAP_SECONDS,
    minSeconds: Double = MIN_SECONDS,
    private val partialIntervalSeconds: Double = PARTIAL_INTERVAL_SECONDS,
) {
    /** Transcribes `samples[start until start + length]`; null when the engine failed. */
    fun interface Transcriber {
        fun transcribe(samples: FloatArray, start: Int, length: Int): String?
    }

    private val chunk: Int
    private val advance: Int
    private val min: Int
    private val maxOverlapWords: Int
    private var committed: List<String> = emptyList()
    private var lastEmit = Double.NEGATIVE_INFINITY

    /** Sample index up to which audio is finalized; audio before it can be dropped. */
    var boundary: Int = 0
        private set

    init {
        require(sampleRate > 0) { "sampleRate must be positive" }
        require(chunkSeconds > 0.0 && chunkSeconds.isFinite()) { "chunkSeconds must be positive" }
        require(overlapSeconds >= 0.0 && overlapSeconds < chunkSeconds) { "overlap must be in [0, chunk)" }
        require(minSeconds >= 0.0 && minSeconds <= chunkSeconds) { "minSeconds must be in [0, chunk]" }
        require(partialIntervalSeconds >= 0.0) { "partialIntervalSeconds must be nonnegative" }
        chunk = maxOf(1, (chunkSeconds * sampleRate).toInt())
        // The server caps the overlap at half a window; kept for identical geometry.
        val overlap = (minOf(overlapSeconds, chunkSeconds * 0.5) * sampleRate).toInt()
        advance = maxOf(1, chunk - overlap)
        min = (minSeconds * sampleRate).toInt()
        maxOverlapWords = maxOf(8, (overlapSeconds * 6).toInt() + 6)
    }

    /**
     * Advances the stream over `samples[0 until size]` at time [now]
     * (seconds, any monotonic origin). Finalizes full windows, then
     * (throttled) transcribes the live tail. Returns the full text to show
     * as a partial, or null when there is nothing new to show.
     */
    fun step(samples: FloatArray, size: Int, now: Double, tx: Transcriber): String? {
        val finalized = finalizeFullWindows(samples, size, tx)
        val committedText = { if (finalized) join(committed) else null }

        val tailLength = size - boundary
        // A full window is still waiting: it failed, and the session stops.
        if (tailLength >= chunk) return committedText()
        val throttled = (now - lastEmit) < partialIntervalSeconds
        if (!finalized && (throttled || tailLength < min)) return null

        if (tailLength > 0 && tailLength >= min) {
            // Only a tail transcription starts the throttle interval; a step
            // that merely finalized a window must not delay the next partial.
            lastEmit = now
            val text = tx.transcribe(samples, boundary, tailLength) ?: return committedText()
            return join(ChunkedTranscription.stitchWords(committed, split(text), maxOverlapWords))
        }
        return committedText()
    }

    /** Finalizes all remaining audio; the full text, or null when the engine failed. */
    fun flush(samples: FloatArray, size: Int, tx: Transcriber): String? {
        finalizeFullWindows(samples, size, tx)
        val tailLength = size - boundary
        if (tailLength == 0) return join(committed)
        if (tailLength < 0 || tailLength >= chunk) return null
        val text = tx.transcribe(samples, boundary, tailLength) ?: return null
        committed = ChunkedTranscription.stitchWords(committed, split(text), maxOverlapWords)
        boundary = size
        return join(committed)
    }

    /** Shifts the boundary after [dropped] finalized samples left the front of the buffer. */
    fun rebase(dropped: Int) {
        boundary = maxOf(0, boundary - dropped)
    }

    private fun finalizeFullWindows(samples: FloatArray, size: Int, tx: Transcriber): Boolean {
        var did = false
        while (size - boundary >= chunk) {
            val text = tx.transcribe(samples, boundary, chunk) ?: break
            committed = ChunkedTranscription.stitchWords(committed, split(text), maxOverlapWords)
            boundary += advance
            did = true
        }
        return did
    }

    private fun split(text: String): List<String> = text.trim().split(WHITESPACE).filter { it.isNotEmpty() }

    private fun join(words: List<String>): String = words.joinToString(" ")

    companion object {
        const val SAMPLE_RATE = 16_000

        // The server's WS /stream window geometry (docs/native-serving.md).
        const val CHUNK_SECONDS = 12.0
        const val OVERLAP_SECONDS = 3.0

        // Shorter than the server's 5 s / 3 s: there is no network round
        // trip to amortize, and the first words should appear quickly.
        const val MIN_SECONDS = 1.0
        const val PARTIAL_INTERVAL_SECONDS = 1.0

        private val WHITESPACE = Regex("\\s+")
    }
}
