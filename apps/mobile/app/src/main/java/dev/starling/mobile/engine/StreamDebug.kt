package dev.starling.mobile.engine

import android.util.Log

/**
 * Debug-build hooks for on-device streaming measurements (#226/#357),
 * installed by StarlingApplication or a device test; release builds never
 * set them, so production sessions use the default cadence and no trace.
 */
object StreamDebug {
    private const val TAG = "StreamDebug"

    /**
     * Preview cadence override, consulted when a live session starts:
     * (first-partial minimum, partial interval) in seconds, or null for
     * [ChunkStreamer]'s defaults. Values outside [validCadence] fall back to
     * the defaults with a log line; a debug override never breaks capture.
     */
    @Volatile
    var cadence: (() -> Pair<Double, Double>?)? = null

    /**
     * Consulted when a live session starts: where its [StreamTrace] goes
     * once the session's worker has ended (called on that worker thread), or
     * null to record no trace for that session.
     */
    @Volatile
    var traceSink: (() -> ((StreamTrace) -> Unit)?)? = null

    /** The streamer a new session uses: the defaults unless [cadence] overrides them. */
    fun streamer(): ChunkStreamer {
        val (min, interval) = cadence?.invoke() ?: return ChunkStreamer()
        if (validCadence(min, interval)) {
            runCatching { ChunkStreamer(minSeconds = min, partialIntervalSeconds = interval) }
                .onSuccess { return it }
        }
        runCatching { Log.w(TAG, "debug: ignoring stream cadence $min s / $interval s; using the defaults") }
        return ChunkStreamer()
    }

    /**
     * Whether ([min], [interval]) is a usable cadence: both finite and within
     * one window (a longer interval would never preview before the window
     * finalizes on its own).
     */
    fun validCadence(min: Double, interval: Double): Boolean =
        min.isFinite() && interval.isFinite() &&
            min in 0.0..ChunkStreamer.CHUNK_SECONDS && interval in 0.0..ChunkStreamer.CHUNK_SECONDS

    /** A trace for a new session using [streamer], or null when tracing is off. */
    fun trace(streamer: ChunkStreamer): StreamTrace? =
        traceSink?.invoke()?.let { sink ->
            StreamTrace(onComplete = sink).apply {
                mark("min_partial_seconds", streamer.minSeconds)
                mark("partial_interval_seconds", streamer.partialIntervalSeconds)
            }
        }
}
