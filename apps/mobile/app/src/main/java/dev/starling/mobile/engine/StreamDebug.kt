package dev.starling.mobile.engine

/**
 * Debug-build hooks for on-device streaming measurements (#226/#357),
 * installed by StarlingApplication or a device test; release builds never
 * set them, so production sessions use the default cadence and no trace.
 */
object StreamDebug {
    /**
     * Preview cadence override, consulted when a live session starts:
     * (first-partial minimum, partial interval) in seconds, or null for
     * [ChunkStreamer]'s defaults.
     */
    @Volatile
    var cadence: (() -> Pair<Double, Double>?)? = null

    /**
     * When set, every live session records a [StreamTrace] and hands it here
     * once its worker has ended (on that worker thread).
     */
    @Volatile
    var traceSink: ((StreamTrace) -> Unit)? = null

    /** The streamer a new session uses: the defaults unless [cadence] overrides them. */
    fun streamer(): ChunkStreamer =
        cadence?.invoke()?.let { (min, interval) -> ChunkStreamer(minSeconds = min, partialIntervalSeconds = interval) }
            ?: ChunkStreamer()

    /** A trace for a new session using [streamer], or null when tracing is off. */
    fun trace(streamer: ChunkStreamer): StreamTrace? =
        traceSink?.let { sink ->
            StreamTrace(onComplete = sink).apply {
                mark("min_partial_seconds", streamer.minSeconds)
                mark("partial_interval_seconds", streamer.partialIntervalSeconds)
            }
        }
}
