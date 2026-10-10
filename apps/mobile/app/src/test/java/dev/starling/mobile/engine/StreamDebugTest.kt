package dev.starling.mobile.engine

import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

/** The debug cadence override and per-session trace switch (#226/#357). */
class StreamDebugTest {
    @After
    fun reset() {
        StreamDebug.cadence = null
        StreamDebug.traceSink = null
    }

    private fun cadenceOf(streamer: ChunkStreamer) = streamer.minSeconds to streamer.partialIntervalSeconds

    private val defaults = ChunkStreamer.MIN_SECONDS to ChunkStreamer.PARTIAL_INTERVAL_SECONDS

    @Test
    fun aValidOverrideIsUsed() {
        StreamDebug.cadence = { 0.5 to 2.0 }
        assertEquals(0.5 to 2.0, cadenceOf(StreamDebug.streamer()))
    }

    @Test
    fun noOverrideUsesTheDefaults() {
        assertEquals(defaults, cadenceOf(StreamDebug.streamer()))
    }

    @Test
    fun anInvalidOverrideFallsBackToTheDefaults() {
        val invalid = listOf(
            20.0 to 1.0, // minimum past the 12 s window
            -1.0 to 1.0,
            1.0 to -0.5,
            Double.NaN to 1.0,
            1.0 to Double.NaN,
            1.0 to Double.POSITIVE_INFINITY,
            1.0 to 1e9,
        )
        for (value in invalid) {
            StreamDebug.cadence = { value }
            assertEquals("$value", defaults, cadenceOf(StreamDebug.streamer()))
        }
    }

    @Test
    fun theTraceSwitchIsConsultedPerSession() {
        var tracing = true
        StreamDebug.traceSink = { if (tracing) { _: StreamTrace -> } else null }
        val streamer = StreamDebug.streamer()
        assertNotNull(StreamDebug.trace(streamer))
        tracing = false
        assertNull(StreamDebug.trace(streamer))
    }
}
