package dev.starling.mobile.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Window geometry and stitching of the on-device streamer. A 10 Hz "sample
 * rate" keeps the arithmetic readable: a 12 s window is 120 samples.
 */
class ChunkStreamerTest {
    private val rate = 10

    /** Records every call and answers with the words assigned to each second of audio. */
    private class ScriptedTranscriber(private val rate: Int, private val wordsPerSecond: (Int) -> String) :
        ChunkStreamer.Transcriber {
        val calls = mutableListOf<Pair<Int, Int>>()
        var failing = false

        override fun transcribe(samples: FloatArray, start: Int, length: Int): String? {
            calls += start to length
            if (failing) return null
            return (start / rate until (start + length) / rate).joinToString(" ") { wordsPerSecond(it) }
        }
    }

    private fun streamer() = ChunkStreamer(
        sampleRate = rate,
        chunkSeconds = 12.0,
        overlapSeconds = 3.0,
        minSeconds = 1.0,
        partialIntervalSeconds = 1.0,
    )

    @Test
    fun tailIsTranscribedForAPartialOnceItReachesTheMinimum() {
        val streamer = streamer()
        val tx = ScriptedTranscriber(rate) { "w$it" }
        val samples = FloatArray(300)

        assertNull(streamer.step(samples, 5, now = 0.0, tx = tx))
        assertEquals("w0 w1", streamer.step(samples, 20, now = 1.0, tx = tx))
        assertEquals(listOf(0 to 20), tx.calls)
    }

    @Test
    fun partialsAreThrottledByTheInterval() {
        val streamer = streamer()
        val tx = ScriptedTranscriber(rate) { "w$it" }
        val samples = FloatArray(300)

        assertEquals("w0 w1", streamer.step(samples, 20, now = 10.0, tx = tx))
        assertNull(streamer.step(samples, 30, now = 10.5, tx = tx))
        assertEquals("w0 w1 w2 w3", streamer.step(samples, 40, now = 11.0, tx = tx))
    }

    @Test
    fun fullWindowsAreFinalizedAndStitchedAcrossTheOverlap() {
        val streamer = streamer()
        val tx = ScriptedTranscriber(rate) { "w$it" }
        val samples = FloatArray(300)

        // 25 s: windows [0, 12) and [9, 21) finalize; the 4 s tail [18, 25) is the partial.
        val text = streamer.step(samples, 250, now = 0.0, tx = tx)

        assertEquals((0 until 25).joinToString(" ") { "w$it" }, text)
        assertEquals(listOf(0 to 120, 90 to 120, 180 to 70), tx.calls)
        assertEquals(180, streamer.boundary)
    }

    @Test
    fun flushFinalizesTheRemainderAndRebaseKeepsTheBoundaryAligned() {
        val streamer = streamer()
        val tx = ScriptedTranscriber(rate) { "w$it" }
        val samples = FloatArray(300)
        streamer.step(samples, 150, now = 0.0, tx = tx)
        assertEquals(90, streamer.boundary)

        // The owner dropped the 90 finalized samples from the buffer front.
        streamer.rebase(90)
        assertEquals(0, streamer.boundary)
        val tail = ScriptedTranscriber(rate) { "w${it + 9}" }

        assertEquals((0 until 16).joinToString(" ") { "w$it" }, streamer.flush(samples, 70, tail))
        assertEquals(listOf(0 to 70), tail.calls)
    }

    @Test
    fun emptyAudioFlushesToEmptyText() {
        assertEquals("", streamer().flush(FloatArray(0), 0, ScriptedTranscriber(rate) { "x" }))
    }

    @Test
    fun anEngineFailureDuringFlushIsReported() {
        val tx = ScriptedTranscriber(rate) { "w$it" }.apply { failing = true }

        assertNull(streamer().flush(FloatArray(50), 50, tx))
    }
}
