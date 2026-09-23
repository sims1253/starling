package dev.starling.mobile.engine

import dev.starling.mobile.network.CommitOutcome
import dev.starling.mobile.network.StreamEvent
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicInteger

/**
 * The on-device live session against a fake engine: partials while audio
 * arrives, the final from the flushed tail, and the fallback contract shared
 * with the WS client — any failure resolves finish() to Fallback so the saved
 * WAV goes through the batch path.
 */
class OnDeviceStreamSessionTest {
    private class FakeEngine(
        private val loadError: String? = null,
        private val failAfterCalls: Int = Int.MAX_VALUE,
    ) : OnDeviceStreamSession.LiveEngine {
        val calls = AtomicInteger()

        override fun prepare(): String? = loadError

        override fun transcribeWindow(samples: FloatArray): OnDeviceStreamSession.WindowResult {
            if (calls.incrementAndGet() > failAfterCalls) {
                return OnDeviceStreamSession.WindowResult.Failed("engine exploded")
            }
            // One word per started second of audio, so the text grows with the take.
            val seconds = (samples.size + ChunkStreamer.SAMPLE_RATE - 1) / ChunkStreamer.SAMPLE_RATE
            return OnDeviceStreamSession.WindowResult.Text((0 until seconds).joinToString(" ") { "s$it" })
        }
    }

    private val events = CopyOnWriteArrayList<StreamEvent>()

    private fun session(engine: OnDeviceStreamSession.LiveEngine, maxLiveSamples: Int = Int.MAX_VALUE) =
        OnDeviceStreamSession(
            engine = engine,
            events = { events += it },
            streamer = ChunkStreamer(minSeconds = 1.0, partialIntervalSeconds = 0.0),
            maxLiveSamples = maxLiveSamples,
        ).start()

    /** [seconds] of PCM16 silence as AudioCapture delivers it. */
    private fun pcm(seconds: Double) = ByteArray((seconds * ChunkStreamer.SAMPLE_RATE).toInt() * 2)

    private fun awaitEvent(predicate: (StreamEvent) -> Boolean): StreamEvent {
        val deadline = System.nanoTime() + 5_000_000_000L
        while (System.nanoTime() < deadline) {
            events.firstOrNull(predicate)?.let { return it }
            Thread.sleep(10)
        }
        throw AssertionError("event not observed; saw $events")
    }

    @Test
    fun partialsGrowWhileRecordingAndFinishReturnsTheFlushedFinal() {
        val session = session(FakeEngine())
        awaitEvent { it == StreamEvent.Live }

        val chunk = pcm(1.0)
        session.onAudio(chunk, chunk.size)
        awaitEvent { it == StreamEvent.Partial("s0") }
        session.onAudio(chunk, chunk.size)
        awaitEvent { it == StreamEvent.Partial("s0 s1") }

        assertEquals(CommitOutcome.Final("s0 s1"), session.finish())
        assertFalse(session.acceptsAudio())
    }

    @Test
    fun aRecordingLongerThanOneWindowIsStitchedIntoOneFinal() {
        val session = session(FakeEngine())
        // 20 s arrives in 100 ms capture chunks.
        val chunk = pcm(0.1)
        repeat(200) { session.onAudio(chunk, chunk.size) }

        val final = session.finish() as CommitOutcome.Final
        // Windows re-count their seconds from zero, so the stitched text is
        // not a clean s0..s19 — but it is non-empty, and finish never falls back.
        assertTrue(final.text.isNotBlank())
    }

    @Test
    fun aModelThatCannotLoadInterruptsTheStreamAndFallsBack() {
        val session = session(FakeEngine(loadError = "no model"))

        val interrupted = awaitEvent { it is StreamEvent.Interrupted } as StreamEvent.Interrupted
        assertEquals("no model", interrupted.reason)
        assertFalse(session.acceptsAudio())
        assertEquals(CommitOutcome.Fallback("no model"), session.finish())
    }

    @Test
    fun anEngineFailureMidStreamFallsBackToTheBatchPath() {
        val session = session(FakeEngine(failAfterCalls = 1))
        val chunk = pcm(1.0)
        session.onAudio(chunk, chunk.size)
        awaitEvent { it is StreamEvent.Partial }
        session.onAudio(chunk, chunk.size)

        awaitEvent { it is StreamEvent.Interrupted }
        assertEquals(CommitOutcome.Fallback("engine exploded"), session.finish())
    }

    @Test
    fun anEngineThatFallsBehindTripsTheLiveBufferCap() {
        val session = session(FakeEngine(), maxLiveSamples = ChunkStreamer.SAMPLE_RATE)
        val chunk = pcm(2.0)
        session.onAudio(chunk, chunk.size)

        val interrupted = awaitEvent { it is StreamEvent.Interrupted } as StreamEvent.Interrupted
        assertTrue(interrupted.bufferLimitReached)
        assertTrue(session.finish() is CommitOutcome.Fallback)
    }

    @Test
    fun aCrashOutsideTheEngineStillSettlesFinish() {
        val session = OnDeviceStreamSession(
            engine = FakeEngine(),
            events = { events += it },
            streamer = ChunkStreamer(minSeconds = 1.0, partialIntervalSeconds = 0.0),
            clock = { throw IllegalStateException("clock broke") },
        ).start()
        val chunk = pcm(1.0)
        session.onAudio(chunk, chunk.size)

        val interrupted = awaitEvent { it is StreamEvent.Interrupted } as StreamEvent.Interrupted
        assertEquals("clock broke", interrupted.reason)
        assertEquals(CommitOutcome.Fallback("clock broke"), session.finish())
    }

    @Test
    fun aBufferCapTrippedOnTheCaptureThreadIsReportedFromTheWorker() {
        val threads = CopyOnWriteArrayList<String>()
        val session = OnDeviceStreamSession(
            engine = FakeEngine(),
            events = { events += it; threads += Thread.currentThread().name },
            maxLiveSamples = ChunkStreamer.SAMPLE_RATE,
        ).start()
        val chunk = pcm(2.0)
        session.onAudio(chunk, chunk.size)

        awaitEvent { it is StreamEvent.Interrupted }
        assertTrue(threads.all { it == "starling-on-device-stream" })
        assertEquals(1, events.count { it is StreamEvent.Interrupted })
    }

    @Test
    fun closeSettlesAPendingFinishWithoutAFinal() {
        val session = session(FakeEngine())
        session.close()

        assertTrue(session.finish() is CommitOutcome.Fallback)
        assertFalse(session.acceptsAudio())
    }
}
