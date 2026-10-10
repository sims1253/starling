package dev.starling.mobile.engine

import dev.starling.mobile.network.CommitOutcome
import dev.starling.mobile.network.StreamEvent
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.roundToInt

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

        override fun prepare(cancelled: () -> Boolean): String? = loadError

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
    fun theFinalNamesEveryModelThatTranscribedItsWindows() {
        val calls = AtomicInteger()
        // A driver failure voided the pin after the first window and the
        // reload picked up another active model.
        val engine = object : OnDeviceStreamSession.LiveEngine {
            override fun prepare(cancelled: () -> Boolean): String? = null
            override fun loadedModelName(): String = "a.gguf"
            override fun transcribeWindow(samples: FloatArray) = OnDeviceStreamSession.WindowResult.Text(
                "w",
                if (calls.incrementAndGet() == 1) "a.gguf" else "b.gguf",
            )
        }
        val session = session(engine)
        awaitEvent { it == StreamEvent.Live }
        val chunk = pcm(1.0)
        session.onAudio(chunk, chunk.size)
        awaitEvent { it is StreamEvent.Partial }
        session.onAudio(chunk, chunk.size)

        assertEquals("a.gguf, b.gguf", (session.finish() as CommitOutcome.Final).model)
    }

    @Test
    fun withoutPerWindowModelsTheFinalNamesThePreparedModel() {
        val engine = object : OnDeviceStreamSession.LiveEngine {
            override fun prepare(cancelled: () -> Boolean): String? = null
            override fun loadedModelName(): String = "a.gguf"
            override fun transcribeWindow(samples: FloatArray) = OnDeviceStreamSession.WindowResult.Text("w")
        }
        val session = session(engine)
        val chunk = pcm(1.0)
        session.onAudio(chunk, chunk.size)

        assertEquals("a.gguf", (session.finish() as CommitOutcome.Final).model)
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

    /**
     * An engine whose words name the absolute second of the audio they came
     * from (each second of test audio carries its index as the sample value),
     * so the final text proves order, completeness and no duplication.
     * [loaded] gates prepare(), like a cold model load.
     */
    private class SecondsEngine(private val loaded: CountDownLatch = CountDownLatch(0)) :
        OnDeviceStreamSession.LiveEngine {
        override fun prepare(cancelled: () -> Boolean): String? {
            loaded.await(10, TimeUnit.SECONDS)
            return null
        }

        override fun transcribeWindow(samples: FloatArray): OnDeviceStreamSession.WindowResult {
            val words = (samples.indices step ChunkStreamer.SAMPLE_RATE).map { i ->
                "w" + (samples[i] * 32768f / SECOND_STEP).roundToInt()
            }
            return OnDeviceStreamSession.WindowResult.Text(words.joinToString(" "))
        }
    }

    /** The capture's saved WAV payload, as SavedAudioBacklog would read it. */
    private class SavedAudio : OnDeviceStreamSession.Backlog {
        val bytes = ByteArrayOutputStream()
        val destinations = mutableListOf<FloatArray>()
        var opened = 0
        var failReads = false

        override fun read(from: Long, into: FloatArray, offset: Int, count: Int): Int {
            if (failReads) throw java.io.IOException("gone")
            destinations += into
            val data = synchronized(bytes) { bytes.toByteArray() }
            var n = 0
            while (n < count && 2 * (from + n) + 1 < data.size) {
                val at = (2 * (from + n)).toInt()
                into[offset + n] = ((data[at + 1].toInt() shl 8) or (data[at].toInt() and 0xff)).toShort() / 32768f
                n++
            }
            return n
        }

        override fun close() = Unit
    }

    /** One second of PCM16 whose every sample is [second]'s marker value. */
    private fun second(second: Int): ByteArray {
        val value = second * SECOND_STEP
        return ByteArray(ChunkStreamer.SAMPLE_RATE * 2) { i -> if (i % 2 == 0) value.toByte() else (value shr 8).toByte() }
    }

    /** Writes like AudioCapture: the WAV first, then the live session. */
    private fun capture(session: OnDeviceStreamSession, saved: SavedAudio?, seconds: IntRange) {
        for (s in seconds) {
            val chunk = second(s)
            saved?.let { synchronized(it.bytes) { it.bytes.write(chunk) } }
            session.onAudio(chunk, chunk.size)
        }
    }

    private fun expected(seconds: Int) = (0 until seconds).joinToString(" ") { "w$it" }

    @Test
    fun audioCapturedWhileTheModelLoadsIsTranscribedInOrder() {
        val loaded = CountDownLatch(1)
        val session = session(SecondsEngine(loaded))
        // The first seconds arrive before the model is ready.
        capture(session, null, 0 until 5)
        loaded.countDown()
        awaitEvent { it == StreamEvent.Live }
        capture(session, null, 5 until 8)

        assertEquals(CommitOutcome.Final(expected(8)), session.finish())
        assertFalse(events.any { it is StreamEvent.Interrupted })
    }

    @Test
    fun stopBeforeTheModelFinishedLoadingStillYieldsTheCompleteFinal() {
        val loaded = CountDownLatch(1)
        val session = session(SecondsEngine(loaded))
        capture(session, null, 0 until 3)
        Thread { Thread.sleep(200); loaded.countDown() }.start()

        assertEquals(CommitOutcome.Final(expected(3)), session.finish())
    }

    @Test
    fun aLoadLongerThanTheLiveBufferReadsTheBacklogBackFromTheSavedAudio() {
        val loaded = CountDownLatch(1)
        val saved = SavedAudio()
        val session = OnDeviceStreamSession(
            engine = SecondsEngine(loaded),
            events = { events += it },
            streamer = ChunkStreamer(minSeconds = 1.0, partialIntervalSeconds = 0.0),
            // Two seconds of memory; the 30 s captured during the load spill.
            maxLiveSamples = 2 * ChunkStreamer.SAMPLE_RATE,
            backlog = { saved.also { it.opened++ } },
        ).start()
        capture(session, saved, 0 until 30)
        loaded.countDown()
        awaitEvent { it == StreamEvent.Live }
        // Recording goes on while the worker catches up.
        capture(session, saved, 30 until 40)

        assertEquals(CommitOutcome.Final(expected(40)), session.finish())
        assertEquals(1, saved.opened)
        assertFalse(events.any { it is StreamEvent.Interrupted })
    }

    @Test
    fun stopDuringASpilledLoadCatchesUpBeforeFinalizing() {
        val loaded = CountDownLatch(1)
        val saved = SavedAudio()
        val session = OnDeviceStreamSession(
            engine = SecondsEngine(loaded),
            events = { events += it },
            streamer = ChunkStreamer(minSeconds = 1.0, partialIntervalSeconds = 0.0),
            maxLiveSamples = ChunkStreamer.SAMPLE_RATE,
            backlog = { saved },
        ).start()
        capture(session, saved, 0 until 25)
        Thread { Thread.sleep(200); loaded.countDown() }.start()

        assertEquals(CommitOutcome.Final(expected(25)), session.finish())
    }

    @Test
    fun catchingUpReusesOneScratchArrayAcrossRefills() {
        val loaded = CountDownLatch(1)
        val saved = SavedAudio()
        val session = OnDeviceStreamSession(
            engine = SecondsEngine(loaded),
            events = { events += it },
            // 1 s windows with a 5 s buffer: catching up on the spilled audio
            // takes several refill passes.
            streamer = ChunkStreamer(chunkSeconds = 1.0, overlapSeconds = 0.0, minSeconds = 1.0, partialIntervalSeconds = 0.0),
            maxLiveSamples = 5 * ChunkStreamer.SAMPLE_RATE,
            backlog = { saved },
        ).start()
        capture(session, saved, 0 until 25)
        loaded.countDown()

        assertEquals(CommitOutcome.Final(expected(25)), session.finish())
        // Several reads, one reused (grown on demand) destination array
        // instead of a fresh maxLiveSamples allocation per refill.
        assertTrue(saved.destinations.size > 1)
        assertTrue(saved.destinations.all { it === saved.destinations.first() })
    }

    @Test
    fun anUnreadableBacklogFallsBackToTheBatchPath() {
        val loaded = CountDownLatch(1)
        val saved = SavedAudio().apply { failReads = true }
        val session = OnDeviceStreamSession(
            engine = SecondsEngine(loaded),
            events = { events += it },
            streamer = ChunkStreamer(minSeconds = 1.0, partialIntervalSeconds = 0.0),
            maxLiveSamples = ChunkStreamer.SAMPLE_RATE,
            backlog = { saved },
        ).start()
        capture(session, saved, 0 until 3)
        loaded.countDown()

        assertEquals(CommitOutcome.Fallback("the saved recording could not be read back"), session.finish())
    }

    @Test
    fun aBacklogThatCannotBeOpenedFailsWithTheRealCauseNotTheBufferCap() {
        val session = OnDeviceStreamSession(
            engine = FakeEngine(),
            events = { events += it },
            maxLiveSamples = ChunkStreamer.SAMPLE_RATE,
            backlog = { throw java.io.FileNotFoundException("no partial WAV") },
        ).start()
        val chunk = pcm(2.0)
        session.onAudio(chunk, chunk.size)

        val interrupted = awaitEvent { it is StreamEvent.Interrupted } as StreamEvent.Interrupted
        // An I/O failure is not the engine falling behind: the reason keeps
        // the exception's message, and the buffer-cap flag stays reserved for
        // the genuine no-backlog case.
        assertFalse(interrupted.bufferLimitReached)
        assertEquals("the saved recording could not be opened: no partial WAV", interrupted.reason)
        assertTrue(session.finish() is CommitOutcome.Fallback)
    }

    /**
     * An engine whose windows answer "n<samples>" (so the final names the
     * audio it covered) and whose previews run [preview] with the session's
     * cancel predicate, as a native call polling its checkpoints would.
     */
    /**
     * Previews come from [preview]; with [tracksLoads] the engine reports
     * [generation] as its load and stamps previews that do not name one.
     */
    private class PreviewEngine(
        private val tracksLoads: Boolean = true,
        private val preview: (cancel: () -> Boolean) -> OnDeviceStreamSession.WindowResult,
    ) : OnDeviceStreamSession.LiveEngine {
        val windows = CopyOnWriteArrayList<Int>()
        val previews = CopyOnWriteArrayList<Int>()
        val previewStarted = java.util.concurrent.Semaphore(0)

        @Volatile
        var generation = 1L

        override fun prepare(cancelled: () -> Boolean): String? = null

        override fun loadGeneration(): Long =
            if (tracksLoads) generation else OnDeviceStreamSession.LiveEngine.UNKNOWN_GENERATION

        override fun transcribeWindow(samples: FloatArray): OnDeviceStreamSession.WindowResult {
            windows += samples.size
            return OnDeviceStreamSession.WindowResult.Text("n${samples.size}")
        }

        override fun transcribePreview(samples: FloatArray, cancel: () -> Boolean): OnDeviceStreamSession.WindowResult {
            previews += samples.size
            previewStarted.release()
            val result = preview(cancel)
            val unstamped = result is OnDeviceStreamSession.WindowResult.Text &&
                result.generation == OnDeviceStreamSession.LiveEngine.UNKNOWN_GENERATION
            return if (tracksLoads && unstamped) {
                (result as OnDeviceStreamSession.WindowResult.Text).copy(generation = generation)
            } else {
                result
            }
        }
    }

    /** Polls [cancel] like an engine checkpoint loop; Cancelled once it holds. */
    private fun untilCancelled(cancel: () -> Boolean): OnDeviceStreamSession.WindowResult {
        val deadline = System.nanoTime() + 5_000_000_000L
        while (System.nanoTime() < deadline) {
            if (cancel()) return OnDeviceStreamSession.WindowResult.Cancelled
            Thread.sleep(2)
        }
        throw AssertionError("the preview was never cancelled")
    }

    private fun tracedSession(engine: OnDeviceStreamSession.LiveEngine, trace: StreamTrace) =
        OnDeviceStreamSession(
            engine = engine,
            events = { events += it },
            streamer = ChunkStreamer(minSeconds = 1.0, partialIntervalSeconds = 0.0),
            trace = trace,
        ).start()

    private fun stopTrace(trace: StreamTrace) =
        trace.toJson().getJSONArray("events").let { it.getJSONArray(it.length() - 1) }
            .getJSONObject(1).getJSONObject("trace").getJSONObject("stop")

    @Test
    fun stopDuringAPreviewCancelsItAndTheFlushCoversTheNewerAudio() {
        val engine = PreviewEngine(preview = ::untilCancelled)
        val trace = StreamTrace()
        val session = tracedSession(engine, trace)
        val second = pcm(1.0)
        session.onAudio(second, second.size)
        assertTrue(engine.previewStarted.tryAcquire(5, TimeUnit.SECONDS))
        val half = pcm(0.5)
        session.onAudio(half, half.size)

        // Stop with audio the running preview does not cover: the preview
        // gives way and the flush transcribes the whole 1.5 s tail.
        assertEquals(CommitOutcome.Final("n24000"), session.finish())
        assertEquals(listOf(16_000), engine.previews)
        assertEquals(listOf(24_000), engine.windows)
        assertFalse(events.any { it is StreamEvent.Partial })
        assertEquals(
            listOf(StreamTrace.RESULT_PREEMPTED, StreamTrace.RESULT_OK),
            trace.calls().map { it.result },
        )
        assertEquals(StreamTrace.PATH_TAIL, stopTrace(trace).getString("path"))
    }

    @Test
    fun stopWithNoNewerAudioLetsThePreviewFinishAndReusesItAsTheFinal() {
        val release = CountDownLatch(1)
        var cancelSeen = false
        val engine = PreviewEngine { cancel ->
            // Stop arrives while this preview runs; it must not be cancelled.
            while (!release.await(2, TimeUnit.MILLISECONDS)) cancelSeen = cancelSeen || cancel()
            OnDeviceStreamSession.WindowResult.Text("p")
        }
        val trace = StreamTrace()
        val session = tracedSession(engine, trace)
        val second = pcm(1.0)
        session.onAudio(second, second.size)
        assertTrue(engine.previewStarted.tryAcquire(5, TimeUnit.SECONDS))

        val outcome = java.util.concurrent.atomic.AtomicReference<CommitOutcome>()
        val stopper = Thread { outcome.set(session.finish()) }.apply { start() }
        while (trace.toJson().getJSONArray("commits").length() == 0) Thread.sleep(2)
        Thread.sleep(50)
        release.countDown()
        stopper.join(5_000)

        assertFalse(cancelSeen)
        assertEquals(CommitOutcome.Final("p"), outcome.get())
        // The unchanged tail was never transcribed a second time.
        assertEquals(emptyList<Int>(), engine.windows)
        assertEquals(StreamTrace.PATH_REUSED, stopTrace(trace).getString("path"))
        assertEquals(StreamTrace.RESULT_REUSED, trace.calls().last().result)
    }

    @Test
    fun aCompletedWindowLetsTheRunningPreviewFinishAndShowsIt() {
        val release = CountDownLatch(1)
        var cancelSeen = false
        val first = java.util.concurrent.atomic.AtomicBoolean(true)
        val engine = PreviewEngine { cancel ->
            if (first.getAndSet(false)) {
                while (!release.await(2, TimeUnit.MILLISECONDS)) cancelSeen = cancelSeen || cancel()
                OnDeviceStreamSession.WindowResult.Text("p")
            } else {
                OnDeviceStreamSession.WindowResult.Text("tail")
            }
        }
        val session = tracedSession(engine, StreamTrace())
        val second = pcm(1.0)
        session.onAudio(second, second.size)
        assertTrue(engine.previewStarted.tryAcquire(5, TimeUnit.SECONDS))

        // The rest of the first 12 s window arrives while the preview runs:
        // the preview still completes and is shown, then the window commits.
        val rest = pcm(11.0)
        session.onAudio(rest, rest.size)
        Thread.sleep(50)
        release.countDown()

        awaitEvent { it == StreamEvent.Partial("p") }
        awaitEvent { it is StreamEvent.Partial && it.text.startsWith("n192000") }
        assertFalse(cancelSeen)
        assertEquals(listOf(12 * ChunkStreamer.SAMPLE_RATE), engine.windows.take(1))
        assertTrue(session.finish() is CommitOutcome.Final)
    }

    @Test
    fun aPreviewThatCompletesAfterItBecameObsoleteIsDiscarded() {
        // An engine past its last checkpoint: it notices nothing and returns
        // text for audio the take has outgrown.
        val engine = PreviewEngine { cancel ->
            while (!cancel()) Thread.sleep(2)
            OnDeviceStreamSession.WindowResult.Text("stale")
        }
        val trace = StreamTrace()
        val session = tracedSession(engine, trace)
        val second = pcm(1.0)
        session.onAudio(second, second.size)
        assertTrue(engine.previewStarted.tryAcquire(5, TimeUnit.SECONDS))
        val half = pcm(0.5)
        session.onAudio(half, half.size)

        assertEquals(CommitOutcome.Final("n24000"), session.finish())
        assertFalse(events.any { it == StreamEvent.Partial("stale") })
        assertEquals(StreamTrace.RESULT_PREEMPTED, trace.calls().first().result)
    }

    @Test
    fun aPreviewFromAnEngineWithoutCheckpointsIsDiscardedOnceStopHasNewerAudio() {
        // The engine never polls the cancel predicate: the session must still
        // see, once the call returns, that Stop brought audio it does not cover.
        val release = CountDownLatch(1)
        val engine = PreviewEngine {
            release.await(5, TimeUnit.SECONDS)
            OnDeviceStreamSession.WindowResult.Text("stale")
        }
        val trace = StreamTrace()
        val session = tracedSession(engine, trace)
        val second = pcm(1.0)
        session.onAudio(second, second.size)
        assertTrue(engine.previewStarted.tryAcquire(5, TimeUnit.SECONDS))
        val half = pcm(0.5)
        session.onAudio(half, half.size)

        val outcome = java.util.concurrent.atomic.AtomicReference<CommitOutcome>()
        val stopper = Thread { outcome.set(session.finish()) }.apply { start() }
        while (trace.toJson().getJSONArray("commits").length() == 0) Thread.sleep(2)
        release.countDown()
        stopper.join(5_000)

        assertEquals(CommitOutcome.Final("n24000"), outcome.get())
        assertFalse(events.any { it is StreamEvent.Partial })
        assertEquals(StreamTrace.RESULT_PREEMPTED, trace.calls().first().result)
    }

    @Test
    fun aPreviewFromAnEarlierModelLoadIsNotReused() {
        val engine = PreviewEngine { OnDeviceStreamSession.WindowResult.Text("p", generation = 1) }
        engine.generation = 1
        val session = tracedSession(engine, StreamTrace())
        val second = pcm(1.0)
        session.onAudio(second, second.size)
        awaitEvent { it == StreamEvent.Partial("p") }

        // A driver failure elsewhere reloaded the engine before Stop.
        engine.generation = 2
        assertEquals(CommitOutcome.Final("n16000"), session.finish())
        assertEquals(listOf(16_000), engine.windows)
    }

    @Test
    fun anEngineThatDoesNotTrackItsLoadsNeverReusesAPreview() {
        val engine = PreviewEngine(tracksLoads = false) { OnDeviceStreamSession.WindowResult.Text("p") }
        val session = tracedSession(engine, StreamTrace())
        val second = pcm(1.0)
        session.onAudio(second, second.size)
        awaitEvent { it == StreamEvent.Partial("p") }

        assertEquals(CommitOutcome.Final("n16000"), session.finish())
        assertEquals(listOf(16_000), engine.windows)
    }

    @Test
    fun unknownGenerationsOnBothSidesAreNotTheSameLoad() {
        val engine = PreviewEngine { OnDeviceStreamSession.WindowResult.Text("p") }
        engine.generation = OnDeviceStreamSession.LiveEngine.UNKNOWN_GENERATION
        val session = tracedSession(engine, StreamTrace())
        val second = pcm(1.0)
        session.onAudio(second, second.size)
        awaitEvent { it == StreamEvent.Partial("p") }

        assertEquals(CommitOutcome.Final("n16000"), session.finish())
        assertEquals(listOf(16_000), engine.windows)
    }

    private companion object {
        /** Sample value step per second of test audio; distinct and exact in PCM16. */
        const val SECOND_STEP = 100
    }
}
