package dev.starling.mobile.engine

import dev.starling.mobile.audio.WavWriter
import dev.starling.mobile.data.RecordingStatus
import dev.starling.mobile.network.InferenceResult
import dev.starling.mobile.network.transcribeStoredAudio
import dev.starling.mobile.storage.RecordingStore
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

/**
 * #356: an on-device transcription whose model call hangs fails its
 * attempt (retryable, audio kept) instead of staying "transcribing", and a
 * result the abandoned call returns later is dropped.
 */
class BoundedTranscriptionTest {
    @get:Rule
    val folder = TemporaryFolder()

    /**
     * A stand-in for OnDeviceEngine: one call at a time under its lock, the
     * age of the native call in progress, and a native call that either
     * honors the cooperative cancel or ignores it like a wedged GPU
     * submission.
     */
    private class FakeEngine {
        val lock = Object()
        val nativeCallStarted = AtomicLong(NONE)
        val calls = AtomicInteger()
        val cancelled = AtomicBoolean()

        fun nativeCallAgeMillis(): Long {
            val since = nativeCallStarted.get()
            return if (since == NONE) 0 else System.nanoTime() / 1_000_000 - since
        }

        fun transcribe(attempt: CallAttempt, native: (CallAttempt) -> InferenceResult): InferenceResult =
            synchronized(lock) {
                if (!attempt.start()) return attempt.stopped()
                calls.incrementAndGet()
                nativeCallStarted.set(System.nanoTime() / 1_000_000)
                try {
                    native(attempt)
                } finally {
                    nativeCallStarted.set(NONE)
                }
            }

        /** A native call that polls the cancel like transcribeCancellable, for up to [millis]. */
        fun cooperative(attempt: CallAttempt, millis: Long = 10_000): InferenceResult {
            val deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(millis)
            while (System.nanoTime() < deadline) {
                if (attempt.cancelRequested()) {
                    cancelled.set(true)
                    return attempt.stopped()
                }
                Thread.sleep(1)
            }
            return InferenceResult.Success("slow but done")
        }

        companion object {
            const val NONE = Long.MIN_VALUE
        }
    }

    private fun bound(
        engine: FakeEngine,
        nativeCallLimitMillis: Long = 10_000,
        graceMillis: Long = 100,
        spawned: AtomicInteger = AtomicInteger(),
    ) = BoundedTranscription(
        nativeCallAgeMillis = engine::nativeCallAgeMillis,
        nativeCallLimitMillis = nativeCallLimitMillis,
        graceMillis = graceMillis,
        pollMillis = 1,
        spawn = { call ->
            spawned.incrementAndGet()
            Thread(call, "test-on-device-call").apply { isDaemon = true }.start()
        },
    )

    @Test
    fun aCallThatFinishesReturnsItsResult() {
        val engine = FakeEngine()
        val result = bound(engine).run(budgetMillis = 10_000) { attempt ->
            engine.transcribe(attempt) { InferenceResult.Success("hello", "model.gguf") }
        }
        assertEquals(InferenceResult.Success("hello", "model.gguf"), result)
    }

    @Test
    fun aCallPastItsBudgetIsCancelledAndFailsRetryable() {
        val engine = FakeEngine()
        val result = bound(engine).run(budgetMillis = 50) { attempt ->
            engine.transcribe(attempt) { engine.cooperative(it) }
        }
        assertTrue(engine.cancelled.get())
        result as InferenceResult.Failure
        assertTrue(result.retryable)
        assertTrue(result.message, result.message.startsWith("On-device transcription did not finish within 1 minutes"))
        assertTrue(result.message.contains("The recording is kept"))
    }

    @Test
    fun aNativeCallOverTheLimitIsStoppedBeforeTheBudget() {
        val engine = FakeEngine()
        val result = bound(engine, nativeCallLimitMillis = 50).run(budgetMillis = 60_000) { attempt ->
            engine.transcribe(attempt) { engine.cooperative(it) }
        }
        assertTrue(engine.cancelled.get())
        assertEquals(InferenceResult.Failure(BoundedTranscription.NO_PROGRESS, true), result)
    }

    /**
     * A call that ignores the cancel (a wedged GPU submission) is
     * abandoned: the attempt fails, the recording is marked failed with its
     * audio, and the result the call returns later is never stored.
     */
    @Test
    fun aHungCallIsAbandonedAndItsLateResultIsDropped() {
        val store = RecordingStore(File(folder.root, "recordings"))
        val recording = store.create()
        WavWriter(store.partialFile(recording)).apply {
            write(ByteArray(32_000), 32_000)
            finish()
        }
        val take = store.commitAudio(recording, 1.0)
        store.markTranscribing(take.id)
        val engine = FakeEngine()
        val wedged = CountDownLatch(1)
        val returned = CountDownLatch(1)
        val bound = bound(engine, nativeCallLimitMillis = 10_000, graceMillis = 50)

        val result = transcribeStoredAudio(store, take.id) { audio ->
            bound.run(budgetMillis = 50) { attempt ->
                engine.transcribe(attempt) {
                    wedged.await()
                    returned.countDown()
                    InferenceResult.Success("late ${audio.length()}")
                }
            }
        }
        // What TranscriptionCoordinator does with the failure.
        val failed = store.markFailed(take.id, (result as InferenceResult.Failure).message)

        assertEquals(InferenceResult.Failure(BoundedTranscription.ABANDONED, true), result)
        assertEquals(RecordingStatus.FAILED, failed.status)
        // The call comes back after all: nothing reads its result.
        wedged.countDown()
        assertTrue(returned.await(5, TimeUnit.SECONDS))
        Thread.sleep(20)
        val after = store.get(take.id)
        assertEquals(RecordingStatus.FAILED, after.status)
        assertEquals(BoundedTranscription.ABANDONED, after.errorMessage)
        assertEquals(null, after.rawTranscript)
        assertTrue(after.revisions.isEmpty())
        // The audio is kept, and a retry transcribes it.
        val retried = transcribeStoredAudio(store, take.id) { audio ->
            bound.run(budgetMillis = 10_000) { attempt ->
                engine.transcribe(attempt) { InferenceResult.Success("retried ${audio.length()}") }
            }
        }
        assertEquals(InferenceResult.Success("retried ${44 + 32_000}"), retried)
    }

    /**
     * While an abandoned call still hangs in the engine, a retry fails at
     * once, without starting a thread that would queue behind it.
     */
    @Test
    fun aRetryOnAStuckEngineFailsFast() {
        val engine = FakeEngine()
        val spawned = AtomicInteger()
        val wedged = CountDownLatch(1)
        val bound = bound(engine, nativeCallLimitMillis = 30, graceMillis = 20, spawned = spawned)
        val first = bound.run(budgetMillis = 60_000) { attempt ->
            engine.transcribe(attempt) {
                wedged.await()
                InferenceResult.Success("late")
            }
        }
        assertEquals(InferenceResult.Failure(BoundedTranscription.ABANDONED, true), first)
        assertEquals(1, spawned.get())

        val retry = bound.run(budgetMillis = 60_000) { error("a stuck engine must not be called") }

        assertEquals(InferenceResult.Failure(BoundedTranscription.ENGINE_STUCK, true), retry)
        assertEquals(1, spawned.get())
        wedged.countDown()
    }

    /**
     * An attempt waiting for the engine behind a long but healthy call is
     * not timed, whatever its budget; it runs once the engine is free.
     */
    @Test
    fun waitingBehindAHealthyCallIsNotATimeout() {
        val engine = FakeEngine()
        val bound = bound(engine)
        val longCall = CountDownLatch(1)
        val holding = CountDownLatch(1)
        val first = Thread {
            bound.run(budgetMillis = 60_000) { attempt ->
                engine.transcribe(attempt) {
                    holding.countDown()
                    longCall.await()
                    InferenceResult.Success("first")
                }
            }
        }.apply { start() }
        assertTrue(holding.await(5, TimeUnit.SECONDS))
        val second = AtomicReference<InferenceResult>()
        val waiter = Thread {
            second.set(bound.run(budgetMillis = 50) { attempt -> engine.transcribe(attempt) { InferenceResult.Success("second") } })
        }.apply { start() }
        Thread.sleep(200)
        assertEquals(null, second.get())
        longCall.countDown()
        waiter.join(5_000)
        first.join(5_000)
        assertEquals(InferenceResult.Success("second"), second.get())
    }

    /**
     * An attempt waiting for the engine gives up once the call ahead of it
     * is stuck, and never starts if the engine frees up later.
     */
    @Test
    fun waitingBehindAStuckCallGivesUpAndNeverStarts() {
        val engine = FakeEngine()
        val bound = bound(engine, nativeCallLimitMillis = 50, graceMillis = 20)
        val wedged = CountDownLatch(1)
        val holding = CountDownLatch(1)
        // Like a live-stream window that hangs: not under any bound of its own.
        val stuck = Thread {
            synchronized(engine.lock) {
                engine.nativeCallStarted.set(System.nanoTime() / 1_000_000)
                holding.countDown()
                wedged.await()
                engine.nativeCallStarted.set(FakeEngine.NONE)
            }
        }.apply { start() }
        assertTrue(holding.await(5, TimeUnit.SECONDS))

        val result = bound.run(budgetMillis = 60_000) { attempt ->
            engine.transcribe(attempt) { InferenceResult.Success("never") }
        }

        assertEquals(InferenceResult.Failure(BoundedTranscription.ENGINE_STUCK, true), result)
        wedged.countDown()
        stuck.join(5_000)
        Thread.sleep(50)
        assertEquals(0, engine.calls.get())
    }

    @Test
    fun anEngineExceptionReachesTheCaller() {
        val engine = FakeEngine()
        val thrown = runCatching {
            bound(engine).run(budgetMillis = 10_000) { throw OutOfMemoryError("decode") }
        }.exceptionOrNull()
        assertTrue(thrown is OutOfMemoryError)
    }

    @Test
    fun anAttemptStoppedBeforeItStartsNeverStarts() {
        val attempt = CallAttempt { 0L }
        attempt.stop("given up")
        assertFalse(attempt.start())
        assertEquals(InferenceResult.Failure("given up", true), attempt.stopped())
        assertEquals(null, attempt.startedAt())
    }

    @Test
    fun theBudgetScalesWithTheAudio() {
        val base = BoundedTranscription.BASE_BUDGET_MILLIS
        assertEquals(base, BoundedTranscription.budgetFor(44))
        // One minute of 16 kHz PCM16 adds a minute; two hours add two hours.
        assertEquals(base + 60_000, BoundedTranscription.budgetFor(44 + 1_920_000L))
        assertEquals(base + 7_200_000, BoundedTranscription.budgetFor(44 + 120 * 1_920_000L))
    }
}
