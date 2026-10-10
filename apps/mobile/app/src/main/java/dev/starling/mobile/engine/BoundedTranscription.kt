package dev.starling.mobile.engine

import dev.starling.mobile.network.InferenceResult
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * One bounded on-device transcription (#356), shared by the side that
 * waits for it ([BoundedTranscription]) and the engine that runs it. The
 * engine [start]s it once it holds the model lock, and polls
 * [cancelRequested] before each window and inside each native call
 * (StarlingNative.transcribeCancellable). The waiting side [stop]s it with
 * the reason the attempt failed; an attempt stopped before it started
 * never starts.
 */
class CallAttempt internal constructor(private val clock: () -> Long) {
    private var startedAt = NOT_STARTED

    @Volatile
    var stopReason: String? = null
        private set

    /** The engine holds the model for this attempt; false when it was given up already. */
    @Synchronized
    fun start(): Boolean {
        if (stopReason != null) return false
        startedAt = clock()
        return true
    }

    fun cancelRequested(): Boolean = stopReason != null

    /** The failure an attempt stopped on request reports: retryable, the audio is kept. */
    fun stopped(): InferenceResult.Failure =
        InferenceResult.Failure(stopReason ?: BoundedTranscription.INTERRUPTED, true)

    /** When the engine started this attempt; null while it waits for the engine. */
    @Synchronized
    internal fun startedAt(): Long? = startedAt.takeIf { it != NOT_STARTED }

    /** Stops the attempt; the first reason stays. */
    @Synchronized
    internal fun stop(reason: String) {
        if (stopReason == null) stopReason = reason
    }

    private companion object {
        const val NOT_STARTED = Long.MIN_VALUE
    }
}

/**
 * Bounds on-device batch transcriptions (#356): a native call that hangs
 * must fail its attempt, retryable and with its audio kept, instead of
 * leaving the recording "transcribing" until the app restarts.
 *
 * [run] runs the engine call on a thread of its own and waits for it. The
 * attempt is stopped when, after the engine started it, it runs past its
 * budget ([budgetMillis], scaled by the audio length), or when one native
 * call of the engine runs longer than [nativeCallLimitMillis]
 * ([nativeCallAgeMillis] is the age of the call in progress; native calls
 * are bounded windows of at most ~32 s of audio, or a model load). The
 * stop goes through the engine's cooperative cancel; a call that does not
 * return within [graceMillis] after it is abandoned: the attempt fails at
 * once and whatever the call still returns is dropped, never stored. A
 * call that returns within the grace is not abandoned and its result
 * stands: a transcript that completed just past the budget is the same
 * audio's correct text, and dropping it would only make the user redo
 * the work.
 *
 * An attempt still waiting for the engine (another call holds it) is not
 * timed: a long transcription ahead of it is legitimate. It gives up only
 * when the engine is stuck, i.e. its call in progress is over the limit;
 * that is also checked before an attempt is started at all, so retries on
 * a wedged engine fail fast instead of queueing behind it.
 */
class BoundedTranscription(
    private val nativeCallAgeMillis: () -> Long,
    private val nativeCallLimitMillis: Long = NATIVE_CALL_LIMIT_MILLIS,
    private val graceMillis: Long = CANCEL_GRACE_MILLIS,
    private val pollMillis: Long = POLL_MILLIS,
    private val clock: () -> Long = { System.nanoTime() / 1_000_000 },
    private val spawn: (Runnable) -> Unit = { call ->
        Thread(call, "starling-on-device-call").apply { isDaemon = true }.start()
    },
) {
    fun run(budgetMillis: Long, call: (CallAttempt) -> InferenceResult): InferenceResult {
        if (engineStuck()) return InferenceResult.Failure(ENGINE_STUCK, true)
        val attempt = CallAttempt(clock)
        val outcome = Outcome()
        spawn {
            try {
                outcome.result = call(attempt)
            } catch (t: Throwable) {
                outcome.error = t
            } finally {
                outcome.done.countDown()
            }
        }
        var stoppedAt: Long? = null
        try {
            while (true) {
                if (outcome.done.await(pollMillis, TimeUnit.MILLISECONDS)) return outcome.get()
                val now = clock()
                val stopped = stoppedAt
                if (stopped != null) {
                    // The cooperative cancel did not land: abandon the call.
                    // Its late result goes nowhere.
                    if (now - stopped >= graceMillis) return InferenceResult.Failure(ABANDONED, true)
                    continue
                }
                val startedAt = attempt.startedAt()
                val reason = when {
                    startedAt == null -> if (engineStuck()) ENGINE_STUCK else null
                    // Only one native call runs at a time, under the engine lock
                    // this attempt holds: the stuck call is its own.
                    engineStuck() -> NO_PROGRESS
                    now - startedAt > budgetMillis -> timedOut(budgetMillis)
                    else -> null
                } ?: continue
                attempt.stop(reason)
                // Not started (the engine never got to it) and now never will.
                if (attempt.startedAt() == null) return InferenceResult.Failure(reason, true)
                stoppedAt = now
            }
        } catch (_: InterruptedException) {
            attempt.stop(INTERRUPTED)
            Thread.currentThread().interrupt()
            return InferenceResult.Failure(INTERRUPTED, true)
        }
    }

    private fun engineStuck(): Boolean = nativeCallAgeMillis() > nativeCallLimitMillis

    private class Outcome {
        val done = CountDownLatch(1)

        @Volatile
        var result: InferenceResult? = null

        @Volatile
        var error: Throwable? = null

        /** The call's result, or its exception rethrown for the caller to settle. */
        fun get(): InferenceResult {
            error?.let { throw it }
            return requireNotNull(result) { "the on-device call ended without a result" }
        }
    }

    companion object {
        /** A take's budget beyond its own length: the model load and slack. */
        const val BASE_BUDGET_MILLIS = 5 * 60_000L

        /**
         * The longest one native call may run before the engine counts as
         * stuck: a window is at most ~32 s of audio (seconds on the Pixel's
         * GPU, well under a minute on the CPU fallback), a model load
         * seconds.
         */
        const val NATIVE_CALL_LIMIT_MILLIS = 5 * 60_000L

        /** How long a stopped call has to return before it is abandoned. */
        const val CANCEL_GRACE_MILLIS = 20_000L

        private const val POLL_MILLIS = 1_000L

        /**
         * The budget of a transcription of [wavBytes] of 16 kHz mono PCM16:
         * [BASE_BUDGET_MILLIS] plus the audio's own length. The engine runs
         * many times faster than real time, so only a stuck one reaches it.
         */
        fun budgetFor(wavBytes: Long): Long =
            BASE_BUDGET_MILLIS + (wavBytes - WAV_HEADER_BYTES).coerceAtLeast(0) * 1_000 / PCM_BYTES_PER_SECOND

        private const val WAV_HEADER_BYTES = 44L
        private const val PCM_BYTES_PER_SECOND = 16_000L * 2

        private fun timedOut(budgetMillis: Long): String {
            val minutes = (budgetMillis + 59_999) / 60_000
            return "On-device transcription did not finish within $minutes minutes and was stopped. " +
                "The recording is kept; retry to transcribe it again."
        }

        const val NO_PROGRESS = "The on-device model stopped making progress and the transcription was stopped. " +
            "The recording is kept; retry to transcribe it again."
        const val ABANDONED = "The on-device model stopped responding. The recording is kept; restart Starling " +
            "to transcribe on this device again, or retry with a server."
        const val ENGINE_STUCK = "The on-device model is stuck on an earlier call. The recording is kept; restart " +
            "Starling to transcribe on this device again, or retry with a server."
        const val INTERRUPTED = "The on-device transcription was interrupted. The recording is kept; retry to " +
            "transcribe it again."
    }
}
