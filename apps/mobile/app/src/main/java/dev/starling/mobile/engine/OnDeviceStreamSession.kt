package dev.starling.mobile.engine

import dev.starling.mobile.network.CommitOutcome
import dev.starling.mobile.network.StreamEvent
import dev.starling.mobile.network.StreamSession
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Live transcription on this device: the on-device counterpart of the
 * `WS /stream` client, behind the same [StreamSession] contract, so the
 * recorder, the voice keyboard, and the recognition service show growing
 * partials without knowing where they come from.
 *
 * Capture chunks are decoded into a rolling float buffer; one worker thread
 * loads the model, then drives a [ChunkStreamer] over the buffer and emits
 * [StreamEvent.Partial] whenever the text changes. [finish] finalizes the
 * remaining tail, so the final transcript is ready moments after Stop
 * instead of after a full pass over the recording. As with the server
 * stream, the saved WAV stays the source of truth: any engine failure or a
 * worker that falls too far behind interrupts the stream, and [finish]
 * answers [CommitOutcome.Fallback] so the batch path transcribes the WAV.
 */
class OnDeviceStreamSession(
    private val engine: LiveEngine,
    private val events: (StreamEvent) -> Unit,
    private val streamer: ChunkStreamer = ChunkStreamer(),
    private val clock: () -> Double = { System.nanoTime() / 1e9 },
    private val maxLiveSamples: Int = MAX_LIVE_SECONDS * ChunkStreamer.SAMPLE_RATE,
) : StreamSession {
    /** The engine surface the session needs; [OnDeviceEngine] in production. */
    interface LiveEngine {
        /** Loads the model if needed; null when ready, else the reason it cannot run. */
        fun prepare(): String?

        /** Transcribes one window of 16 kHz mono samples. */
        fun transcribeWindow(samples: FloatArray): WindowResult

        /** Brackets a session so the engine never unloads under a live recording. */
        fun liveSessionStarted() = Unit
        fun liveSessionEnded() = Unit
    }

    sealed interface WindowResult {
        data class Text(val text: String) : WindowResult
        data class Failed(val reason: String) : WindowResult
    }

    private val lock = ReentrantLock()
    private val changed = lock.newCondition()
    private val settled = CountDownLatch(1)

    // Guarded by [lock]. The capture thread appends; the worker snapshots
    // and trims finalized audio from the front.
    private var buffer = FloatArray(ChunkStreamer.SAMPLE_RATE * 4)
    private var size = 0
    private var inputEnded = false
    private var closed = false
    private var failure: String? = null
    private var outcome: CommitOutcome? = null

    // The Interrupted event of a failure, emitted once by the worker thread:
    // a failure noticed on the capture thread (the buffer cap) is recorded
    // there but reported from the worker, so every event comes from one thread.
    private var interruption: StreamEvent.Interrupted? = null
    private var interruptionEmitted = false

    private val worker = Thread(::run, "starling-on-device-stream").apply { isDaemon = true }

    fun start(): OnDeviceStreamSession = apply { worker.start() }

    override fun acceptsAudio(): Boolean = lock.withLock { acceptsAudioLocked() }

    override fun onAudio(bytes: ByteArray, count: Int) {
        val samples = count / 2
        if (samples <= 0) return
        lock.withLock {
            if (!acceptsAudioLocked()) return
            if (size + samples > maxLiveSamples) {
                // The engine is not keeping up with real time; stop before the
                // buffer (and the eventual flush) grows without bound.
                failLocked("the on-device engine fell behind the recording", bufferLimitReached = true)
                return
            }
            if (size + samples > buffer.size) {
                buffer = buffer.copyOf(maxOf(size + samples, buffer.size * 2))
            }
            // PCM16 little-endian, exactly as AudioCapture writes the WAV.
            for (i in 0 until samples) {
                val lo = bytes[2 * i].toInt() and 0xff
                val hi = bytes[2 * i + 1].toInt()
                buffer[size + i] = ((hi shl 8) or lo).toShort() / 32768f
            }
            size += samples
            changed.signalAll()
        }
    }

    override fun finish(): CommitOutcome {
        lock.withLock {
            inputEnded = true
            changed.signalAll()
        }
        settled.await()
        return lock.withLock { requireNotNull(outcome) { "stream settled without an outcome" } }
    }

    override fun close() {
        lock.withLock {
            closed = true
            settleLocked(CommitOutcome.Fallback("the live session was closed"))
            changed.signalAll()
        }
    }

    private fun run() {
        engine.liveSessionStarted()
        try {
            runLoop()
        } catch (t: Throwable) {
            // Whatever broke, finish() must never block forever: settle as a
            // fallback so the saved WAV goes through the batch path.
            lock.withLock { failLocked(t.message ?: t::class.java.simpleName, bufferLimitReached = false) }
        } finally {
            engine.liveSessionEnded()
            emitInterruption()
        }
    }

    private fun runLoop() {
        val loadError = runCatching { engine.prepare() }.getOrElse { it.message ?: it::class.java.simpleName }
        if (loadError != null) {
            lock.withLock { failLocked(loadError, bufferLimitReached = false) }
            return
        }
        // Closed or already failed (e.g. the buffer cap) while the model loaded.
        if (lock.withLock { closed || failure != null }) return
        events(StreamEvent.Live)

        var steppedSize = -1
        var lastPartial: String? = null
        var windowFailure: String? = null
        val tx = ChunkStreamer.Transcriber { samples, start, length ->
            // The snapshot is exactly the live tail, so a window that spans all
            // of it (every flush, most partials) is passed without a copy.
            val window = if (start == 0 && length == samples.size) samples else samples.copyOfRange(start, start + length)
            when (val result = runCatching { engine.transcribeWindow(window) }
                .getOrElse { WindowResult.Failed(it.message ?: it::class.java.simpleName) }) {
                is WindowResult.Text -> result.text
                is WindowResult.Failed -> {
                    windowFailure = result.reason
                    null
                }
            }
        }
        while (true) {
            val snapshot: FloatArray
            val snapshotSize: Int
            val ending: Boolean
            lock.withLock {
                while (!closed && failure == null && !inputEnded && size == steppedSize) {
                    changed.await(IDLE_WAIT_MILLIS, TimeUnit.MILLISECONDS)
                }
                if (closed || failure != null) return
                ending = inputEnded
                snapshot = buffer.copyOf(size)
                snapshotSize = size
            }
            steppedSize = snapshotSize
            windowFailure = null

            if (ending) {
                val text = streamer.flush(snapshot, snapshotSize, tx)
                lock.withLock {
                    if (text != null) {
                        settleLocked(CommitOutcome.Final(text))
                    } else {
                        failLocked(windowFailure ?: "the on-device engine failed", bufferLimitReached = false)
                    }
                }
                return
            }

            val partial = streamer.step(snapshot, snapshotSize, clock(), tx)
            windowFailure?.let { reason ->
                lock.withLock { failLocked(reason, bufferLimitReached = false) }
                return
            }
            if (partial != null && partial != lastPartial) {
                lastPartial = partial
                events(StreamEvent.Partial(partial))
            }
            steppedSize -= trimFinalized()
            // Throttle: a step that transcribed nothing new must not spin.
            if (partial == null) sleepQuietly(STEP_BACKOFF_MILLIS)
        }
    }

    /**
     * Drops finalized audio from the buffer front after every step, so the
     * next snapshot copies only the live tail. Returns the samples dropped.
     */
    private fun trimFinalized(): Int {
        val dropped = streamer.boundary
        if (dropped == 0) return 0
        lock.withLock {
            System.arraycopy(buffer, dropped, buffer, 0, size - dropped)
            size -= dropped
        }
        streamer.rebase(dropped)
        return dropped
    }

    private fun sleepQuietly(millis: Long) {
        lock.withLock {
            if (!closed && !inputEnded && failure == null) changed.await(millis, TimeUnit.MILLISECONDS)
        }
    }

    private fun acceptsAudioLocked(): Boolean = !closed && !inputEnded && failure == null

    /** Records a failure and wakes the worker, which reports it (see [emitInterruption]). */
    private fun failLocked(reason: String, bufferLimitReached: Boolean) {
        if (failure != null || outcome != null) return
        failure = reason
        if (!closed) interruption = StreamEvent.Interrupted(reason, bufferLimitReached)
        settleLocked(CommitOutcome.Fallback(reason))
        changed.signalAll()
    }

    /** Worker thread only, outside the lock: reports a recorded failure once. */
    private fun emitInterruption() {
        val event = lock.withLock {
            if (interruptionEmitted || closed) return
            interruptionEmitted = true
            interruption
        } ?: return
        events(event)
    }

    private fun settleLocked(result: CommitOutcome) {
        if (outcome != null) return
        outcome = result
        settled.countDown()
    }

    companion object {
        /** Live (unfinalized) audio bound, like the server's 60 s stream buffer cap. */
        const val MAX_LIVE_SECONDS = 60
        private const val IDLE_WAIT_MILLIS = 250L
        private const val STEP_BACKOFF_MILLIS = 100L
    }
}
