package dev.starling.mobile.engine

import dev.starling.mobile.network.CommitOutcome
import dev.starling.mobile.network.StreamEvent
import dev.starling.mobile.network.StreamSession
import java.io.Closeable
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
 * instead of after a full pass over the recording.
 *
 * The capture never waits for the model (#229): audio is saved from the
 * first sample while the worker is still loading, and the samples that
 * arrived meanwhile are fed to the streamer in order once the model is
 * ready. The in-memory buffer is bounded by [maxLiveSamples]; audio beyond
 * it (a slow cold load, or an engine that fell behind) stays only in the
 * recording's saved WAV, which [backlog] opens for the worker to read back.
 * Without a [backlog] the bound interrupts the stream as before.
 *
 * As with the server stream, the saved WAV stays the source of truth: any
 * engine failure interrupts the stream, and [finish] answers
 * [CommitOutcome.Fallback] so the batch path transcribes the WAV.
 *
 * A [trace] (debug builds only, see [StreamDebug]) records every engine
 * call, capture chunk, partial and the stop path for the #226 measurements.
 */
class OnDeviceStreamSession(
    private val engine: LiveEngine,
    private val events: (StreamEvent) -> Unit,
    private val streamer: ChunkStreamer = ChunkStreamer(),
    private val clock: () -> Double = { System.nanoTime() / 1e9 },
    private val maxLiveSamples: Int = MAX_LIVE_SECONDS * ChunkStreamer.SAMPLE_RATE,
    private val backlog: (() -> Backlog)? = null,
    private val trace: StreamTrace? = null,
) : StreamSession {
    /** The engine surface the session needs; [OnDeviceEngine] in production. */
    interface LiveEngine {
        /**
         * Loads the model if needed; null when ready, else the reason it
         * cannot run. A successful prepare pins the loaded model to this
         * session until [liveSessionEnded] with `prepared = true`. Gives up,
         * without loading, once [cancelled] (the session was closed) holds.
         */
        fun prepare(cancelled: () -> Boolean): String?

        /** Transcribes one window of 16 kHz mono samples. */
        fun transcribeWindow(samples: FloatArray): WindowResult

        /**
         * Transcribes a preview of the live tail that the session may stop
         * needing mid-call: the engine stops at its next checkpoint once
         * [cancel] holds and answers [WindowResult.Cancelled]. An engine
         * without checkpoints finishes the call; the session then discards
         * a result that completed after [cancel] fired.
         */
        fun transcribePreview(samples: FloatArray, cancel: () -> Boolean): WindowResult = transcribeWindow(samples)

        /** The model a successful [prepare] pinned to this session, for the transcript's record. */
        fun loadedModelName(): String? = null

        /**
         * Brackets a session so the engine never unloads the model a live
         * recording uses; [prepared] tells whether [prepare] succeeded.
         */
        fun liveSessionStarted() = Unit
        fun liveSessionEnded(prepared: Boolean) = Unit
    }

    sealed interface WindowResult {
        /** [model]: the model that transcribed this window, when the engine knows it. */
        data class Text(val text: String, val model: String? = null) : WindowResult
        data class Failed(val reason: String) : WindowResult

        /** A preview stopped because the session no longer needed it; not a failure. */
        data object Cancelled : WindowResult
    }

    /**
     * Random access to the audio the capture has already saved, for samples
     * the bounded live buffer could not hold. Opened on the capture thread
     * the first time the buffer is full (the recording's file exists then),
     * read and closed by the worker.
     */
    interface Backlog : Closeable {
        /**
         * Reads [count] samples starting at absolute sample [from] into [into]
         * at [offset] as floats in [-1, 1]; returns how many it read.
         */
        fun read(from: Long, into: FloatArray, offset: Int, count: Int): Int
    }

    private val lock = ReentrantLock()
    private val changed = lock.newCondition()
    private val settled = CountDownLatch(1)

    // Guarded by [lock]. The buffer holds the samples [base, base + size) of
    // the recording, out of [captured] delivered so far. The capture thread
    // appends while the buffer is complete (base + size == captured) and has
    // room; otherwise the samples stay in the saved WAV ([openBacklog]) and
    // the worker reads them back in order. The worker snapshots and trims
    // finalized audio from the front.
    private var buffer = FloatArray(INITIAL_BUFFER_SAMPLES)
    private var size = 0
    private var base = 0L
    // Volatile as well: a running preview's cancel predicate reads it unlocked.
    @Volatile
    private var captured = 0L
    private var openBacklog: Backlog? = null
    @Volatile
    private var inputEnded = false
    // Written under [lock]; volatile so the engine can poll it from prepare().
    @Volatile
    private var closed = false
    private var failure: String? = null
    private var outcome: CommitOutcome? = null

    // The Interrupted event of a failure, emitted once by the worker thread:
    // a failure noticed on the capture thread (the buffer cap) is recorded
    // there but reported from the worker, so every event comes from one thread.
    private var interruption: StreamEvent.Interrupted? = null
    private var interruptionEmitted = false

    private val worker = Thread(::run, "starling-on-device-stream").apply { isDaemon = true }

    // Worker thread only: whether prepare() succeeded and pinned the model.
    private var prepared = false

    // Worker thread only (refillFromBacklog): the read destination, reused
    // across refills and grown on demand, so catching up on a cold load does
    // not allocate a fresh maxLiveSamples array per loop iteration.
    private var refillScratch = FloatArray(0)

    fun start(): OnDeviceStreamSession = apply {
        // Counted before the worker exists, so a memory-pressure release can
        // never slip in between start() and the worker's first instruction.
        engine.liveSessionStarted()
        try {
            worker.start()
        } catch (t: Throwable) {
            engine.liveSessionEnded(prepared = false)
            throw t
        }
    }

    override fun acceptsAudio(): Boolean = lock.withLock { acceptsAudioLocked() }

    override fun onAudio(bytes: ByteArray, count: Int) {
        val samples = count / 2
        if (samples <= 0) return
        lock.withLock {
            if (!acceptsAudioLocked()) return
            if (base + size == captured && size + samples <= maxLiveSamples) {
                appendLocked(bytes, samples)
            } else if (!spillLocked()) {
                // No saved audio to fall back on: the engine is not keeping up
                // with real time; stop before the buffer (and the eventual
                // flush) grows without bound. A backlog that exists but could
                // not be opened already failed the stream with its own cause
                // (see [spillLocked]); failLocked is first-wins, so this cap
                // failure stays the genuine no-backlog case.
                failLocked("the on-device engine fell behind the recording", bufferLimitReached = true)
                return
            }
            captured += samples
            trace?.audio(captured, samples)
            changed.signalAll()
        }
    }

    private fun appendLocked(bytes: ByteArray, samples: Int) {
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
    }

    /**
     * Leaves a chunk in the saved WAV only (the buffer is full, or behind);
     * false when there is no saved audio to read it back from. A backlog
     * that exists but cannot be opened fails the stream right here with
     * the real cause, so an I/O failure is not misreported as the engine
     * falling behind (the caller's cap message applies to the genuine
     * no-backlog case only, and [failLocked] is first-wins).
     */
    private fun spillLocked(): Boolean {
        val opener = backlog ?: return false
        if (openBacklog == null) {
            openBacklog = runCatching(opener).onFailure { t ->
                failLocked(
                    "the saved recording could not be opened: ${t.message ?: t::class.java.simpleName}",
                    bufferLimitReached = false,
                )
            }.getOrNull()
        }
        return openBacklog != null
    }

    override fun finish(): CommitOutcome {
        trace?.stopRequested()
        lock.withLock {
            inputEnded = true
            changed.signalAll()
        }
        // The worker settles on every path, including crashes; the bound only
        // guards against a native call that never returns, so Stop cannot
        // hang forever (the WS client caps its wait the same way).
        if (!settled.await(FINISH_TIMEOUT_MINUTES, TimeUnit.MINUTES)) {
            lock.withLock { settleLocked(CommitOutcome.Fallback("the on-device engine did not finish in time")) }
        }
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
        try {
            runLoop()
        } catch (t: Throwable) {
            // Whatever broke, finish() must never block forever: settle as a
            // fallback so the saved WAV goes through the batch path.
            lock.withLock { failLocked(t.message ?: t::class.java.simpleName, bufferLimitReached = false) }
        } finally {
            lock.withLock {
                runCatching { openBacklog?.close() }
                openBacklog = null
            }
            engine.liveSessionEnded(prepared)
            emitInterruption()
            trace?.let { runCatching { it.complete() } }
        }
    }

    private fun runLoop() {
        val prepareStarted = clock()
        val loadError = runCatching { engine.prepare { closed } }.getOrElse { it.message ?: it::class.java.simpleName }
        if (loadError != null) {
            lock.withLock { failLocked(loadError, bufferLimitReached = false) }
            return
        }
        prepared = true
        trace?.prepared(prepareStarted)
        val model = runCatching { engine.loadedModelName() }.getOrNull()
        // Closed or already failed (e.g. the buffer cap) while the model loaded.
        if (lock.withLock { closed || failure != null }) return
        events(StreamEvent.Live)

        var steppedSize = -1
        var lastPartial: String? = null
        var windowFailure: String? = null
        // The models that transcribed windows: a driver failure voids the
        // pin, and the reload may pick up another active model mid-take.
        val windowModels = LinkedHashSet<String>()
        // Absolute sample index of the current snapshot's first sample, for the trace.
        var snapshotBase = 0L
        // Whether the last step's preview was preempted (see [previewObsolete]).
        var preempted = false
        val tx = ChunkStreamer.Transcriber { samples, start, length, kind ->
            // The snapshot is exactly the live tail, so a window that spans all
            // of it (every flush, most partials) is passed without a copy.
            val window = if (start == 0 && length == samples.size) samples else samples.copyOfRange(start, start + length)
            val t0 = clock()
            var obsolete = false
            val result = runCatching {
                if (kind == ChunkStreamer.CallKind.PREVIEW) {
                    // The preview starts at the window boundary.
                    val end = snapshotBase + start + length
                    val windowEnd = snapshotBase + start + streamer.windowSamples
                    engine.transcribePreview(window) { obsolete || previewObsolete(end, windowEnd).also { obsolete = it } }
                } else {
                    engine.transcribeWindow(window)
                }
            }.getOrElse { WindowResult.Failed(it.message ?: it::class.java.simpleName) }
            // A preview that completed after it became obsolete (past the
            // engine's last checkpoint) is discarded like a cancelled one:
            // its audio stays buffered for the window or flush that follows.
            val outcome = if (obsolete && result is WindowResult.Text) WindowResult.Cancelled else result
            trace?.call(
                StreamTrace.Call(
                    kind,
                    snapshotBase + start,
                    length,
                    t0,
                    clock(),
                    when (outcome) {
                        is WindowResult.Text -> StreamTrace.RESULT_OK
                        is WindowResult.Failed -> StreamTrace.RESULT_FAILED
                        WindowResult.Cancelled -> StreamTrace.RESULT_PREEMPTED
                    },
                ),
            )
            when (outcome) {
                is WindowResult.Text -> {
                    outcome.model?.let(windowModels::add)
                    outcome.text
                }
                is WindowResult.Failed -> {
                    windowFailure = outcome.reason
                    null
                }
                WindowResult.Cancelled -> {
                    preempted = true
                    null
                }
            }
        }
        while (true) {
            if (!refillFromBacklog()) return
            val snapshot: FloatArray
            val snapshotSize: Int
            val ending: Boolean
            val behind: Boolean
            lock.withLock {
                while (!closed && failure == null && !inputEnded && size == steppedSize && !behindLocked()) {
                    changed.await(IDLE_WAIT_MILLIS, TimeUnit.MILLISECONDS)
                }
                if (closed || failure != null) return
                behind = behindLocked()
                // Stop finalizes only once the backlog has been read back.
                ending = inputEnded && !behind
                snapshot = buffer.copyOf(size)
                snapshotSize = size
                snapshotBase = base
            }
            steppedSize = snapshotSize
            windowFailure = null
            preempted = false

            if (ending) {
                trace?.flushing((snapshotSize - streamer.boundary).toLong())
                val tailStart = snapshotBase + streamer.boundary
                val text = streamer.flush(snapshot, snapshotSize, tx)
                if (streamer.flushReusedTail) {
                    val now = clock()
                    trace?.call(
                        StreamTrace.Call(
                            ChunkStreamer.CallKind.FLUSH_TAIL,
                            tailStart,
                            (snapshotBase + snapshotSize - tailStart).toInt(),
                            now,
                            now,
                            StreamTrace.RESULT_REUSED,
                        ),
                    )
                }
                lock.withLock {
                    if (text != null) {
                        settleLocked(CommitOutcome.Final(text, windowModels.joinToString(", ").ifEmpty { model }))
                    } else {
                        failLocked(windowFailure ?: "the on-device engine failed", bufferLimitReached = false)
                    }
                }
                return
            }

            // Catching up on a backlog finalizes whole windows only: a tail
            // partial would be overtaken by the next window right away.
            val partial = if (behind) {
                streamer.catchUp(snapshot, snapshotSize, tx)
            } else {
                streamer.step(snapshot, snapshotSize, clock(), tx)
            }
            windowFailure?.let { reason ->
                lock.withLock { failLocked(reason, bufferLimitReached = false) }
                return
            }
            if (partial != null && partial != lastPartial) {
                lastPartial = partial
                trace?.partial(partial)
                events(StreamEvent.Partial(partial))
            }
            steppedSize -= trimFinalized()
            // Throttle: a step that transcribed nothing new must not spin. A
            // preempted preview gave way to work that is already waiting.
            if (partial == null && !behind && !preempted) sleepQuietly(STEP_BACKOFF_MILLIS)
        }
    }

    private fun behindLocked(): Boolean = base + size < captured

    /**
     * Polled by the engine while a preview of the audio up to absolute sample
     * [end] runs (on the worker thread, without the lock): true once required
     * work is waiting behind it, so Stop and window commits never wait for a
     * preview (#357, as the native server does since #428). That is Stop
     * with audio the preview does not cover, a full window ending at
     * [windowEnd] already captured, or a closed session. Stop with no newer
     * audio lets the preview finish: it is exactly the tail the flush
     * needs, and [ChunkStreamer.flush] reuses it.
     */
    private fun previewObsolete(end: Long, windowEnd: Long): Boolean {
        val captured = captured
        return closed || (inputEnded && captured > end) || captured >= windowEnd
    }

    /**
     * Reads saved audio the buffer is missing back into it, in order, up to
     * the buffer bound (never below one streamer window, so catching up
     * always progresses). The read happens outside the lock: while the
     * buffer is behind, the capture thread only counts new samples and the
     * buffer's end is the worker's alone. False when the read failed (the
     * stream is then failed and falls back to the batch path).
     */
    private fun refillFromBacklog(): Boolean {
        val from: Long
        val count: Int
        val source: Backlog
        lock.withLock {
            if (!behindLocked()) return true
            // Defensive: behind implies an open backlog (the spill that made
            // the buffer behind opened one), so this is unreachable today —
            // but returning true would busy-spin the loop below, whose idle
            // wait never ends while behind. Fail the stream instead.
            source = openBacklog ?: run {
                failLocked("the saved recording could not be read back", bufferLimitReached = false)
                return false
            }
            from = base + size
            count = minOf(captured - from, (refillLimit() - size).toLong()).toInt()
        }
        if (count <= 0) return true
        if (refillScratch.size < count) refillScratch = FloatArray(count)
        val read = runCatching { source.read(from, refillScratch, 0, count) }.getOrDefault(-1)
        lock.withLock {
            if (read != count) {
                failLocked("the saved recording could not be read back", bufferLimitReached = false)
                return false
            }
            if (size + count > buffer.size) buffer = buffer.copyOf(maxOf(size + count, buffer.size * 2))
            System.arraycopy(refillScratch, 0, buffer, size, count)
            size += count
        }
        return true
    }

    private fun refillLimit(): Int = maxOf(maxLiveSamples, streamer.windowSamples)

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
            base += dropped
            // Give back a peak-sized array once the live tail is small again.
            if (buffer.size > INITIAL_BUFFER_SAMPLES && buffer.size > size * 4) {
                buffer = buffer.copyOf(maxOf(INITIAL_BUFFER_SAMPLES, size * 2))
            }
        }
        streamer.rebase(dropped)
        return dropped
    }

    private fun sleepQuietly(millis: Long) {
        lock.withLock {
            if (!closed && !inputEnded && failure == null && !behindLocked()) {
                changed.await(millis, TimeUnit.MILLISECONDS)
            }
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
        trace?.settled((result as? CommitOutcome.Final)?.text, (result as? CommitOutcome.Fallback)?.reason)
        settled.countDown()
    }

    companion object {
        /** Live (unfinalized) audio bound, like the server's 60 s stream buffer cap. */
        const val MAX_LIVE_SECONDS = 60
        private const val INITIAL_BUFFER_SAMPLES = ChunkStreamer.SAMPLE_RATE * 4
        private const val FINISH_TIMEOUT_MINUTES = 10L
        private const val IDLE_WAIT_MILLIS = 250L
        private const val STEP_BACKOFF_MILLIS = 100L
    }
}
