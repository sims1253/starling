package dev.starling.mobile.audio

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.Handler
import android.os.Looper
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit

sealed interface CaptureResult {
    /**
     * The capture produced a complete, finalized WAV. [cappedAtLimit] is true
     * when it ended at the maximum recording duration instead of an explicit
     * stop: the audio is valid up to the cap and must be committed through
     * the normal completed path so it stays transcribable.
     */
    data class Completed(
        val durationSeconds: Double,
        val cappedAtLimit: Boolean = false,
    ) : CaptureResult
    data class Failed(val message: String) : CaptureResult
    data object AlreadyStopped : CaptureResult
}

/**
 * Observer for the PCM chunks of a live capture. Invoked on the capture
 * worker thread, after the chunk has been written to the WAV: the WAV write
 * stays the source of truth and this listener only observes it. The [bytes]
 * buffer is the capture loop's scratch buffer — it is only valid during the
 * call, so an implementation that retains the audio must copy it. A throwing
 * listener can never fail or interrupt the recording; its exception is
 * swallowed on the spot.
 */
fun interface AudioChunkListener {
    fun onChunk(bytes: ByteArray, count: Int)
}

/**
 * Owns one microphone capture at a time. The worker always closes the WAV
 * writer, including when Android tears down the audio device or the service.
 * Mic access is attributed through the context passed to [start]: callers
 * pass their own context, or one created for the recognition client they
 * act for. An optional [AudioChunkListener] observes the same PCM chunks the
 * worker writes to the WAV; it exists for live streaming and cannot affect
 * the capture, its stop choreography, or the two-hour cap.
 */
class AudioCapture {
    private val lock = Any()

    @Volatile
    private var stopRequested = false
    private var recorder: AudioRecord? = null
    private var worker: Thread? = null
    private var writer: WavWriter? = null
    private var workerError: String? = null
    private var cappedAtLimit = false
    private var state = State.IDLE

    // Guarded by [lock]: completion callbacks of stop() calls that are
    // still waiting for the worker, and whether a background task for the
    // forced-release phase is already in flight so concurrent stops share
    // it instead of stacking one task per call.
    private val stopCallbacks = mutableListOf<(CaptureResult) -> Unit>()
    private var escalationArmed = false
    private var stopExecutor: ExecutorService? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    fun isRecording(): Boolean = synchronized(lock) { state != State.IDLE }

    @SuppressLint("MissingPermission")
    fun start(
        context: Context,
        outputFile: File,
        onChunk: AudioChunkListener? = null,
    ): String? = synchronized(lock) {
        if (state != State.IDLE || worker?.isAlive == true) {
            return@synchronized "A recording is already stopping"
        }

        val minBuffer = AudioRecord.getMinBufferSize(
            WavWriter.SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        if (minBuffer <= 0) return@synchronized "This device does not support 16 kHz microphone capture"

        val bufferSize = maxOf(minBuffer * 2, WavWriter.SAMPLE_RATE / 2)
        val audioRecord = try {
            val builder = AudioRecord.Builder()
                .setAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION)
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setSampleRate(WavWriter.SAMPLE_RATE)
                        .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .build(),
                )
                .setBufferSizeInBytes(bufferSize)
            if (Build.VERSION.SDK_INT >= 31) {
                // Mic attribution via a context needs API 31; below it the
                // capture stays self-attributed as it always was.
                builder.setContext(context)
            }
            builder.build()
        } catch (_: IllegalArgumentException) {
            return@synchronized "Unable to initialize the microphone"
        } catch (_: SecurityException) {
            return@synchronized "Unable to initialize the microphone"
        }
        if (audioRecord.state != AudioRecord.STATE_INITIALIZED) {
            audioRecord.release()
            return@synchronized "Unable to initialize the microphone"
        }

        val wavWriter = try {
            WavWriter(outputFile)
        } catch (_: Exception) {
            audioRecord.release()
            return@synchronized "Unable to open the private recording file"
        }

        try {
            audioRecord.startRecording()
        } catch (_: IllegalStateException) {
            wavWriter.finish()
            audioRecord.release()
            return@synchronized "The microphone could not start"
        } catch (_: SecurityException) {
            wavWriter.finish()
            audioRecord.release()
            return@synchronized "The microphone could not start"
        }

        recorder = audioRecord
        writer = wavWriter
        workerError = null
        cappedAtLimit = false
        writerBytes = 0
        stopRequested = false
        state = State.RECORDING
        worker = Thread({ captureLoop(audioRecord, wavWriter, bufferSize, onChunk) }, "starling-audio-capture")
            .also { it.start() }
        null
    }

    /**
     * Stops the capture and delivers the outcome to [onSettled] exactly
     * once. The fast path stays synchronous and safe to call from
     * Activity/IME lifecycle teardown: the recorder is stopped and the
     * worker gets [CaptureStopPolicy.FAST_STOP_WAIT_MILLIS] to exit, which
     * healthy devices do in milliseconds. If the worker is still alive —
     * an AudioRecord.read wedged past the recorder stop — the forced
     * release phase continues on a background daemon executor instead of
     * blocking the caller for the rest of the grace window plus the forced
     * join (up to ~3.6 s). In that escalated case [onSettled] fires later,
     * once, on the main thread; otherwise it fires synchronously on the
     * calling thread before stop() returns. Every outcome, including
     * [CaptureResult.AlreadyStopped], reaches the callback, so no caller
     * can silently lose the result.
     *
     * The state machine is unchanged: the capture returns to IDLE only
     * after the worker has exited and closed the WAV writer, and a worker
     * that survives even the forced release keeps the capture in STOPPING
     * so a later start() cannot race the zombie or its writer, while a
     * later stop() can keep waiting for it.
     */
    fun stop(onSettled: (CaptureResult) -> Unit) {
        val pending = synchronized(lock) {
            if (state == State.IDLE) {
                null
            } else {
                stopRequested = true
                state = State.STOPPING
                stopCallbacks.add(onSettled)
                recorder to worker
            }
        } ?: run {
            onSettled(CaptureResult.AlreadyStopped)
            return
        }
        val (audioRecord, captureThread) = pending

        try {
            audioRecord?.stop()
        } catch (_: IllegalStateException) {
            // The worker still closes the writer in its finally block.
        }

        if (captureThread != null) {
            try {
                captureThread.join(CaptureStopPolicy.FAST_STOP_WAIT_MILLIS)
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
            }
        }

        val (error, bytes, capped) = workerOutcome()
        if (captureThread?.isAlive != true) {
            // The worker exited: settle inline on the calling thread,
            // exactly like the synchronous stop this replaces.
            finishSettlement(CaptureStopPolicy.settle(error, bytes, capped), fromEscalation = false)
            return
        }

        // The worker outlived the fast wait. Escalate the forced-release
        // phase to the background instead of blocking the caller.
        val executor = synchronized(lock) {
            if (escalationArmed) null else {
                escalationArmed = true
                stopExecutor()
            }
        }
        executor?.execute(::forcedRelease)
    }

    /**
     * The forced-release phase on the stop executor: what stop() used to
     * run inline after its long join. Gives the worker the rest of the
     * grace window, then releases the AudioRecord to unblock a wedged
     * blocking read, interrupts, and joins once more before settling.
     */
    private fun forcedRelease() {
        val (audioRecord, captureThread) = synchronized(lock) {
            // A task left queued after the capture settled (and possibly
            // relaunched through start()) must not touch whatever is
            // capturing now.
            if (state != State.STOPPING) return
            recorder to worker
        }
        if (captureThread == null || !captureThread.isAlive) {
            // The worker exited on its own after stop() armed this task,
            // so nobody else will settle the capture or deliver the
            // pending callbacks.
            val (error, bytes, capped) = workerOutcome()
            finishSettlement(CaptureStopPolicy.settle(error, bytes, capped), fromEscalation = true)
            return
        }

        try {
            captureThread.join(CaptureStopPolicy.ESCALATED_GRACE_MILLIS)
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        }

        if (captureThread.isAlive) {
            // Releasing AudioRecord unblocks a blocking read on affected
            // devices. The partial WAV remains available for explicit cleanup.
            try {
                audioRecord?.release()
            } catch (_: Exception) {
                // Already released by the worker.
            }
            captureThread.interrupt()
            try {
                captureThread.join(CaptureStopPolicy.FORCED_STOP_WAIT_MILLIS)
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
            }
        }

        val (error, bytes, capped) = workerOutcome()
        if (captureThread.isAlive) {
            // Keep STOPPING and the worker reference while a device refuses
            // to unblock. A subsequent stop can continue waiting, and start()
            // cannot race a zombie worker or its writer. The registered
            // callbacks still learn the outcome; only the state is retained.
            val callbacks = synchronized(lock) {
                state = State.STOPPING
                escalationArmed = false
                drainStopCallbacks()
            }
            deliver(callbacks, CaptureStopPolicy.ZOMBIE_RESULT, fromEscalation = true)
            return
        }
        finishSettlement(CaptureStopPolicy.settle(error, bytes, capped), fromEscalation = true)
    }

    /**
     * Returns the capture to IDLE after its worker has exited (and thereby
     * closed the WAV writer in its finally block), clears the device
     * references, and hands the result to every registered callback. A
     * delayed settler that lost the race to another stop's settlement (and
     * a possible relaunch through start()) must not clobber the new state;
     * the winner has already drained and delivered the callbacks.
     */
    private fun finishSettlement(result: CaptureResult, fromEscalation: Boolean) {
        val callbacks = synchronized(lock) {
            if (state != State.STOPPING) return
            recorder = null
            worker = null
            writer = null
            cappedAtLimit = false
            escalationArmed = false
            state = State.IDLE
            drainStopCallbacks()
        }
        deliver(callbacks, result, fromEscalation)
    }

    private fun drainStopCallbacks(): List<(CaptureResult) -> Unit> {
        val drained = stopCallbacks.toList()
        stopCallbacks.clear()
        return drained
    }

    /**
     * Escalated results are delivered on the main thread, where the
     * lifecycle callbacks that own the capture and its recording run; the
     * stop executor itself has no looper. Fast-path results were produced
     * on the calling thread and are delivered inline, before stop()
     * returns, like the synchronous stop this replaces.
     */
    private fun deliver(
        callbacks: List<(CaptureResult) -> Unit>,
        result: CaptureResult,
        fromEscalation: Boolean,
    ) {
        if (fromEscalation) {
            callbacks.forEach { mainHandler.post { it(result) } }
        } else {
            callbacks.forEach { it(result) }
        }
    }

    private fun workerOutcome(): Triple<String?, Long, Boolean> = synchronized(lock) {
        Triple(workerError, writerBytes, cappedAtLimit)
    }

    /**
     * Lazily created single-thread executor for the escalated stop phase.
     * Its daemon thread dies after [ESCALATION_EXECUTOR_IDLE_MILLIS]
     * without a task, so nothing stays alive between captures.
     */
    private fun stopExecutor(): ExecutorService {
        stopExecutor?.let { return it }
        return ThreadPoolExecutor(
            1,
            1,
            ESCALATION_EXECUTOR_IDLE_MILLIS,
            TimeUnit.MILLISECONDS,
            LinkedBlockingQueue(),
        ) { runnable ->
            Thread(runnable, "starling-audio-stop").apply { isDaemon = true }
        }.apply {
            allowCoreThreadTimeOut(true)
            stopExecutor = this
        }
    }

    private var writerBytes: Long = 0

    private fun captureLoop(
        audioRecord: AudioRecord,
        wavWriter: WavWriter,
        bufferSize: Int,
        onChunk: AudioChunkListener?,
    ) {
        val buffer = ByteArray(bufferSize)
        var bytesWritten = 0L
        try {
            while (!stopRequested) {
                val count = audioRecord.read(buffer, 0, buffer.size, AudioRecord.READ_BLOCKING)
                when {
                    count > 0 -> {
                        wavWriter.write(buffer, count)
                        bytesWritten += count
                        // Observer of the durable write above, never a gate on
                        // it: the recording must survive any streaming failure.
                        if (onChunk != null) {
                            runCatching { onChunk.onChunk(buffer, count) }
                        }
                    }
                    count == AudioRecord.ERROR_DEAD_OBJECT -> {
                        throw IllegalStateException("The microphone became unavailable")
                    }
                    count < 0 -> {
                        throw IllegalStateException("The microphone returned an audio read error")
                    }
                    else -> Thread.yield()
                }
                if (bytesWritten >= MAX_CAPTURE_BYTES) {
                    // End at the cap like an explicit stop: the WAV written so
                    // far is complete and valid, and stop() must return it on
                    // the completed path so it is committed and stays
                    // transcribable instead of being stranded as failed.
                    synchronized(lock) { cappedAtLimit = true }
                    break
                }
            }
        } catch (exception: Exception) {
            if (!stopRequested) {
                synchronized(lock) {
                    workerError = exception.message ?: "Microphone capture failed"
                }
            }
        } finally {
            synchronized(lock) { writerBytes = bytesWritten }
            try {
                wavWriter.finish()
            } catch (exception: Exception) {
                synchronized(lock) {
                    workerError = exception.message ?: "Unable to finalize the WAV recording"
                }
            }
            try {
                audioRecord.stop()
            } catch (_: IllegalStateException) {
                // It may already have been stopped by stop().
            }
            audioRecord.release()
        }
    }

    private enum class State { IDLE, RECORDING, STOPPING }

    companion object {
        /**
         * How long the single stop-executor thread may idle before it
         * exits. Escalated stops are rare, so between captures the
         * executor holds no live thread.
         */
        private val ESCALATION_EXECUTOR_IDLE_MILLIS = TimeUnit.SECONDS.toMillis(1)

        /**
         * Terminal cap for a single capture: two hours of 16 kHz mono PCM16
         * (~220 MiB). The WAV RIFF sizes would only overflow after ~18.6
         * hours, but this earlier bound also keeps the file under
         * InferenceClient's 256 MiB upload limit with margin. A capture that
         * reaches the cap ends like an explicit stop and is committed as a
         * completed recording.
         */
        private val MAX_CAPTURE_SECONDS = TimeUnit.HOURS.toSeconds(2)
        private val MAX_CAPTURE_BYTES =
            (WavWriter.SAMPLE_RATE * WavWriter.CHANNELS * WavWriter.BYTES_PER_SAMPLE).toLong() *
                MAX_CAPTURE_SECONDS
    }
}
