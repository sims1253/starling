package dev.starling.mobile.audio

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import java.io.File
import java.util.concurrent.TimeUnit

sealed interface CaptureResult {
    data class Completed(val durationSeconds: Double) : CaptureResult
    data class Failed(val message: String) : CaptureResult
    data object AlreadyStopped : CaptureResult
}

/**
 * Owns one microphone capture at a time. The worker always closes the WAV
 * writer, including when Android tears down the audio device or the service.
 * Mic access is attributed through the context passed to [start]: callers
 * pass their own context, or one created for the recognition client they
 * act for.
 */
class AudioCapture {
    private val lock = Any()

    @Volatile
    private var stopRequested = false
    private var recorder: AudioRecord? = null
    private var worker: Thread? = null
    private var writer: WavWriter? = null
    private var workerError: String? = null
    private var state = State.IDLE

    fun isRecording(): Boolean = synchronized(lock) { state != State.IDLE }

    @SuppressLint("MissingPermission")
    fun start(context: Context, outputFile: File): String? = synchronized(lock) {
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
        writerBytes = 0
        stopRequested = false
        state = State.RECORDING
        worker = Thread({ captureLoop(audioRecord, wavWriter, bufferSize) }, "starling-audio-capture")
            .also { it.start() }
        null
    }

    /**
     * Stops and closes the capture synchronously. This is intentionally safe
     * to call from Activity/IME lifecycle teardown so no open microphone or
     * incomplete WAV writer is left behind.
     */
    fun stop(): CaptureResult {
        val (audioRecord, captureThread) = synchronized(lock) {
            if (state == State.IDLE) return CaptureResult.AlreadyStopped
            stopRequested = true
            state = State.STOPPING
            recorder to worker
        }

        try {
            audioRecord?.stop()
        } catch (_: IllegalStateException) {
            // The worker still closes the writer in its finally block.
        }

        if (captureThread != null) {
            try {
                captureThread.join(STOP_WAIT_MILLIS)
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
            }
        }
        if (captureThread?.isAlive == true) {
            // Releasing AudioRecord unblocks a blocking read on affected
            // devices. The partial WAV remains available for explicit cleanup.
            try {
                audioRecord?.release()
            } catch (_: Exception) {
                // Already released by the worker.
            }
            captureThread.interrupt()
            try {
                captureThread.join(FORCED_STOP_WAIT_MILLIS)
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
            }
        }

        if (captureThread?.isAlive == true) {
            // Keep STOPPING and the worker reference while a device refuses to
            // unblock. A subsequent stop can continue waiting, and start()
            // cannot race a zombie worker or its writer.
            synchronized(lock) { state = State.STOPPING }
            return CaptureResult.Failed("The microphone did not stop cleanly; the partial recording was kept")
        }
        val (error, bytes) = synchronized(lock) {
            val capturedError = workerError
            val capturedBytes = writerBytes
            recorder = null
            worker = null
            writer = null
            state = State.IDLE
            capturedError to capturedBytes
        }
        if (error != null) return CaptureResult.Failed(error)

        return CaptureResult.Completed(bytes.toDouble() / (WavWriter.SAMPLE_RATE * WavWriter.BYTES_PER_SAMPLE))
    }

    private var writerBytes: Long = 0

    private fun captureLoop(audioRecord: AudioRecord, wavWriter: WavWriter, bufferSize: Int) {
        val buffer = ByteArray(bufferSize)
        var bytesWritten = 0L
        try {
            while (!stopRequested) {
                val count = audioRecord.read(buffer, 0, buffer.size, AudioRecord.READ_BLOCKING)
                when {
                    count > 0 -> {
                        wavWriter.write(buffer, count)
                        bytesWritten += count
                        if (bytesWritten >= MAX_CAPTURE_BYTES) {
                            throw IllegalStateException(CAP_REACHED_MESSAGE)
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
        private val STOP_WAIT_MILLIS = TimeUnit.SECONDS.toMillis(3)
        private val FORCED_STOP_WAIT_MILLIS = TimeUnit.SECONDS.toMillis(1)

        /**
         * Terminal cap for a single capture. The WAV header only breaks down
         * after ~18.2 hours, but the file is kept and the user gets a clear
         * message only when the stop is deliberate and early: two hours of
         * 16 kHz mono PCM16 (~220 MiB) stays far inside the RIFF size limit.
         */
        private val MAX_CAPTURE_SECONDS = TimeUnit.HOURS.toSeconds(2)
        private val MAX_CAPTURE_BYTES =
            (WavWriter.SAMPLE_RATE * WavWriter.CHANNELS * WavWriter.BYTES_PER_SAMPLE).toLong() *
                MAX_CAPTURE_SECONDS
        private const val CAP_REACHED_MESSAGE =
            "The recording reached the 2-hour limit and was stopped"
    }
}
