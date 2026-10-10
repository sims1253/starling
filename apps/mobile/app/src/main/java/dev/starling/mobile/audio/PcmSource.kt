package dev.starling.mobile.audio

import android.media.AudioRecord
import java.io.File

/**
 * Where a capture reads its 16 kHz mono PCM16 from: the microphone, or in
 * debug builds a test WAV standing in for it ([debugSource]). The calls
 * follow [AudioRecord]'s: [read] blocks for audio and returns a byte count
 * or a negative AudioRecord error code; [stop] makes a blocked read return.
 */
internal interface PcmSource {
    fun start()
    fun read(buffer: ByteArray, count: Int): Int
    fun stop()
    fun release()

    companion object {
        /**
         * Debug builds only (StarlingApplication sets it): a stand-in for the
         * microphone, consulted at every capture start; null opens the mic.
         * Every entry point captures through [AudioCapture], so a device test
         * can drive the app, the keyboard and the recognizer with the same
         * audio without a speaker next to the phone.
         */
        @Volatile
        var debugSource: (() -> PcmSource?)? = null
    }
}

internal class MicrophoneSource(private val audioRecord: AudioRecord) : PcmSource {
    override fun start() = audioRecord.startRecording()

    override fun read(buffer: ByteArray, count: Int): Int =
        audioRecord.read(buffer, 0, count, AudioRecord.READ_BLOCKING)

    override fun stop() = audioRecord.stop()

    override fun release() = audioRecord.release()
}

/**
 * Plays a PCM16 mono 16 kHz WAV in a loop at real-time pace, as if it were
 * spoken into the microphone: [read] returns audio no faster than the wall
 * clock allows, so captures, live streaming and their timing behave as
 * with a real take. Test hook only (see [PcmSource.debugSource]).
 */
internal class LoopingWavSource(
    private val pcm: ByteArray,
    private val nanoTime: () -> Long = System::nanoTime,
    private val sleep: (Long) -> Unit = Thread::sleep,
) : PcmSource {
    @Volatile
    private var stopped = false
    private var startNanos = 0L
    private var emitted = 0L

    init {
        require(pcm.size >= WavWriter.BYTES_PER_SAMPLE) { "The test WAV has no audio" }
    }

    override fun start() {
        startNanos = nanoTime()
    }

    override fun read(buffer: ByteArray, count: Int): Int {
        val wanted = count - count % WavWriter.BYTES_PER_SAMPLE
        if (wanted <= 0) return 0
        // Due once the wall clock has reached the end of this chunk.
        val dueNanos = startNanos + (emitted + wanted) * NANOS_PER_SECOND / BYTES_PER_SECOND
        while (!stopped) {
            val waitNanos = dueNanos - nanoTime()
            if (waitNanos <= 0) break
            sleep(minOf(waitNanos / 1_000_000 + 1, MAX_SLEEP_MILLIS))
        }
        if (stopped) return 0
        for (i in 0 until wanted) {
            buffer[i] = pcm[((emitted + i) % pcm.size).toInt()]
        }
        emitted += wanted
        return wanted
    }

    override fun stop() {
        stopped = true
    }

    override fun release() {
        stopped = true
    }

    companion object {
        private const val NANOS_PER_SECOND = 1_000_000_000L
        private const val BYTES_PER_SECOND = (WavWriter.SAMPLE_RATE * WavWriter.BYTES_PER_SAMPLE).toLong()
        private const val MAX_SLEEP_MILLIS = 50L

        /** The PCM payload of a 16 kHz mono PCM16 WAV, or null when [file] is not one. */
        fun fromWav(file: File): LoopingWavSource? {
            val decoded = dev.starling.mobile.engine.WavPcm.decodePcm16(file) ?: return null
            if (decoded.sampleRate != WavWriter.SAMPLE_RATE || decoded.pcm.size < WavWriter.BYTES_PER_SAMPLE) return null
            return LoopingWavSource(decoded.pcm)
        }
    }
}
