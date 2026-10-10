package dev.starling.mobile.audio

import dev.starling.mobile.storage.Durability
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.util.concurrent.ConcurrentHashMap

/**
 * Writes little-endian mono PCM16 at 16 kHz with a finalized RIFF header.
 *
 * Crash durability (#356): every [write] goes straight to the file (no
 * user-space buffer), so a killed process leaves every chunk the capture
 * wrote. [checkpoint], called periodically from another thread, fsyncs the
 * file and then records the confirmed payload size in the header, so after
 * a power loss or OS crash the header names the samples that were on
 * storage. The header write itself is synced by the next checkpoint, so
 * the recorded boundary trails the last write by about two intervals
 * when storage keeps up (a slow or failing fsync holds it back further: it
 * is always the last size that was actually synced). [finish] syncs the
 * payload before it writes the final header, then syncs that, so a
 * finished WAV is durable as a whole and an interrupted finish never
 * claims unsynced samples.
 *
 * While a writer is open its file is in [isOpen]: recovery
 * (RecordingStore) never repairs or moves a file a capture of this process
 * may still write, even a capture whose worker outlived its stop.
 */
internal class WavWriter(
    private val file: File,
    private val maxDataBytes: Long = MAX_DATA_BYTES,
) {
    private val output = RandomAccessFile(file, "rw")

    // Written by the capture thread only; read by [checkpoint] from another.
    @Volatile
    private var dataBytes = 0L

    // Serializes [checkpoint] with [finish]: both write the header, and a
    // checkpoint must never touch a closed descriptor.
    private val headerLock = Any()

    @Volatile
    private var closed = false

    init {
        try {
            output.setLength(0)
            // A valid header with an empty payload from the start: a capture
            // that dies before its first checkpoint is still recognizably this
            // app's WAV. The file and its directory entry are synced before any
            // audio, so a later checkpoint never confirms samples of a file
            // whose name a power loss could take away.
            output.write(header(0))
            output.fd.sync()
            file.parentFile?.let(Durability::syncDirectory)
        } catch (exception: Exception) {
            output.close()
            throw exception
        }
        open.add(file.absolutePath)
    }

    /** Whether [finish] has closed the file. */
    val isClosed: Boolean get() = closed

    /** Bytes of PCM written so far. */
    val writtenBytes: Long get() = dataBytes

    fun write(bytes: ByteArray, count: Int) {
        check(!closed) { "WAV writer is closed" }
        check(dataBytes + count <= maxDataBytes) { EXCEEDED_MESSAGE }
        output.write(bytes, 0, count)
        dataBytes += count
    }

    /**
     * Makes everything written so far durable and records it in the header
     * (a positional write, so it never moves the capture thread's offset).
     * Returns the confirmed payload size, or -1 once the writer is closed.
     * Safe to call from any thread while [write] runs.
     */
    fun checkpoint(): Long = synchronized(headerLock) {
        if (closed) return -1
        val confirmed = dataBytes
        output.fd.sync()
        output.channel.write(ByteBuffer.wrap(header(confirmed)), 0)
        confirmed
    }

    fun finish() = synchronized(headerLock) {
        if (closed) return
        check(dataBytes <= maxDataBytes) { EXCEEDED_MESSAGE }
        try {
            // A write the disk refused part-way (ENOSPC) may have left a torn
            // chunk past the counted payload; the WAV ends at the last whole one.
            if (output.length() != WAV_HEADER_SIZE + dataBytes) output.setLength(WAV_HEADER_SIZE + dataBytes)
            output.fd.sync()
            output.seek(0)
            output.write(header(dataBytes))
            output.fd.sync()
        } finally {
            closed = true
            try {
                output.close()
            } finally {
                open.remove(file.absolutePath)
            }
        }
    }

    companion object {
        const val SAMPLE_RATE = 16_000
        const val CHANNELS = 1
        const val BYTES_PER_SAMPLE = 2
        const val BITS_PER_SAMPLE = 16
        const val WAV_HEADER_SIZE = 44

        /**
         * Largest payload the 32-bit RIFF size fields can express without
         * overflow (~18.6 hours of audio at 32,000 B/s). The writer refuses
         * to grow past this bound instead of finalizing a corrupt header.
         */
        const val MAX_DATA_BYTES: Long = Int.MAX_VALUE.toLong() - (WAV_HEADER_SIZE - 8)
        private const val EXCEEDED_MESSAGE = "The recording exceeded the maximum WAV size"

        private val open: MutableSet<String> = ConcurrentHashMap.newKeySet()

        /** Whether a writer of this process still has [file] open. */
        fun isOpen(file: File): Boolean = file.absolutePath in open

        /** Whether any capture of this process is writing (audio upkeep waits for it). */
        fun anyOpen(): Boolean = open.isNotEmpty()

        /** The 44-byte PCM16 mono 16 kHz header for a payload of [dataBytes]. */
        fun header(dataBytes: Long): ByteArray {
            val header = ByteBuffer.allocate(WAV_HEADER_SIZE).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            header.put("RIFF".toByteArray(Charsets.US_ASCII))
            header.putInt((WAV_HEADER_SIZE - 8 + dataBytes).toInt())
            header.put("WAVE".toByteArray(Charsets.US_ASCII))
            header.put("fmt ".toByteArray(Charsets.US_ASCII))
            header.putInt(16)
            header.putShort(1) // PCM
            header.putShort(CHANNELS.toShort())
            header.putInt(SAMPLE_RATE)
            header.putInt(SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE)
            header.putShort((CHANNELS * BYTES_PER_SAMPLE).toShort())
            header.putShort(BITS_PER_SAMPLE.toShort())
            header.put("data".toByteArray(Charsets.US_ASCII))
            header.putInt(dataBytes.toInt())
            return header.array()
        }
    }
}
