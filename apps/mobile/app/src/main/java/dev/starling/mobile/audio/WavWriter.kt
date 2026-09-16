package dev.starling.mobile.audio

import java.io.File
import java.io.RandomAccessFile

/** Writes little-endian mono PCM16 at 16 kHz with a finalized RIFF header. */
internal class WavWriter(
    file: File,
    private val maxDataBytes: Long = MAX_DATA_BYTES,
) {
    private val output = RandomAccessFile(file, "rw")
    private var dataBytes = 0L
    private var closed = false

    init {
        output.setLength(0)
        output.seek(WAV_HEADER_SIZE.toLong())
    }

    fun write(bytes: ByteArray, count: Int) {
        check(!closed) { "WAV writer is closed" }
        check(dataBytes + count <= maxDataBytes) { EXCEEDED_MESSAGE }
        output.write(bytes, 0, count)
        dataBytes += count
    }

    fun finish() {
        if (closed) return
        check(dataBytes <= maxDataBytes) { EXCEEDED_MESSAGE }
        try {
            output.seek(0)
            output.writeAscii("RIFF")
            output.writeIntLittleEndian((WAV_HEADER_SIZE - 8 + dataBytes).toInt())
            output.writeAscii("WAVE")
            output.writeAscii("fmt ")
            output.writeIntLittleEndian(16)
            output.writeShortLittleEndian(1) // PCM
            output.writeShortLittleEndian(1) // mono
            output.writeIntLittleEndian(SAMPLE_RATE)
            output.writeIntLittleEndian(SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE)
            output.writeShortLittleEndian(CHANNELS * BYTES_PER_SAMPLE)
            output.writeShortLittleEndian(BITS_PER_SAMPLE)
            output.writeAscii("data")
            output.writeIntLittleEndian(dataBytes.toInt())
        } finally {
            closed = true
            output.close()
        }
    }

    private fun RandomAccessFile.writeAscii(value: String) {
        write(value.toByteArray(Charsets.US_ASCII))
    }

    private fun RandomAccessFile.writeIntLittleEndian(value: Int) {
        write(value and 0xff)
        write(value ushr 8 and 0xff)
        write(value ushr 16 and 0xff)
        write(value ushr 24 and 0xff)
    }

    private fun RandomAccessFile.writeShortLittleEndian(value: Int) {
        write(value and 0xff)
        write(value ushr 8 and 0xff)
    }

    companion object {
        const val SAMPLE_RATE = 16_000
        const val CHANNELS = 1
        const val BYTES_PER_SAMPLE = 2
        const val BITS_PER_SAMPLE = 16
        const val WAV_HEADER_SIZE = 44

        /**
         * Largest payload the 32-bit RIFF size fields can express without
         * overflow (~18.2 hours of audio at 32 KB/s). The writer refuses to
         * grow past this bound instead of finalizing a corrupt header.
         */
        const val MAX_DATA_BYTES: Long = Int.MAX_VALUE.toLong() - (WAV_HEADER_SIZE - 8)
        private const val EXCEEDED_MESSAGE = "The recording exceeded the maximum WAV size"
    }
}
