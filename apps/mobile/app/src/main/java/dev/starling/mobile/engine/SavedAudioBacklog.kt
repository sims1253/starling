package dev.starling.mobile.engine

import dev.starling.mobile.audio.WavWriter
import java.io.File
import java.io.RandomAccessFile

/**
 * Reads a live session's backlog from the recording's WAV while the capture
 * is still writing it. [WavWriter] writes the PCM16 payload straight to the
 * file from offset [WavWriter.WAV_HEADER_SIZE] and only fills the header in
 * at the end, and the session asks only for samples the capture has already
 * written, so every byte read here is final. The descriptor is opened at
 * construction and stays valid when the finished WAV is renamed into place.
 */
internal class SavedAudioBacklog(file: File) : OnDeviceStreamSession.Backlog {
    private val input = RandomAccessFile(file, "r")
    private var bytes = ByteArray(0)

    override fun read(from: Long, into: FloatArray, offset: Int, count: Int): Int {
        require(from >= 0 && count >= 0 && offset >= 0 && offset + count <= into.size)
        val wanted = count * BYTES_PER_SAMPLE
        if (bytes.size < wanted) bytes = ByteArray(wanted)
        input.seek(WavWriter.WAV_HEADER_SIZE + from * BYTES_PER_SAMPLE)
        var filled = 0
        while (filled < wanted) {
            val n = input.read(bytes, filled, wanted - filled)
            if (n < 0) break
            filled += n
        }
        val samples = filled / BYTES_PER_SAMPLE
        for (i in 0 until samples) {
            val lo = bytes[2 * i].toInt() and 0xff
            val hi = bytes[2 * i + 1].toInt()
            into[offset + i] = ((hi shl 8) or lo).toShort() / 32768f
        }
        return samples
    }

    override fun close() = input.close()

    private companion object {
        const val BYTES_PER_SAMPLE = WavWriter.BYTES_PER_SAMPLE
    }
}
