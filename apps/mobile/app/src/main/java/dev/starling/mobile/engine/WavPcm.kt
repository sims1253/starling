package dev.starling.mobile.engine

import java.io.File
import java.io.RandomAccessFile

/** Decodes the app's own finalized recordings (RIFF/WAVE PCM16) to float32. */
object WavPcm {
    data class Decoded(val sampleRate: Int, val samples: FloatArray)

    /**
     * Mono float32 samples in [-1, 1] from a PCM16 WAV file, or null when the
     * file is not a decodable PCM16 WAVE. The app writes 16 kHz mono; the
     * sample rate is returned so the caller can reject anything else.
     */
    fun decodeMonoFloat(file: File): Decoded? = runCatching {
        RandomAccessFile(file, "r").use { input ->
            val length = input.length()
            if (length < HEADER_BYTES) return null
            val riff = ByteArray(HEADER_BYTES.toInt())
            input.readFully(riff)
            if (!riff.copyOfRange(0, 4).contentEquals(RIFF) ||
                !riff.copyOfRange(8, 12).contentEquals(WAVE)
            ) {
                return null
            }

            var sampleRate = 0
            var channels = 0
            var bitsPerSample = 0
            var pcm16 = false
            var data: Pair<Long, Long>? = null
            var offset = 12L
            while (offset + 8 <= length) {
                input.seek(offset)
                val id = ByteArray(4)
                input.readFully(id)
                val size = readLeInt(input)
                val body = offset + 8
                when (String(id, Charsets.US_ASCII)) {
                    "fmt " -> {
                        if (size < 16) return null
                        val fmt = ByteArray(16)
                        input.readFully(fmt)
                        pcm16 = leShort(fmt, 0) == 1
                        channels = leShort(fmt, 2)
                        sampleRate = leInt(fmt, 4)
                        bitsPerSample = leShort(fmt, 14)
                    }
                    "data" -> data = body to minOf(size.toLong(), length - body)
                }
                offset = body + size + (size and 1)
            }
            val dataRange = data ?: return null
            if (!pcm16 || channels != 1 || bitsPerSample != 16 || dataRange.second < 2) {
                return null
            }

            val pcm = ByteArray(dataRange.second.toInt())
            input.seek(dataRange.first)
            input.readFully(pcm)
            val samples = FloatArray(pcm.size / 2) { index ->
                val low = pcm[index * 2].toInt() and 0xff
                val high = pcm[index * 2 + 1].toInt() and 0xff
                ((high shl 8) or low).toShort() / 32768f
            }
            Decoded(sampleRate, samples)
        }
    }.getOrNull()

    private fun readLeInt(input: RandomAccessFile): Int {
        val bytes = ByteArray(4)
        input.readFully(bytes)
        return leInt(bytes, 0)
    }

    private fun leShort(bytes: ByteArray, offset: Int): Int =
        (bytes[offset].toInt() and 0xff) or ((bytes[offset + 1].toInt() and 0xff) shl 8)

    private fun leInt(bytes: ByteArray, offset: Int): Int =
        (bytes[offset].toInt() and 0xff) or
            ((bytes[offset + 1].toInt() and 0xff) shl 8) or
            ((bytes[offset + 2].toInt() and 0xff) shl 16) or
            ((bytes[offset + 3].toInt() and 0xff) shl 24)

    private val RIFF = "RIFF".toByteArray(Charsets.US_ASCII)
    private val WAVE = "WAVE".toByteArray(Charsets.US_ASCII)
    private const val HEADER_BYTES = 12L
}
