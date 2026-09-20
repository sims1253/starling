package dev.starling.mobile.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertNotNull
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.RandomAccessFile

class WavPcmTest {
    @Test
    fun `decodes mono pcm16 wav to normalized floats`() {
        val file = wavFile(
            sampleRate = 16_000,
            samples = shortArrayOf(0, 16384, -16384, 32767, -32768),
        )

        val decoded = WavPcm.decodeMonoFloat(file)

        assertNotNull(decoded)
        decoded!!
        assertEquals(16_000, decoded.sampleRate)
        assertEquals(5, decoded.samples.size)
        assertEquals(0f, decoded.samples[0])
        assertEquals(0.5f, decoded.samples[1], 1e-4f)
        assertEquals(-0.5f, decoded.samples[2], 1e-4f)
        assertEquals(32767f / 32768f, decoded.samples[3], 1e-4f)
        assertEquals(-1f, decoded.samples[4], 1e-4f)
    }

    @Test
    fun `rejects non-wav and truncated files`() {
        val notWav = fileWithBytes("not a wave file at all".toByteArray())
        assertNull(WavPcm.decodeMonoFloat(notWav))
        assertNull(WavPcm.decodeMonoFloat(fileWithBytes(ByteArray(6))))
    }

    @Test
    fun `rejects non-pcm formats`() {
        val file = wavFile(sampleRate = 16_000, samples = shortArrayOf(1, 2), audioFormat = 3)
        assertNull(WavPcm.decodeMonoFloat(file))
    }

    private fun wavFile(sampleRate: Int, samples: ShortArray, audioFormat: Int = 1): File {
        val data = ByteArray(samples.size * 2)
        samples.forEachIndexed { index, value ->
            val bits = value.toInt() and 0xffff
            data[index * 2] = (bits and 0xff).toByte()
            data[index * 2 + 1] = (bits shr 8).toByte()
        }
        val out = ByteArrayOutputStream()
        out.write("RIFF".toByteArray())
        out.write(leInt(36 + data.size))
        out.write("WAVE".toByteArray())
        out.write("fmt ".toByteArray())
        out.write(leInt(16))
        out.write(leShort(audioFormat))
        out.write(leShort(1)) // mono
        out.write(leInt(sampleRate))
        out.write(leInt(sampleRate * 2))
        out.write(leShort(2))
        out.write(leShort(16))
        out.write("data".toByteArray())
        out.write(leInt(data.size))
        out.write(data)
        return fileWithBytes(out.toByteArray())
    }

    private fun leInt(value: Int): ByteArray = byteArrayOf(
        (value and 0xff).toByte(),
        (value shr 8 and 0xff).toByte(),
        (value shr 16 and 0xff).toByte(),
        (value shr 24 and 0xff).toByte(),
    )

    private fun leShort(value: Int): ByteArray = byteArrayOf(
        (value and 0xff).toByte(),
        (value shr 8 and 0xff).toByte(),
    )

    private fun fileWithBytes(bytes: ByteArray): File {
        val file = File.createTempFile("starling-wavtest-", ".wav")
        RandomAccessFile(file, "rw").use { it.write(bytes) }
        file.deleteOnExit()
        return file
    }
}
