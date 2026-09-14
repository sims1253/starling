package dev.starling.mobile.audio

import java.io.File
import java.nio.file.Files
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class WavWriterTest {
    @Test
    fun writesPcm16MonoHeaderAndPayload() {
        val directory = Files.createTempDirectory("starling-wav-test").toFile()
        val file = File(directory, "sample.wav.part")
        try {
            val payload = byteArrayOf(0, 0, 1, 0, -1, -1, 2, 0)
            WavWriter(file).useAndFinish(payload)
            val bytes = file.readBytes()

            assertEquals(44 + payload.size, bytes.size)
            assertEquals("RIFF", bytes.copyOfRange(0, 4).toString(Charsets.US_ASCII))
            assertEquals("WAVE", bytes.copyOfRange(8, 12).toString(Charsets.US_ASCII))
            assertEquals("fmt ", bytes.copyOfRange(12, 16).toString(Charsets.US_ASCII))
            assertEquals(1, littleEndianShort(bytes, 20))
            assertEquals(1, littleEndianShort(bytes, 22))
            assertEquals(16_000, littleEndianInt(bytes, 24))
            assertEquals("data", bytes.copyOfRange(36, 40).toString(Charsets.US_ASCII))
            assertEquals(payload.size, littleEndianInt(bytes, 40))
            assertArrayEquals(payload, bytes.copyOfRange(44, bytes.size))
        } finally {
            file.delete()
            directory.delete()
        }
    }

    private fun WavWriter.useAndFinish(payload: ByteArray) {
        write(payload, payload.size)
        finish()
    }

    private fun littleEndianShort(bytes: ByteArray, offset: Int): Int =
        (bytes[offset].toInt() and 0xff) or ((bytes[offset + 1].toInt() and 0xff) shl 8)

    private fun littleEndianInt(bytes: ByteArray, offset: Int): Int =
        (bytes[offset].toInt() and 0xff) or
            ((bytes[offset + 1].toInt() and 0xff) shl 8) or
            ((bytes[offset + 2].toInt() and 0xff) shl 16) or
            ((bytes[offset + 3].toInt() and 0xff) shl 24)
}
