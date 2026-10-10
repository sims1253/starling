package dev.starling.mobile.audio

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class LoopingWavSourceTest {
    @get:Rule
    val folder = TemporaryFolder()

    @Test
    fun loopsTheWavAtRealTimePace() {
        var now = 0L
        val slept = mutableListOf<Long>()
        val source = LoopingWavSource(
            pcm = byteArrayOf(1, 2, 3, 4, 5, 6),
            nanoTime = { now },
            sleep = { millis -> slept += millis; now += millis * 1_000_000 },
        )
        source.start()

        // 3200 bytes are 100 ms of 16 kHz PCM16: due 100 ms after start.
        val buffer = ByteArray(3_200)
        assertEquals(3_200, source.read(buffer, buffer.size))
        assertTrue(now >= 100_000_000)
        assertTrue(slept.isNotEmpty())
        assertArrayEquals(byteArrayOf(1, 2, 3, 4, 5, 6, 1, 2), buffer.copyOfRange(0, 8))
        // The loop continues where the previous read stopped (3200 % 6 = 2).
        val next = ByteArray(4)
        assertEquals(4, source.read(next, next.size))
        assertArrayEquals(byteArrayOf(3, 4, 5, 6), next)
    }

    @Test
    fun anOddCountReadsWholeSamplesAndStopEndsReads() {
        var now = 0L
        val source = LoopingWavSource(byteArrayOf(1, 2), nanoTime = { now }, sleep = { now += it * 1_000_000 })
        source.start()
        val buffer = ByteArray(5)
        assertEquals(4, source.read(buffer, 5))
        source.stop()
        assertEquals(0, source.read(buffer, 4))
    }

    @Test
    fun onlyA16kMonoPcm16WavIsAccepted() {
        val good = File(folder.root, "good.wav").apply { writeBytes(WavWriter.header(4) + byteArrayOf(1, 0, 2, 0)) }
        val empty = File(folder.root, "empty.wav").apply { writeBytes(WavWriter.header(0)) }
        val notWav = File(folder.root, "noise.bin").apply { writeBytes(ByteArray(100)) }

        val source = LoopingWavSource.fromWav(good)!!
        source.start()
        assertNull(LoopingWavSource.fromWav(empty))
        assertNull(LoopingWavSource.fromWav(notWav))
    }
}
