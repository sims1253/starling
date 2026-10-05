package dev.starling.mobile.engine

import dev.starling.mobile.audio.WavWriter
import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.file.Files

class SavedAudioBacklogTest {
    @Test
    fun readsSamplesTheWriterHasWrittenWhileItIsStillOpen() {
        val file = Files.createTempFile("starling-backlog", ".wav.part").toFile()
        try {
            val writer = WavWriter(file)
            // Samples 0..9 as PCM16 values 0, 1000, 2000, ...
            val pcm = ByteArray(20) { i -> val v = (i / 2) * 1000; if (i % 2 == 0) v.toByte() else (v shr 8).toByte() }
            writer.write(pcm, pcm.size)

            SavedAudioBacklog(file).use { backlog ->
                val into = FloatArray(6)
                assertEquals(4, backlog.read(3, into, 1, 4))
                assertEquals(listOf(3000, 4000, 5000, 6000), (1..4).map { Math.round(into[it] * 32768f) })
                // Asking past what was written returns only what exists.
                assertEquals(2, backlog.read(8, into, 0, 5))
            }
            writer.finish()
        } finally {
            file.delete()
        }
    }
}
