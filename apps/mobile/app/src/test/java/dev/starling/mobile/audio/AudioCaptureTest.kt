package dev.starling.mobile.audio

import java.util.concurrent.TimeUnit
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AudioCaptureTest {
    @Test
    fun captureCapIsTwoHoursAndStaysInsideTheWavHeaderRange() {
        val twoHoursOfPcm16 = TimeUnit.HOURS.toSeconds(2) *
            WavWriter.SAMPLE_RATE *
            WavWriter.BYTES_PER_SAMPLE
        assertEquals(twoHoursOfPcm16, AudioCapture.MAX_CAPTURE_BYTES)
        // WavWriter.finish() writes the RIFF sizes as 32-bit values; the cap
        // must stop a capture far below the point where they would overflow
        // (~18 hours at this rate) and corrupt the finalized header.
        assertTrue(AudioCapture.MAX_CAPTURE_BYTES < Int.MAX_VALUE.toLong())
    }
}
