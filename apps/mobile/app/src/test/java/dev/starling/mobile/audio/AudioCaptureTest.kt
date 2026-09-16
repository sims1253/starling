package dev.starling.mobile.audio

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AudioCaptureTest {
    @Test
    fun captureCapIsTwoHoursAndStaysInsideTheWavHeaderRange() {
        // Pinned as a literal so an accidental change to the rate or duration
        // constants cannot silently redefine the expected cap with it.
        assertEquals(230_400_000L, AudioCapture.MAX_CAPTURE_BYTES)
        // WavWriter.finish() writes the RIFF sizes as 32-bit values; the cap
        // must stop a capture far below the point where they would overflow
        // (~18 hours at this rate) and corrupt the finalized header. It must
        // also stay under the server upload ceiling (256 MB).
        assertTrue(AudioCapture.MAX_CAPTURE_BYTES < Int.MAX_VALUE.toLong())
        assertTrue(AudioCapture.MAX_CAPTURE_BYTES < 256L * 1024 * 1024)
    }
}
