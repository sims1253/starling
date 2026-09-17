package dev.starling.mobile.audio

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CaptureStopPolicyTest {
    @Test
    fun settleMapsWrittenBytesToTheExactDuration() {
        val oneSecond = CaptureStopPolicy.settle(error = null, bytes = 32_000, cappedAtLimit = false)

        val completed = oneSecond as CaptureResult.Completed
        assertEquals(1.0, completed.durationSeconds, 0.0)
        assertFalse(completed.cappedAtLimit)
    }

    @Test
    fun settleReportsPartialSecondsWithoutRounding() {
        val completed = CaptureStopPolicy.settle(error = null, bytes = 15_999, cappedAtLimit = false)
            as CaptureResult.Completed

        assertEquals(15_999.0 / 32_000, completed.durationSeconds, 0.0)
    }

    @Test
    fun settlePropagatesTheCapFlagLikeACompletion() {
        val capped = CaptureStopPolicy.settle(error = null, bytes = 1, cappedAtLimit = true)
            as CaptureResult.Completed

        assertTrue(capped.cappedAtLimit)
    }

    @Test
    fun settleSurfacesAWorkerErrorOverTheCapturedAudio() {
        val failed = CaptureStopPolicy.settle(
            error = "Unable to finalize the WAV recording",
            bytes = 32_000,
            cappedAtLimit = true,
        ) as CaptureResult.Failed

        assertEquals("Unable to finalize the WAV recording", failed.message)
    }

    @Test
    fun zombieResultKeepsThePartialRecordingMessage() {
        val zombie = CaptureStopPolicy.ZOMBIE_RESULT

        assertEquals(
            "The microphone did not stop cleanly; the partial recording was kept",
            zombie.message,
        )
    }

    @Test
    fun fastWaitStaysWithinTheBoundedCallerWindow() {
        // The issue bounds the synchronous caller block to 200-500 ms: long
        // enough for a healthy worker to exit after the recorder stop, short
        // enough to stay imperceptible.
        assertTrue(CaptureStopPolicy.FAST_STOP_WAIT_MILLIS in 200..500)
    }

    @Test
    fun escalationBudgetPreservesTheOriginalTotalPatience() {
        // The 3 s grace plus the 1 s forced join of the old synchronous stop
        // are preserved; only their thread moves, so a device that needs the
        // full budget still gets it.
        assertEquals(3_000, CaptureStopPolicy.GRACEFUL_STOP_MILLIS)
        assertEquals(1_000, CaptureStopPolicy.FORCED_STOP_WAIT_MILLIS)
        assertEquals(
            CaptureStopPolicy.GRACEFUL_STOP_MILLIS - CaptureStopPolicy.FAST_STOP_WAIT_MILLIS,
            CaptureStopPolicy.ESCALATED_GRACE_MILLIS,
        )
        assertTrue(CaptureStopPolicy.ESCALATED_GRACE_MILLIS > 0)
    }
}
