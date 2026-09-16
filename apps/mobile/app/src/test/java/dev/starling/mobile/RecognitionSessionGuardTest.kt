package dev.starling.mobile

import android.speech.SpeechRecognizer
import dev.starling.mobile.audio.CaptureResult
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class RecognitionSessionGuardTest {
    private val owner = Any()

    private fun recording() = Recording(
        id = "recording-1",
        createdAtMillis = 0L,
        wavName = "recording-1.wav",
        status = RecordingStatus.RECORDING,
    )

    @Test
    fun stopFromOwnerFinalizesForTranscription() {
        val guard = RecognitionSessionGuard<Any>()
        guard.begin(recording(), owner)

        val ending = guard.stopListening(owner) as RecognitionSessionGuard.Ending.Finalize

        assertSame(owner, ending.session.owner)
        assertEquals("recording-1", ending.session.recording.id)
        val settlement = guard.settle(
            ending,
            CaptureResult.Completed(2.5),
        ) as RecognitionSessionGuard.Settlement.Transcribe
        assertEquals(2.5, settlement.durationSeconds, 0.0)
    }

    @Test
    fun cancelKeepsAudioAndSkipsUpload() {
        val guard = RecognitionSessionGuard<Any>()
        guard.begin(recording(), owner)

        val ending = guard.cancel(owner) as RecognitionSessionGuard.Ending.Abandon

        val kept = guard.settle(
            ending,
            CaptureResult.Completed(3.0),
        ) as RecognitionSessionGuard.Settlement.Keep
        assertEquals(3.0, kept.durationSeconds, 0.0)
        val failed = guard.settle(
            ending,
            CaptureResult.Failed("The microphone did not stop cleanly"),
        ) as RecognitionSessionGuard.Settlement.Fail
        assertNull(failed.errorCode)
    }

    @Test
    fun watchdogTimeoutFinalizesTheLiveSession() {
        val guard = RecognitionSessionGuard<Any>()
        guard.begin(recording(), owner)

        val ending = guard.expire() as RecognitionSessionGuard.Ending.Finalize

        // The session is over; a late stop from the same client is a no-op.
        assertNull(guard.stopListening(owner))
    }

    @Test
    fun captureFailureMarksTheRecordingFailed() {
        val guard = RecognitionSessionGuard<Any>()
        guard.begin(recording(), owner)

        val ending = guard.stopListening(owner) as RecognitionSessionGuard.Ending.Finalize
        val settlement = guard.settle(
            ending,
            CaptureResult.Failed("The microphone did not stop cleanly"),
        ) as RecognitionSessionGuard.Settlement.Fail

        assertEquals("recording-1", settlement.recordingId)
        assertEquals(SpeechRecognizer.ERROR_CLIENT, settlement.errorCode)
    }

    @Test
    fun foreignClientCannotEndTheSession() {
        val guard = RecognitionSessionGuard<Any>()
        guard.begin(recording(), owner)

        assertNull(guard.stopListening(Any()))
        assertNull(guard.cancel(Any()))

        assertTrue(guard.cancel(owner) is RecognitionSessionGuard.Ending.Abandon)
    }

    @Test
    fun teardownKeepsTheAudioLikeCancel() {
        val guard = RecognitionSessionGuard<Any>()
        guard.begin(recording(), owner)

        val ending = guard.teardown() as RecognitionSessionGuard.Ending.Abandon

        assertTrue(
            guard.settle(ending, CaptureResult.Completed(1.0)) is RecognitionSessionGuard.Settlement.Keep,
        )
    }

    @Test
    fun deliveryToADeadClientIsSwallowed() {
        val guard = RecognitionSessionGuard<(Int) -> Unit>()
        val deadClient: (Int) -> Unit = { throw RuntimeException("keyboard process died") }
        guard.begin(recording(), deadClient)
        val ending = guard.stopListening(deadClient) as RecognitionSessionGuard.Ending.Finalize
        val settlement = guard.settle(
            ending,
            CaptureResult.Failed("The microphone became unavailable"),
        ) as RecognitionSessionGuard.Settlement.Fail

        // The binder call into the dead keyboard's process throws; the
        // delivery must be swallowed instead of crashing this process.
        settlement.errorCode?.let { code -> guard.deliver(deadClient) { it(code) } }
    }
}
