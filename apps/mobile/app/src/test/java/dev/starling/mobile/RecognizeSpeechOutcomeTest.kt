package dev.starling.mobile

import android.app.Activity
import android.speech.RecognizerIntent
import dev.starling.mobile.audio.CaptureResult
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
import dev.starling.mobile.network.TranscriptionEngine
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class RecognizeSpeechOutcomeTest {
    private fun recording(status: RecordingStatus, transcript: String? = null) = Recording(
        id = "recording-1",
        createdAtMillis = 0L,
        wavName = "recording-1.wav",
        status = status,
        rawTranscript = transcript,
    )

    @Test
    fun transcribedTextIsReturnedVerbatim() {
        val outcome = RecognizeSpeechOutcome.settled(
            recording(RecordingStatus.TRANSCRIBED, "  not like auth  "),
            TranscriptionEngine.REMOTE,
        )

        assertEquals(Activity.RESULT_OK, outcome.resultCode)
        assertEquals("  not like auth  ", outcome.transcript)
    }

    @Test
    fun blankTranscriptIsNoMatch() {
        val outcome = RecognizeSpeechOutcome.settled(
            recording(RecordingStatus.TRANSCRIBED, " "),
            TranscriptionEngine.REMOTE,
        )

        assertEquals(RecognizerIntent.RESULT_NO_MATCH, outcome.resultCode)
        assertNull(outcome.transcript)
    }

    @Test
    fun serverFailureIsNetworkError() {
        val outcome = RecognizeSpeechOutcome.settled(recording(RecordingStatus.FAILED), TranscriptionEngine.REMOTE)

        assertEquals(RecognizerIntent.RESULT_NETWORK_ERROR, outcome.resultCode)
        assertNull(outcome.transcript)
    }

    @Test
    fun onDeviceFailureIsClientError() {
        val outcome = RecognizeSpeechOutcome.settled(recording(RecordingStatus.FAILED), TranscriptionEngine.ON_DEVICE)

        assertEquals(RecognizerIntent.RESULT_CLIENT_ERROR, outcome.resultCode)
    }

    @Test
    fun transcribedStatusWithoutTextIsAFailure() {
        val outcome = RecognizeSpeechOutcome.settled(recording(RecordingStatus.TRANSCRIBED), TranscriptionEngine.REMOTE)

        assertEquals(RecognizerIntent.RESULT_NETWORK_ERROR, outcome.resultCode)
    }

    @Test
    fun onlyACompletedCaptureProceedsToTranscription() {
        assertNull(RecognizeSpeechOutcome.captureEnded(CaptureResult.Completed(1.0)))
        assertEquals(
            RecognizerIntent.RESULT_AUDIO_ERROR,
            RecognizeSpeechOutcome.captureEnded(CaptureResult.Failed("mic gone"))?.resultCode,
        )
        assertEquals(
            RecognizerIntent.RESULT_AUDIO_ERROR,
            RecognizeSpeechOutcome.captureEnded(CaptureResult.AlreadyStopped)?.resultCode,
        )
    }
}
