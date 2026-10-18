package dev.starling.mobile

import android.app.Activity
import android.speech.RecognizerIntent
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
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
        )

        assertEquals(Activity.RESULT_OK, outcome.resultCode)
        assertEquals("  not like auth  ", outcome.transcript)
    }

    @Test
    fun blankTranscriptIsNoMatch() {
        val outcome = RecognizeSpeechOutcome.settled(
            recording(RecordingStatus.TRANSCRIBED, " "),
        )

        assertEquals(RecognizerIntent.RESULT_NO_MATCH, outcome.resultCode)
        assertNull(outcome.transcript)
    }

    @Test
    fun failureIsClientError() {
        val outcome = RecognizeSpeechOutcome.settled(recording(RecordingStatus.FAILED))

        assertEquals(RecognizerIntent.RESULT_CLIENT_ERROR, outcome.resultCode)
        assertNull(outcome.transcript)
    }

    @Test
    fun transcribedStatusWithoutTextIsAFailure() {
        val outcome = RecognizeSpeechOutcome.settled(recording(RecordingStatus.TRANSCRIBED))

        assertEquals(RecognizerIntent.RESULT_CLIENT_ERROR, outcome.resultCode)
    }

}
