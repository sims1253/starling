package dev.starling.mobile

import android.speech.SpeechRecognizer
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
        assertNull(guard.stopListening(owner))
    }

    @Test
    fun cancelKeepsAudioAndSkipsTranscription() {
        val guard = RecognitionSessionGuard<Any>()
        guard.begin(recording(), owner)
        val ending = guard.cancel(owner) as RecognitionSessionGuard.Ending.Abandon
        assertSame(owner, ending.session.owner)
        assertEquals("recording-1", ending.session.recording.id)
        assertNull(guard.cancel(owner))
    }

    @Test
    fun watchdogTimeoutFinalizesTheLiveSession() {
        val guard = RecognitionSessionGuard<Any>()
        guard.begin(recording(), owner)
        val ending = guard.expire() as RecognitionSessionGuard.Ending.Finalize
        assertSame(owner, ending.session.owner)
        assertNull(guard.stopListening(owner))
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

        assertSame(owner, ending.session.owner)
        assertNull(guard.teardown())
    }

    @Test
    fun deliveryToADeadClientIsSwallowed() {
        val guard = RecognitionSessionGuard<(Int) -> Unit>()
        val deadClient: (Int) -> Unit = { throw RuntimeException("keyboard process died") }
        guard.begin(recording(), deadClient)
        val ending = guard.stopListening(deadClient) as RecognitionSessionGuard.Ending.Finalize
        // A binder call to a dead keyboard cannot crash the recorder.
        guard.deliver(ending.session.owner) { it(SpeechRecognizer.ERROR_CLIENT) }
    }
}
