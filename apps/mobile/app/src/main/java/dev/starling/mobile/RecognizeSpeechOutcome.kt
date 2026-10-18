package dev.starling.mobile

import android.app.Activity
import android.speech.RecognizerIntent
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus

/**
 * Maps [RecognizeSpeechActivity] outcomes onto the `ACTION_RECOGNIZE_SPEECH`
 * result contract: `RESULT_OK` with the verbatim transcript as the single
 * `EXTRA_RESULTS` entry, or one of the RecognizerIntent `RESULT_*` error
 * codes. Pure Kotlin like the other guard classes, so the mapping is
 * unit-testable on the JVM.
 */
object RecognizeSpeechOutcome {
    data class Outcome(val resultCode: Int, val transcript: String? = null)

    /** A settled transcription. Failures report the class the caller can act on. */
    fun settled(completed: Recording): Outcome {
        val text = completed.rawTranscript
        return when {
            completed.status == RecordingStatus.TRANSCRIBED && text != null ->
                if (text.isBlank()) Outcome(RecognizerIntent.RESULT_NO_MATCH) else Outcome(Activity.RESULT_OK, text)
            // A local engine problem (missing model, load failure) is the
            // client's; the recording stays retryable in Starling Mobile.
            else -> Outcome(RecognizerIntent.RESULT_CLIENT_ERROR)
        }
    }

}
