package dev.starling.mobile

import android.app.Activity
import android.speech.RecognizerIntent
import dev.starling.mobile.audio.CaptureResult
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
import dev.starling.mobile.network.TranscriptionEngine

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
    fun settled(completed: Recording, engine: TranscriptionEngine): Outcome {
        val text = completed.rawTranscript
        return when {
            completed.status == RecordingStatus.TRANSCRIBED && text != null ->
                if (text.isBlank()) Outcome(RecognizerIntent.RESULT_NO_MATCH) else Outcome(Activity.RESULT_OK, text)
            // A local engine problem (missing model, load failure) is the
            // client's; everything on the server path is reported as network.
            engine == TranscriptionEngine.ON_DEVICE -> Outcome(RecognizerIntent.RESULT_CLIENT_ERROR)
            else -> Outcome(RecognizerIntent.RESULT_NETWORK_ERROR)
        }
    }

    /** A capture that ended without a finalized recording to transcribe. */
    fun captureEnded(result: CaptureResult): Outcome? = when (result) {
        is CaptureResult.Completed -> null
        is CaptureResult.Failed -> Outcome(RecognizerIntent.RESULT_AUDIO_ERROR)
        CaptureResult.AlreadyStopped -> Outcome(RecognizerIntent.RESULT_AUDIO_ERROR)
    }
}
