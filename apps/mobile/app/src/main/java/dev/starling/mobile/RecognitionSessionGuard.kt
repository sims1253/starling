package dev.starling.mobile

import android.speech.SpeechRecognizer
import dev.starling.mobile.audio.CaptureResult
import dev.starling.mobile.data.Recording

/**
 * Session-state decisions for [StarlingRecognitionService]: which client owns
 * the live session, what ends it, how its stopped capture settles (committed
 * for upload, kept locally without upload, or marked failed), and whether a
 * terminal error is delivered. Pure Kotlin in the guard-class shape of
 * InputTargetGuard and RequestGenerationGuard so every transition is
 * unit-testable on the JVM; the service applies the returned decisions
 * against the recordings store and the host keyboard's callback.
 */
class RecognitionSessionGuard<T : Any> {
    /** The single live session: its durable recording and owning client. */
    data class Session<out T : Any>(val recording: Recording, val owner: T)

    /** What happens to a session the service ends. */
    sealed interface Ending<out T : Any> {
        val session: Session<T>

        /** Upload the finalized audio, then deliver one final result or error. */
        data class Finalize<out T : Any>(override val session: Session<T>) : Ending<T>

        /** Keep the audio; no upload and no delivery. */
        data class Abandon<out T : Any>(override val session: Session<T>) : Ending<T>
    }

    /** How the stopped capture of an ended session settles. */
    sealed interface Settlement {
        /** Commit the WAV and upload it; deliver the transcript when it returns. */
        data class Transcribe(val recording: Recording, val durationSeconds: Double) : Settlement

        /** Commit the WAV locally; the audio stays and nothing further happens. */
        data class Keep(val recording: Recording, val durationSeconds: Double) : Settlement

        /** Mark the recording failed; deliver the code when one is set. */
        data class Fail(val recordingId: String, val message: String, val errorCode: Int?) : Settlement

        /** Nothing was captured to settle; deliver the code when one is set. */
        data class Empty(val errorCode: Int?) : Settlement
    }

    private var session: Session<T>? = null

    /** Registers the only live session once its capture has started. */
    fun begin(recording: Recording, owner: T) {
        session = Session(recording, owner)
    }

    /** The owning client stopped: finalize the audio for transcription. */
    fun stopListening(owner: T): Ending<T>? = endIfOwned(owner) { Ending.Finalize(it) }

    /** The owning client cancelled: keep the audio, skip the upload. */
    fun cancel(owner: T): Ending<T>? = endIfOwned(owner) { Ending.Abandon(it) }

    /** Watchdog expiry or a defensive restart: finalize whatever is live. */
    fun expire(): Ending<T>? = endLive { Ending.Finalize(it) }

    /** Service teardown: keep the audio like a cancel. */
    fun teardown(): Ending<T>? = endLive { Ending.Abandon(it) }

    fun settle(ending: Ending<T>, result: CaptureResult): Settlement = when (ending) {
        is Ending.Finalize -> when (result) {
            is CaptureResult.Completed ->
                Settlement.Transcribe(ending.session.recording, result.durationSeconds)
            is CaptureResult.Failed ->
                Settlement.Fail(ending.session.recording.id, result.message, SpeechRecognizer.ERROR_CLIENT)
            CaptureResult.AlreadyStopped -> Settlement.Empty(SpeechRecognizer.ERROR_CLIENT)
        }
        is Ending.Abandon -> when (result) {
            is CaptureResult.Completed ->
                Settlement.Keep(ending.session.recording, result.durationSeconds)
            is CaptureResult.Failed ->
                Settlement.Fail(ending.session.recording.id, result.message, errorCode = null)
            CaptureResult.AlreadyStopped -> Settlement.Empty(errorCode = null)
        }
    }

    /**
     * Terminal deliveries are binder calls into the host keyboard's process.
     * A dead listener binder must not crash this process; the recording is
     * already durable, so a swallowed delivery failure loses nothing.
     */
    fun deliver(owner: T, delivery: (T) -> Unit) {
        runCatching { delivery(owner) }
    }

    private fun endIfOwned(owner: T, ending: (Session<T>) -> Ending<T>): Ending<T>? {
        val current = session ?: return null
        if (current.owner != owner) return null
        session = null
        return ending(current)
    }

    private fun endLive(ending: (Session<T>) -> Ending<T>): Ending<T>? {
        val current = session ?: return null
        session = null
        return ending(current)
    }
}
