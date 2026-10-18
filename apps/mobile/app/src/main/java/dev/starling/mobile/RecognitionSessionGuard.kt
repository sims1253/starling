package dev.starling.mobile

import android.speech.SpeechRecognizer
import dev.starling.mobile.data.Recording
import dev.starling.mobile.transcription.TakeRecorder

/**
 * Tracks the recognition client that owns the live take. [TakeRecorder]
 * owns audio settlement; this guard decides whether an ending transcribes
 * and protects binder deliveries to a client that may have died.
 */
class RecognitionSessionGuard<T : Any> {
    /** The single live session: its durable recording and owning client. */
    data class Session<out T : Any>(val recording: Recording, val owner: T)

    /** What happens to a session the service ends. */
    sealed interface Ending<out T : Any> {
        val session: Session<T>

        /** Upload the finalized audio, then deliver one final result or error. */
        data class Finalize<out T : Any>(override val session: Session<T>) : Ending<T>

        /** Keep the audio; no transcription and no delivery. */
        data class Abandon<out T : Any>(override val session: Session<T>) : Ending<T>
    }

    private var session: Session<T>? = null

    /** Registers the only live session once its capture has started. */
    fun begin(recording: Recording, owner: T) {
        session = Session(recording, owner)
    }

    /**
     * Whether [owner] still owns the live session. Used to gate partial
     * deliveries that may race a session ending underneath them.
     */
    fun isLive(owner: T): Boolean = session?.owner == owner

    /** The owning client stopped: finalize the audio for transcription. */
    fun stopListening(owner: T): Ending<T>? = endIfOwned(owner) { Ending.Finalize(it) }

    /** The owning client cancelled: keep the audio, skip the transcription. */
    fun cancel(owner: T): Ending<T>? = endIfOwned(owner) { Ending.Abandon(it) }

    /** Watchdog expiry or a defensive restart: finalize whatever is live. */
    fun expire(): Ending<T>? = endLive { Ending.Finalize(it) }

    /** Service teardown: keep the audio like a cancel. */
    fun teardown(): Ending<T>? = endLive { Ending.Abandon(it) }

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

    companion object {
        /**
         * The error a take refused for lack of storage gets (#342):
         * ERROR_AUDIO, "audio recording error". The platform has no storage
         * error, the recording is what cannot be made, and it is the class
         * the RECOGNIZE_SPEECH popup reports (RESULT_AUDIO_ERROR).
         */
        const val DISK_FULL_ERROR = SpeechRecognizer.ERROR_AUDIO
    }
}
