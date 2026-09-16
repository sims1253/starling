package dev.starling.mobile

import android.Manifest
import android.content.ContextParams
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionService
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import dev.starling.mobile.audio.AudioCapture
import dev.starling.mobile.data.RecordingStatus
import java.util.concurrent.TimeUnit

/**
 * System-wide speech recognizer for third-party keyboards. Host keyboards
 * that use the platform SpeechRecognizer API route their microphone button
 * here once Starling is selected as the voice input service, so dictation
 * works inside a normal keyboard without switching IMEs.
 *
 * The transcript is delivered once, after the batch server response. The
 * audio and its outcome stay durable in the recordings store, so a failed
 * upload remains retryable from the app. No partial results are produced.
 * Session transitions are decided by [RecognitionSessionGuard], which is
 * unit-tested separately.
 */
class StarlingRecognitionService : RecognitionService() {
    private val application by lazy { starlingApplication() }
    private val capture = AudioCapture()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val sessions = RecognitionSessionGuard<Callback>()

    private val listenTimeout = Runnable {
        // A wedged host keyboard must not hold the microphone open forever.
        sessions.expire()?.let(::endSession)
    }

    override fun onStartListening(recognizerIntent: Intent?, callback: Callback) {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            sessions.deliver(callback) { it.error(SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS) }
            return
        }
        // Finishing a still-live session here is defensive only: the framework
        // answers a second start with ERROR_RECOGNIZER_BUSY without calling us.
        sessions.expire()?.let(::endSession)
        val recording = runCatching { application.recordings.create() }.getOrElse {
            sessions.deliver(callback) { it.error(SpeechRecognizer.ERROR_CLIENT) }
            return
        }
        // Attribute mic access to the calling keyboard so its identity, not
        // ours, is checked and shown during data delivery (API 33+).
        val captureContext = if (Build.VERSION.SDK_INT >= 33) {
            runCatching {
                createContext(
                    ContextParams.Builder()
                        .setNextAttributionSource(callback.callingAttributionSource)
                        .build(),
                )
            }.getOrElse { this }
        } else {
            this
        }
        val error = capture.start(captureContext, application.recordings.partialFile(recording))
        if (error != null) {
            runCatching { application.recordings.markFailed(recording.id, error) }
            sessions.deliver(callback) { it.error(SpeechRecognizer.ERROR_CLIENT) }
            return
        }
        sessions.begin(recording, callback)
        mainHandler.removeCallbacks(listenTimeout)
        mainHandler.postDelayed(listenTimeout, MAX_LISTEN_MILLIS)
        // Keyboards wait for this before showing their "speak now" state.
        sessions.deliver(callback) { it.readyForSpeech(Bundle()) }
    }

    override fun onStopListening(callback: Callback) {
        sessions.stopListening(callback)?.let(::endSession)
    }

    override fun onCancel(callback: Callback) {
        sessions.cancel(callback)?.let(::endSession)
    }

    override fun onDestroy() {
        mainHandler.removeCallbacks(listenTimeout)
        sessions.teardown()?.let(::endSession)
        super.onDestroy()
    }

    private fun endSession(ending: RecognitionSessionGuard.Ending<Callback>) {
        mainHandler.removeCallbacks(listenTimeout)
        when (val settlement = sessions.settle(ending, capture.stop())) {
            is RecognitionSessionGuard.Settlement.Transcribe -> {
                val finalized = runCatching {
                    application.recordings.commitAudio(settlement.recording, settlement.durationSeconds)
                }.getOrNull()
                if (finalized == null) {
                    sessions.deliver(ending.session.owner) { it.error(SpeechRecognizer.ERROR_CLIENT) }
                    return
                }
                val config = application.backendSettings.load()
                application.transcription.transcribe(finalized.id, config) { completed ->
                    if (completed.status == RecordingStatus.TRANSCRIBED &&
                        completed.rawTranscript != null
                    ) {
                        sessions.deliver(ending.session.owner) {
                            it.results(
                                Bundle().apply {
                                    putStringArrayList(
                                        RecognizerIntent.EXTRA_RESULTS,
                                        arrayListOf(completed.rawTranscript),
                                    )
                                },
                            )
                        }
                    } else {
                        // The upload failed but the recording is retained and
                        // retryable from the app; report a network-class error.
                        sessions.deliver(ending.session.owner) { it.error(SpeechRecognizer.ERROR_NETWORK) }
                    }
                }
            }
            is RecognitionSessionGuard.Settlement.Keep ->
                // The recording itself was the user's intent; keep it, skip upload.
                runCatching {
                    application.recordings.commitAudio(settlement.recording, settlement.durationSeconds)
                }
            is RecognitionSessionGuard.Settlement.Fail -> {
                runCatching { application.recordings.markFailed(settlement.recordingId, settlement.message) }
                settlement.errorCode?.let { code ->
                    sessions.deliver(ending.session.owner) { it.error(code) }
                }
            }
            is RecognitionSessionGuard.Settlement.Empty ->
                settlement.errorCode?.let { code ->
                    sessions.deliver(ending.session.owner) { it.error(code) }
                }
        }
    }

    companion object {
        private val MAX_LISTEN_MILLIS = TimeUnit.MINUTES.toMillis(1)
    }
}
