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
import dev.starling.mobile.audio.CaptureResult
import dev.starling.mobile.data.Recording
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
 */
class StarlingRecognitionService : RecognitionService() {
    private val application by lazy { starlingApplication() }
    private val capture = AudioCapture()
    private val mainHandler = Handler(Looper.getMainLooper())

    private var activeRecording: Recording? = null
    private var activeCallback: Callback? = null

    private val listenTimeout = Runnable {
        val callback = activeCallback ?: return@Runnable
        // A wedged host keyboard must not hold the microphone open forever.
        finishAndTranscribe(callback)
    }

    override fun onStartListening(recognizerIntent: Intent?, callback: Callback) {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            callback.deliverError(SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS)
            return
        }
        // Finishing a still-live session here is defensive only: the framework
        // answers a second start with ERROR_RECOGNIZER_BUSY without calling us.
        val previous = activeCallback
        if (activeRecording != null && previous != null) {
            finishAndTranscribe(previous)
        }
        val recording = runCatching { application.recordings.create() }.getOrElse {
            callback.deliverError(SpeechRecognizer.ERROR_CLIENT)
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
            callback.deliverError(SpeechRecognizer.ERROR_CLIENT)
            return
        }
        mainHandler.removeCallbacks(listenTimeout)
        mainHandler.postDelayed(listenTimeout, MAX_LISTEN_MILLIS)
        activeRecording = recording
        activeCallback = callback
        // Keyboards wait for this before showing their "speak now" state.
        callback.deliverReadyForSpeech()
    }

    override fun onStopListening(callback: Callback) {
        if (activeCallback != callback) return
        finishAndTranscribe(callback)
    }

    override fun onCancel(callback: Callback) {
        if (activeCallback != callback) return
        val recording = activeRecording ?: return
        mainHandler.removeCallbacks(listenTimeout)
        activeRecording = null
        activeCallback = null
        // The recording itself was the user's intent; keep it, skip upload.
        when (val result = capture.stop()) {
            is CaptureResult.Completed ->
                runCatching { application.recordings.commitAudio(recording, result.durationSeconds) }
            is CaptureResult.Failed ->
                runCatching { application.recordings.markFailed(recording.id, result.message) }
            CaptureResult.AlreadyStopped -> Unit
        }
    }

    override fun onDestroy() {
        mainHandler.removeCallbacks(listenTimeout)
        val recording = activeRecording
        activeRecording = null
        activeCallback = null
        if (recording != null) {
            when (val result = capture.stop()) {
                is CaptureResult.Completed ->
                    runCatching { application.recordings.commitAudio(recording, result.durationSeconds) }
                is CaptureResult.Failed ->
                    runCatching { application.recordings.markFailed(recording.id, result.message) }
                CaptureResult.AlreadyStopped -> Unit
            }
        }
        super.onDestroy()
    }

    private fun finishAndTranscribe(callback: Callback) {
        mainHandler.removeCallbacks(listenTimeout)
        val recording = activeRecording
        activeRecording = null
        activeCallback = null
        if (recording == null) {
            callback.deliverError(SpeechRecognizer.ERROR_CLIENT)
            return
        }
        when (val result = capture.stop()) {
            is CaptureResult.Completed -> {
                val finalized = runCatching {
                    application.recordings.commitAudio(recording, result.durationSeconds)
                }.getOrNull()
                if (finalized == null) {
                    callback.deliverError(SpeechRecognizer.ERROR_CLIENT)
                    return
                }
                val config = application.backendSettings.load()
                application.transcription.transcribe(finalized.id, config) { completed ->
                    if (completed.status == RecordingStatus.TRANSCRIBED &&
                        completed.rawTranscript != null
                    ) {
                        callback.deliverResults(
                            Bundle().apply {
                                putStringArrayList(
                                    RecognizerIntent.EXTRA_RESULTS,
                                    arrayListOf(completed.rawTranscript),
                                )
                            },
                        )
                    } else {
                        // The upload failed but the recording is retained and
                        // retryable from the app; report a network-class error.
                        callback.deliverError(SpeechRecognizer.ERROR_NETWORK)
                    }
                }
            }
            is CaptureResult.Failed -> {
                runCatching { application.recordings.markFailed(recording.id, result.message) }
                callback.deliverError(SpeechRecognizer.ERROR_CLIENT)
            }
            CaptureResult.AlreadyStopped -> callback.deliverError(SpeechRecognizer.ERROR_CLIENT)
        }
    }

    // Terminal deliveries are binder calls into the host keyboard's process.
    // A dead listener binder must not crash this process; the recording is
    // already durable, so a swallowed delivery failure loses nothing.
    private fun Callback.deliverResults(bundle: Bundle) {
        runCatching { results(bundle) }
    }

    private fun Callback.deliverError(errorCode: Int) {
        runCatching { error(errorCode) }
    }

    private fun Callback.deliverReadyForSpeech() {
        runCatching { readyForSpeech(Bundle()) }
    }

    companion object {
        private val MAX_LISTEN_MILLIS = TimeUnit.MINUTES.toMillis(1)
    }
}
