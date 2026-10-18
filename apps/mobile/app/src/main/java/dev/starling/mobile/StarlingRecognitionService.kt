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
import android.util.Log
import android.widget.Toast
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
import dev.starling.mobile.storage.DiskLevel
import dev.starling.mobile.transcription.StreamEvent
import dev.starling.mobile.transcription.TakeRecorder
import java.util.concurrent.TimeUnit

/**
 * System-wide speech recognizer for third-party keyboards. Host keyboards
 * that use the platform SpeechRecognizer API route their microphone button
 * here once Starling is selected as the voice input service, so dictation
 * works inside a normal keyboard without switching IMEs.
 *
 * Capture is transcribed live on the device and growing partials are
 * delivered through [RecognitionService.Callback.partialResults]; the final
 * result still arrives once, in [RecognitionService.Callback.results], after
 * the live session finishes or — on any failure of it — after the batch
 * transcription of the saved WAV. The audio and its outcome stay durable in
 * the recordings store, so a failed transcription remains retryable from
 * the app. Session transitions are decided by [RecognitionSessionGuard],
 * which is unit-tested separately.
 */
class StarlingRecognitionService : RecognitionService() {
    private val application by lazy { starlingApplication() }
    private val recorder by lazy { TakeRecorder(this, application.recordings, application.transcription) }
    private val mainHandler = Handler(Looper.getMainLooper())
    private val sessions = RecognitionSessionGuard<Callback>()

    private var recordedTake: TakeRecorder.Take? = null

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
        // Free space first (#342): no take starts that the disk cannot hold.
        // The service has no screen of its own; the host keyboard gets the
        // error (RecognitionSessionGuard.DISK_FULL_ERROR), and the reason
        // shows as a toast and in the log.
        val disk = application.diskBeforeTake()
        if (disk?.level == DiskLevel.CRITICAL) {
            runCatching { Log.w(TAG, "refused a take: ${disk.availableBytes} bytes free, below the stop threshold") }
            toast(getString(R.string.disk_full_refused, (disk.availableBytes / 1_000_000).toInt()))
            sessions.deliver(callback) { it.error(RecognitionSessionGuard.DISK_FULL_ERROR) }
            return
        }
        application.diskWarning(disk)?.let(::toast)
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
        val recorded = when (val started = recorder.start(
            attributionContext = captureContext,
            diskWarned = application.diskWarnedAtStart(disk),
            onEvent = { onStreamEvent(callback, it) },
            onEnded = { sessions.stopListening(callback)?.let(::endSession) },
            onDiskLow = { minutes ->
                if (sessions.isLive(callback)) {
                    runCatching { Log.w(TAG, "storage is low during a take: about $minutes minutes left") }
                    toast(application.diskLowDuringTake(minutes))
                }
            },
        )) {
            is TakeRecorder.Start.Started -> started.take
            is TakeRecorder.Start.Failed -> {
                sessions.deliver(callback) { it.error(SpeechRecognizer.ERROR_CLIENT) }
                return
            }
        }
        recordedTake = recorded
        sessions.begin(recorded.recording, callback)
        mainHandler.removeCallbacks(listenTimeout)
        mainHandler.postDelayed(listenTimeout, MAX_LISTEN_MILLIS)
        // Keyboards wait for this before showing their "speak now" state.
        sessions.deliver(callback) { it.readyForSpeech(Bundle()) }
    }

    /**
     * Live-stream events, already marshalled to the main thread. Partials
     * map one-to-one onto the platform's partialResults callback; live and
     * interrupted states have no RecognitionService equivalent, and an
     * interrupted stream needs no error here — the final result still
     * arrives through the batch fallback after Stop.
     */
    private fun onStreamEvent(owner: Callback, event: StreamEvent) {
        if (event !is StreamEvent.Partial) return
        if (!sessions.isLive(owner)) return
        sessions.deliver(owner) {
            it.partialResults(
                Bundle().apply {
                    putStringArrayList(
                        RecognizerIntent.EXTRA_RESULTS,
                        arrayListOf(event.text),
                    )
                },
            )
        }
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
        val recorded = recordedTake ?: return
        recordedTake = null
        val finalize = ending is RecognitionSessionGuard.Ending.Finalize
        recorded.stop(
            transcribe = finalize,
            onStopped = { result ->
                when (result) {
                    is TakeRecorder.Stopped.Saved -> if (finalize && result.capture.stoppedForLowDisk) {
                        runCatching { Log.w(TAG, "a take stopped early: storage is nearly full") }
                        toast(getString(R.string.recording_stopped_low_disk))
                    }
                    is TakeRecorder.Stopped.Failed -> if (finalize) {
                        sessions.deliver(ending.session.owner) { it.error(SpeechRecognizer.ERROR_CLIENT) }
                    }
                    TakeRecorder.Stopped.Discarded -> Unit
                }
            },
            onTranscribed = { deliverSettled(ending, it) },
        )
    }

    private fun deliverSettled(
        ending: RecognitionSessionGuard.Ending<Callback>,
        completed: Recording,
    ) {
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
            // The transcription failed but the recording is retained
            // and retryable from the app; a local engine problem (such as
            // a missing model) is the client's.
            sessions.deliver(ending.session.owner) { it.error(SpeechRecognizer.ERROR_CLIENT) }
        }
    }

    private fun toast(text: String) {
        runCatching { Toast.makeText(this, text, Toast.LENGTH_LONG).show() }
    }

    companion object {
        private const val TAG = "StarlingRecognition"
        private val MAX_LISTEN_MILLIS = TimeUnit.MINUTES.toMillis(1)
    }
}
