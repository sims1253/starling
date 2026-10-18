package dev.starling.mobile

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.speech.RecognizerIntent
import android.text.method.ScrollingMovementMethod
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.window.OnBackInvokedDispatcher
import dev.starling.mobile.storage.DiskLevel
import dev.starling.mobile.transcription.StreamEvent
import dev.starling.mobile.transcription.TakeRecorder

/**
 * The `ACTION_RECOGNIZE_SPEECH` popup (E22): apps that ask the system for
 * speech input with `startActivityForResult` get a small Starling dialog
 * that starts listening at once, shows live partials from the on-device
 * model, and returns the verbatim final
 * transcript as the single `EXTRA_RESULTS` entry when the user taps Done.
 *
 * The capture follows the same durable-first rules as the rest of the app:
 * the WAV is finalized before any transcription, every dictation stays in
 * Saved recordings (including failures, for retry), and Cancel keeps the
 * audio without transcribing it. The caller's identity and extras other than
 * the prompt are not read or stored.
 *
 * Supported subset of the RecognizerIntent contract: EXTRA_PROMPT is shown
 * (bounded, since any app can start this exported activity); EXTRA_LANGUAGE,
 * EXTRA_LANGUAGE_MODEL, EXTRA_MAX_RESULTS and EXTRA_PARTIAL_RESULTS are not
 * honored. The result is always the single verbatim final transcript, and
 * the language is whatever the active model detects.
 *
 * Threading: every method here runs on the main thread (capture stop
 * callbacks and TranscriptionCoordinator callbacks are delivered there), so
 * the plain fields need no synchronization. [TakeRecorder] forwards PCM
 * to the live session on the capture worker.
 */
class RecognizeSpeechActivity : Activity() {
    private val application by lazy { starlingApplication() }
    private val recorder by lazy { TakeRecorder(this, application.recordings, application.transcription) }

    private lateinit var statusView: TextView
    private lateinit var partialView: TextView
    private lateinit var doneButton: Button
    private lateinit var cancelButton: Button

    private var activeTake: TakeRecorder.Take? = null

    // The listening status and the take's free-space warning (#342) below it.
    private var takeStatus = ""
    private var diskNotice: String? = null

    /** Whether the microphone capture is running; instrumented tests branch on it. */
    internal val capturing: Boolean get() = activeTake != null

    /** Set once a result has been delivered; later outcomes are store-only. */
    private var resultDelivered = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_recognize_speech)
        setFinishOnTouchOutside(false)
        statusView = findViewById(R.id.recognize_status)
        partialView = findViewById(R.id.recognize_partial)
        partialView.movementMethod = ScrollingMovementMethod()
        doneButton = findViewById(R.id.recognize_done_button)
        cancelButton = findViewById(R.id.recognize_cancel_button)
        findViewById<TextView>(R.id.recognize_prompt).text =
            intent?.getStringExtra(RecognizerIntent.EXTRA_PROMPT)
                ?.filterNot(Char::isISOControl)
                ?.let { prompt ->
                    // Truncate on code points so a surrogate pair is never split.
                    if (prompt.codePointCount(0, prompt.length) <= MAX_PROMPT_CHARS) prompt
                    else prompt.substring(0, prompt.offsetByCodePoints(0, MAX_PROMPT_CHARS))
                }
                ?.takeIf { it.isNotBlank() }
                ?: getString(R.string.recognize_prompt)

        doneButton.setOnClickListener { if (activeTake != null) stopAndTranscribe() else finish() }
        cancelButton.setOnClickListener { cancel() }
        if (Build.VERSION.SDK_INT >= 33) {
            // The activity opts in to OnBackInvokedCallback (manifest), so
            // system Back and the predictive-back gesture on Android 14+
            // arrive here; Android 13 ignores the per-activity opt-in and,
            // like older releases, uses onBackPressed below.
            onBackInvokedDispatcher.registerOnBackInvokedCallback(OnBackInvokedDispatcher.PRIORITY_DEFAULT) { cancel() }
        }
        setResult(RESULT_CANCELED)
        // The popup records at once; loading now also covers the time a
        // permission prompt is on screen.
        application.preloadOnDeviceModel()

        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            beginCapture()
        } else {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != REQUEST_RECORD_AUDIO) return
        if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            beginCapture()
        } else {
            deliver(RecognizeSpeechOutcome.Outcome(RecognizerIntent.RESULT_AUDIO_ERROR), R.string.recognize_permission_denied)
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        cancel()
    }

    override fun onStop() {
        // Never keep the microphone open behind another app. Leaving counts
        // as Done: the audio is saved and transcribed into Saved recordings.
        // The caller only gets the transcript if this dialog is still alive
        // when it settles; otherwise it sees RESULT_CANCELED and the
        // transcript stays retrievable in Starling Mobile.
        if (activeTake != null) stopAndTranscribe()
        super.onStop()
    }

    override fun onDestroy() {
        if (activeTake != null) stopAndTranscribe()
        super.onDestroy()
    }

    private fun beginCapture() {
        // A permission grant can land after Cancel already finished the
        // dialog; never open the microphone behind a dead dialog.
        if (isFinishing || isDestroyed || resultDelivered) return
        // Free space first (#342): no take starts that the disk cannot hold.
        val disk = application.diskBeforeTake()
        if (disk?.level == DiskLevel.CRITICAL) {
            deliver(
                RecognizeSpeechOutcome.Outcome(RecognizerIntent.RESULT_AUDIO_ERROR),
                null,
                getString(R.string.disk_full_refused, (disk.availableBytes / 1_000_000).toInt()),
            )
            return
        }
        val recorded = when (val started = recorder.start(
            diskWarned = application.diskWarnedAtStart(disk),
            onEvent = ::onStreamEvent,
            onEnded = ::stopAndTranscribe,
            onDiskLow = { minutes ->
                if (!isDestroyed && !isFinishing) {
                    diskNotice = application.diskLowDuringTake(minutes)
                    showListeningStatus(takeStatus)
                }
            },
        )) {
            is TakeRecorder.Start.Started -> started.take
            is TakeRecorder.Start.Failed -> {
                deliver(
                    RecognizeSpeechOutcome.Outcome(if (started.storage) RecognizerIntent.RESULT_CLIENT_ERROR else RecognizerIntent.RESULT_AUDIO_ERROR),
                    if (started.storage) R.string.recording_storage_error else null,
                    if (started.storage) null else started.message,
                )
                return
            }
        }
        activeTake = recorded
        val loading = recorded.live && application.isOnDeviceModelLoading()
        diskNotice = application.diskWarning(disk)
        showListeningStatus(getString(if (loading) R.string.recognize_listening_loading else R.string.recognize_listening))
    }

    /** [status] of the running take, with its free-space warning on the next line. */
    private fun showListeningStatus(status: String) {
        takeStatus = status
        statusView.text = listOfNotNull(status, diskNotice).joinToString("\n")
    }

    private fun onStreamEvent(event: StreamEvent) {
        if (isDestroyed || isFinishing) return
        when (event) {
            StreamEvent.Live -> if (activeTake != null) showListeningStatus(getString(R.string.recognize_listening))
            is StreamEvent.Partial -> {
                partialView.visibility = View.VISIBLE
                partialView.text = event.text
                // Keep the newest words in view as the partial grows.
                partialView.post {
                    val layout = partialView.layout ?: return@post
                    val bottom = layout.getLineTop(partialView.lineCount) -
                        (partialView.height - partialView.totalPaddingTop - partialView.totalPaddingBottom)
                    partialView.scrollTo(0, maxOf(0, bottom))
                }
            }
            // The stop path falls back to the batch transcription of the WAV;
            // the status must not keep claiming the model is loading.
            is StreamEvent.Interrupted -> if (activeTake != null) {
                showListeningStatus(getString(R.string.keyboard_stream_interrupted, event.reason))
            }
        }
    }

    private fun stopAndTranscribe() {
        val recorded = activeTake ?: return
        activeTake = null
        diskNotice = null
        doneButton.isEnabled = false
        statusView.setText(R.string.recognize_transcribing)
        recorded.stop(
            onStopped = { result ->
                when (result) {
                    is TakeRecorder.Stopped.Saved -> if (result.capture.stoppedForLowDisk && !isDestroyed && !isFinishing) {
                        statusView.setText(R.string.recording_stopped_low_disk)
                    }
                    is TakeRecorder.Stopped.Failed -> deliver(
                        RecognizeSpeechOutcome.Outcome(
                            when (result.kind) {
                                TakeRecorder.Failure.FINALIZATION, TakeRecorder.Failure.TRANSCRIPTION -> RecognizerIntent.RESULT_CLIENT_ERROR
                                else -> RecognizerIntent.RESULT_AUDIO_ERROR
                            },
                        ),
                        if (result.kind == TakeRecorder.Failure.FINALIZATION) R.string.recording_finalize_error else R.string.recognize_failed,
                        result.message,
                    )
                    TakeRecorder.Stopped.Discarded -> Unit
                }
            },
            onTranscribed = { done ->
                val outcome = RecognizeSpeechOutcome.settled(done)
                val message = when (outcome.resultCode) {
                    RESULT_OK -> null
                    RecognizerIntent.RESULT_NO_MATCH -> R.string.recognize_no_match
                    else -> R.string.recognize_failed
                }
                deliver(outcome, message)
            },
        )
    }

    private fun cancel() {
        val recorded = activeTake
        activeTake = null
        recorded?.stop(transcribe = false)
        // Dismissing an error screen keeps the error code already set.
        if (!resultDelivered) {
            resultDelivered = true
            setResult(RESULT_CANCELED)
        }
        finish()
    }

    /**
     * Sets the caller's result once. Success closes the dialog right away;
     * failures stay on screen with their reason until the user closes it.
     */
    private fun deliver(outcome: RecognizeSpeechOutcome.Outcome, messageRes: Int?, detail: String? = null) {
        if (resultDelivered) return
        resultDelivered = true
        val transcript = outcome.transcript
        if (outcome.resultCode == RESULT_OK && !transcript.isNullOrBlank()) {
            setResult(
                RESULT_OK,
                Intent().putStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS, arrayListOf(transcript)),
            )
            if (!isFinishing) finish()
            return
        }
        // RESULT_OK is never sent without a transcript.
        val code = if (outcome.resultCode == RESULT_OK) RecognizerIntent.RESULT_NO_MATCH else outcome.resultCode
        setResult(code)
        if (isDestroyed || isFinishing) return
        // A leftover live partial next to "failed" would read as a result.
        partialView.visibility = View.GONE
        statusView.text = listOfNotNull(messageRes?.let(::getString), detail).joinToString(" ")
        doneButton.isEnabled = true
        doneButton.setText(R.string.recognize_close)
        cancelButton.visibility = View.GONE
    }

    companion object {
        private const val REQUEST_RECORD_AUDIO = 4101
        private const val MAX_PROMPT_CHARS = 200
    }
}
