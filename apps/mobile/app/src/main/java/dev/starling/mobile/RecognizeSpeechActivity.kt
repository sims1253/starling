package dev.starling.mobile

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.RecognizerIntent
import android.text.method.ScrollingMovementMethod
import android.view.View
import android.widget.Button
import android.widget.TextView
import dev.starling.mobile.audio.AudioCapture
import dev.starling.mobile.audio.AudioChunkListener
import dev.starling.mobile.audio.CaptureResult
import dev.starling.mobile.data.Recording
import dev.starling.mobile.network.BackendConfig
import dev.starling.mobile.network.StreamEvent
import dev.starling.mobile.network.StreamSession

/**
 * The `ACTION_RECOGNIZE_SPEECH` popup (E22): apps that ask the system for
 * speech input with `startActivityForResult` get a small Starling dialog
 * that starts listening at once, shows live partials when the configuration
 * streams (on-device or a Starling server), and returns the verbatim final
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
 * the language is whatever the configured model detects.
 *
 * Threading: every method here runs on the main thread (capture stop
 * callbacks and TranscriptionCoordinator callbacks are delivered there), so
 * the plain fields need no synchronization; only [streamSession] is read
 * by the capture worker.
 */
class RecognizeSpeechActivity : Activity() {
    private val application by lazy { starlingApplication() }
    private val capture = AudioCapture()

    private lateinit var statusView: TextView
    private lateinit var partialView: TextView
    private lateinit var doneButton: Button
    private lateinit var cancelButton: Button

    private var activeRecording: Recording? = null

    // Read by the capture worker through the chunk listener.
    @Volatile
    private var streamSession: StreamSession? = null

    /** Set once a result has been delivered; later outcomes are store-only. */
    private var resultDelivered = false

    /** The configuration this capture started with; its outcome is classified by it. */
    private var captureConfig: BackendConfig? = null

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

        doneButton.setOnClickListener { if (activeRecording != null) stopAndTranscribe() else finish() }
        cancelButton.setOnClickListener { cancel() }
        setResult(RESULT_CANCELED)

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
        if (activeRecording != null) stopAndTranscribe()
        super.onStop()
    }

    override fun onDestroy() {
        if (activeRecording != null) stopAndTranscribe()
        super.onDestroy()
    }

    private fun beginCapture() {
        // A permission grant can land after Cancel already finished the
        // dialog; never open the microphone behind a dead dialog.
        if (isFinishing || isDestroyed || resultDelivered) return
        val recording = runCatching { application.recordings.create() }.getOrElse {
            deliver(RecognizeSpeechOutcome.Outcome(RecognizerIntent.RESULT_CLIENT_ERROR), R.string.recording_storage_error)
            return
        }
        val config = application.backendSettings.load()
        captureConfig = config
        var session: StreamSession? = null
        session = application.transcription.beginStreaming(config) { event ->
            if (streamSession === session) onStreamEvent(event)
        }
        val error = capture.start(
            this,
            application.recordings.partialFile(recording),
            onChunk = session?.let { streaming -> AudioChunkListener { bytes, count -> streaming.onAudio(bytes, count) } },
        )
        if (error != null) {
            session?.close()
            runCatching { application.recordings.markFailed(recording.id, error) }
            deliver(RecognizeSpeechOutcome.Outcome(RecognizerIntent.RESULT_AUDIO_ERROR), null, error)
            return
        }
        activeRecording = recording
        streamSession = session
        statusView.setText(R.string.recognize_listening)
    }

    private fun onStreamEvent(event: StreamEvent) {
        if (isDestroyed || isFinishing) return
        when (event) {
            StreamEvent.Live -> Unit
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
            // The stop path falls back to the batch transcription of the WAV.
            is StreamEvent.Interrupted -> Unit
        }
    }

    private fun stopAndTranscribe() {
        val recording = activeRecording ?: return
        activeRecording = null
        val session = streamSession
        streamSession = null
        doneButton.isEnabled = false
        statusView.setText(R.string.recognize_transcribing)
        capture.stop { result -> settleCapture(recording, session, result) }
    }

    private fun settleCapture(recording: Recording, session: StreamSession?, result: CaptureResult) {
        RecognizeSpeechOutcome.captureEnded(result)?.let { failed ->
            session?.close()
            val message = (result as? CaptureResult.Failed)?.message
            if (message != null) runCatching { application.recordings.markFailed(recording.id, message) }
            deliver(failed, R.string.recognize_failed, message)
            return
        }
        val completed = result as CaptureResult.Completed
        // The WAV is finalized and durable before any transcription.
        val finalized = runCatching {
            application.recordings.commitAudio(recording, completed.durationSeconds)
        }.getOrElse {
            session?.close()
            runCatching { application.recordings.markFailed(recording.id, "Unable to finalize the private WAV recording") }
            deliver(RecognizeSpeechOutcome.Outcome(RecognizerIntent.RESULT_CLIENT_ERROR), R.string.recording_finalize_error)
            return
        }
        val config = captureConfig ?: application.backendSettings.load()
        val settled: (Recording) -> Unit = { done ->
            val outcome = RecognizeSpeechOutcome.settled(done, config.engine)
            val message = when (outcome.resultCode) {
                RESULT_OK -> null
                RecognizerIntent.RESULT_NO_MATCH -> R.string.recognize_no_match
                else -> R.string.recognize_failed
            }
            deliver(outcome, message)
        }
        if (session != null) {
            application.transcription.finishStreaming(session, finalized.id, config, settled)
        } else {
            application.transcription.transcribe(finalized.id, config, settled)
        }
    }

    private fun cancel() {
        val recording = activeRecording
        if (recording != null) {
            activeRecording = null
            val session = streamSession
            streamSession = null
            // Like a cancelled recognizer session: keep the audio, skip the
            // transcription, and drop any live session.
            session?.close()
            capture.stop { result ->
                when (result) {
                    is CaptureResult.Completed ->
                        runCatching { application.recordings.commitAudio(recording, result.durationSeconds) }
                    is CaptureResult.Failed ->
                        runCatching { application.recordings.markFailed(recording.id, result.message) }
                    CaptureResult.AlreadyStopped -> Unit
                }
            }
        }
        // Dismissing an error screen keeps the error code already set for
        // the caller; RESULT_CANCELED is only for a genuine user cancel.
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
