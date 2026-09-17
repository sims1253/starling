package dev.starling.mobile

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.inputmethodservice.InputMethodService
import android.view.LayoutInflater
import android.view.View
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputConnection
import android.widget.Button
import android.widget.FrameLayout
import android.widget.TextView
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import dev.starling.mobile.audio.AudioCapture
import dev.starling.mobile.audio.AudioChunkListener
import dev.starling.mobile.audio.CaptureResult
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
import dev.starling.mobile.data.TranscriptionProvenance
import dev.starling.mobile.network.StreamEvent
import dev.starling.mobile.network.StreamSession
import dev.starling.mobile.ui.InputTargetGuard
import dev.starling.mobile.ui.RequestGenerationGuard

/**
 * Lightweight voice keyboard. It never reads surrounding editor text or
 * package names. Two text paths exist, both guarded by the editor target
 * the recording started against:
 *
 * - **Live streaming** (Starling server): while recording, growing partial
 *   transcripts are shown through `setComposingText`, the Android dictation
 *   idiom — composing text is ephemeral, replaces itself with each partial,
 *   and is removed when the stream fails. The single `commitText` happens
 *   only when the server's final transcript arrives after Stop.
 * - **Batch transcription** (fallback after any streaming failure, on-device
 *   engine, or OpenAI-shaped endpoint): no asynchronous insertion at all;
 *   the transcript is shown first and `commitText` is a separate, explicit
 *   user action on the Insert button.
 */
class VoiceInputService : InputMethodService() {
    private val application by lazy { starlingApplication() }
    private val capture = AudioCapture()
    private val targetGuard = InputTargetGuard<InputConnection>()
    private val requestGuard = RequestGenerationGuard()

    private var keyboardView: View? = null
    private var recordButton: Button? = null
    private var insertButton: Button? = null
    private var statusView: TextView? = null
    private var transcriptView: TextView? = null
    private var activeRecording: Recording? = null
    private var recordingTarget: InputTargetGuard.Snapshot<InputConnection>? = null
    private var activeRequestGeneration = 0L
    private var readyTranscript: ReadyTranscript? = null

    // Live-stream state. The session is set before the capture starts and
    // read by the capture worker through the chunk listener; composingTarget
    // is the editor snapshot the composing region belongs to, main thread
    // only, kept until the stream's final text settles it.
    @Volatile
    private var streamSession: StreamSession? = null
    private var composingTarget: InputTargetGuard.Snapshot<InputConnection>? = null

    override fun onCreate() {
        super.onCreate()
    }

    override fun onCreateInputView(): View {
        // InputMethodService.setInputView() re-parameters whatever view it is
        // handed with MATCH_PARENT/WRAP_CONTENT, so a fixed height on the
        // returned root would be discarded before the first measure. The
        // keyboard therefore lives inside a container that wraps the
        // @dimen/keyboard_height child, which keeps the height stable.
        val container = FrameLayout(this)
        val view = LayoutInflater.from(this).inflate(R.layout.keyboard_view, container, true)
        applyWindowInsets(view)
        keyboardView = view
        recordButton = view.findViewById(R.id.keyboard_record_button)
        insertButton = view.findViewById(R.id.keyboard_insert_button)
        statusView = view.findViewById(R.id.keyboard_status)
        transcriptView = view.findViewById(R.id.keyboard_transcript)

        recordButton?.setOnClickListener {
            if (activeRecording == null) requestOrStartRecording() else stopAndQueueRecording()
        }
        insertButton?.setOnClickListener { insertReadyTranscript() }
        renderIdle()
        return container
    }

    override fun onStartInput(attribute: EditorInfo?, restarting: Boolean) {
        super.onStartInput(attribute, restarting)
        if (activeRecording != null) stopAndQueueRecording()
        val connection = currentInputConnection
        if (connection != null) targetGuard.targetStarted(connection) else targetGuard.targetFinished()
        // A response belonging to a previous editor must never become an
        // insert button for this one. The recording itself stays on disk.
        readyTranscript = null
        transcriptView?.visibility = View.GONE
        transcriptView?.text = null
        renderIdle()
    }

    override fun onFinishInput() {
        if (activeRecording != null) stopAndQueueRecording()
        targetGuard.targetFinished()
        readyTranscript = null
        transcriptView?.visibility = View.GONE
        transcriptView?.text = null
        renderIdle()
        super.onFinishInput()
    }

    override fun onDestroy() {
        if (activeRecording != null) stopAndQueueRecording()
        super.onDestroy()
    }

    private fun requestOrStartRecording() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            statusView?.setText(R.string.keyboard_microphone_permission)
            startActivity(
                Intent(this, MainActivity::class.java)
                    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
            )
            return
        }
        beginCapture()
    }

    private fun beginCapture() {
        val target = targetGuard.capture()
        if (target == null || currentInputConnection == null) {
            statusView?.setText(R.string.keyboard_no_target)
            return
        }
        // Invalidate any earlier request whose callbacks may still be queued.
        val requestGeneration = requestGuard.begin()
        val recording = runCatching { application.recordings.create() }.getOrElse {
            statusView?.setText(R.string.recording_storage_error)
            return
        }
        // The live stream is an observer of the capture; null means this
        // configuration records in the plain batch mode.
        val config = application.backendSettings.load()
        var session: StreamSession? = null
        session = application.transcription.beginStreaming(config) { event ->
            // Events from a superseded session must not touch the state of
            // the recording that replaced it.
            if (streamSession === session) onStreamEvent(event)
        }
        val error = capture.start(
            this,
            application.recordings.partialFile(recording),
            onChunk = session?.let { streaming ->
                AudioChunkListener { bytes, count -> streaming.onAudio(bytes, count) }
            },
        )
        if (error != null) {
            session?.close()
            runCatching { application.recordings.markFailed(recording.id, error) }
            statusView?.text = error
            return
        }
        activeRecording = recording
        streamSession = session
        composingTarget = if (session == null) null else target
        recordingTarget = target
        activeRequestGeneration = requestGeneration
        readyTranscript = null
        transcriptView?.visibility = View.GONE
        transcriptView?.text = null
        insertButton?.visibility = View.GONE
        recordButton?.setText(R.string.keyboard_stop)
        statusView?.setText(
            if (session == null) R.string.keyboard_recording else R.string.keyboard_streaming,
        )
    }

    /**
     * Live-stream events, already marshalled to the main thread. Composing
     * text is written only while the recorded editor target is still the
     * focused one, and is removed when the stream fails — the batch
     * fallback then takes over after Stop.
     */
    private fun onStreamEvent(event: StreamEvent) {
        when (event) {
            StreamEvent.Live -> if (activeRecording != null) {
                statusView?.setText(R.string.keyboard_streaming)
            }
            is StreamEvent.Partial -> {
                val target = composingTarget ?: return
                val connection = currentInputConnection
                if (targetGuard.isCurrent(target, connection)) {
                    // The server's partial grows over the whole session, so
                    // each one replaces the composing region entirely.
                    connection.setComposingText(event.text, 1)
                } else {
                    // The editor changed under the recording; the composing
                    // region died with the old editor's connection.
                    composingTarget = null
                }
            }
            is StreamEvent.Interrupted -> {
                clearComposingText()
                statusView?.text = getString(R.string.keyboard_stream_interrupted, event.reason)
            }
        }
    }

    /**
     * Removes any composing region this keyboard owns, without committing
     * its partial text, so a failed stream leaves the editor unchanged.
     */
    private fun clearComposingText() {
        val target = composingTarget ?: return
        composingTarget = null
        val connection = currentInputConnection
        if (targetGuard.isCurrent(target, connection)) {
            connection.setComposingText("", 0)
            connection.finishComposingText()
        }
    }

    private fun stopAndQueueRecording() {
        val recording = activeRecording ?: return
        // Capture immutable request-local state before stopping. A new editor
        // or recording may replace the fields below while this request waits.
        val requestTarget = recordingTarget
        val requestGeneration = activeRequestGeneration
        val session = streamSession
        activeRecording = null
        recordingTarget = null
        activeRequestGeneration = 0L
        streamSession = null
        recordButton?.setText(R.string.keyboard_record)

        // The capture settles inline on this main thread in the common
        // case; when the microphone refuses to stop, the outcome is
        // delivered later, still on the main thread, so input teardown and
        // onDestroy never block on the forced-release wait.
        capture.stop { result ->
            settleStoppedRecording(recording, requestTarget, requestGeneration, session, result)
        }
    }

    private fun settleStoppedRecording(
        recording: Recording,
        requestTarget: InputTargetGuard.Snapshot<InputConnection>?,
        requestGeneration: Long,
        session: StreamSession?,
        result: CaptureResult,
    ) {
        when (result) {
            is CaptureResult.Completed -> {
                // The WAV is finalized and durable before any network use.
                val finalized = runCatching {
                    application.recordings.commitAudio(recording, result.durationSeconds)
                }.getOrElse {
                    session?.close()
                    application.recordings.markFailed(recording.id, "Unable to finalize the private WAV recording")
                    statusView?.setText(R.string.recording_finalize_error)
                    return
                }
                statusView?.setText(
                    if (result.cappedAtLimit) R.string.recording_capped else R.string.keyboard_sending,
                )
                val config = application.backendSettings.load()
                val settled: (Recording) -> Unit = { completed ->
                    onTranscriptionSettled(completed, requestTarget, requestGeneration)
                }
                if (session != null) {
                    // Commit the live stream; the coordinator falls back to
                    // the batch upload of the same saved WAV on any failure.
                    application.transcription.finishStreaming(session, finalized.id, config, settled)
                } else {
                    application.transcription.transcribe(finalized.id, config, settled)
                }
            }
            is CaptureResult.Failed -> {
                // Nothing will be transcribed; drop the server session and
                // any composing region the live stream left in the editor.
                session?.close()
                clearComposingText()
                runCatching { application.recordings.markFailed(recording.id, result.message) }
                statusView?.text = result.message
            }
            CaptureResult.AlreadyStopped -> {
                session?.close()
                clearComposingText()
                statusView?.setText(R.string.recording_already_stopped)
            }
        }
    }

    /**
     * One settlement for both the batch and the streamed path: the audio and
     * exact transcript are already durable, so a superseded request may
     * update only the store.
     */
    private fun onTranscriptionSettled(
        completed: Recording,
        requestTarget: InputTargetGuard.Snapshot<InputConnection>?,
        requestGeneration: Long,
    ) {
        if (!requestGuard.isCurrent(requestGeneration)) return
        if (completed.status != RecordingStatus.TRANSCRIBED || completed.rawTranscript == null) {
            // Any composing region is stale by now; the editor must not
            // keep partial text the failed stream produced.
            clearComposingText()
            statusView?.setText(R.string.keyboard_transcription_failed)
            return
        }
        val text = completed.rawTranscript
        if (completed.provenance == TranscriptionProvenance.LIVE_STREAM) {
            settleLiveFinal(text, requestTarget)
            return
        }
        // Batch result (first attempt or fallback after a stream failure):
        // the explicit Insert flow, with any leftover composing removed.
        clearComposingText()
        if (requestTarget != null) {
            val current = currentInputConnection
            if (targetGuard.isCurrent(requestTarget, current)) {
                readyTranscript = ReadyTranscript(text, requestTarget)
                transcriptView?.visibility = View.VISIBLE
                transcriptView?.text = text
                insertButton?.visibility = View.VISIBLE
                statusView?.setText(R.string.keyboard_ready_to_insert)
            } else {
                transcriptView?.visibility = View.VISIBLE
                transcriptView?.text = text
                insertButton?.visibility = View.GONE
                statusView?.setText(R.string.keyboard_target_changed)
            }
        } else {
            statusView?.setText(R.string.keyboard_transcription_failed)
        }
    }

    /**
     * Final text of a healthy live stream. The single asynchronous
     * `commitText` of the composing path — and only into the editor the
     * composing region belongs to.
     */
    private fun settleLiveFinal(
        text: String,
        requestTarget: InputTargetGuard.Snapshot<InputConnection>?,
    ) {
        val target = composingTarget
        val connection = currentInputConnection
        when {
            target != null && targetGuard.isCurrent(target, connection) -> {
                // commitText replaces the composing region and finishes
                // composing in one step.
                connection.commitText(text, 1)
                composingTarget = null
                transcriptView?.visibility = View.VISIBLE
                transcriptView?.text = text
                insertButton?.visibility = View.GONE
                statusView?.setText(R.string.keyboard_inserted)
            }
            requestTarget != null && targetGuard.isCurrent(requestTarget, connection) -> {
                // The stream stayed healthy but composing never started (or
                // was cleared when the editor flickered): degrade to the
                // explicit Insert flow rather than dropping the text.
                readyTranscript = ReadyTranscript(text, requestTarget)
                transcriptView?.visibility = View.VISIBLE
                transcriptView?.text = text
                insertButton?.visibility = View.VISIBLE
                statusView?.setText(R.string.keyboard_ready_to_insert)
            }
            else -> {
                composingTarget = null
                transcriptView?.visibility = View.VISIBLE
                transcriptView?.text = text
                insertButton?.visibility = View.GONE
                statusView?.setText(R.string.keyboard_target_changed)
            }
        }
    }

    private fun insertReadyTranscript() {
        val ready = readyTranscript ?: return
        val connection = currentInputConnection
        if (!targetGuard.isCurrent(ready.target, connection)) {
            readyTranscript = null
            insertButton?.visibility = View.GONE
            statusView?.setText(R.string.keyboard_target_changed)
            return
        }
        // This is the sole insertion path, reached only from the explicit
        // Insert button. No editor text or package name is read or persisted.
        connection.commitText(ready.text, 1)
        readyTranscript = null
        insertButton?.visibility = View.GONE
        statusView?.setText(R.string.keyboard_inserted)
    }

    /**
     * The IME window also draws edge to edge with targetSdk 35, so without
     * extra padding the navigation bar would overlap the bottom button row.
     * The base padding is captured once because insets can be dispatched
     * more than once.
     */
    private fun applyWindowInsets(view: View) {
        val baseLeft = view.paddingLeft
        val baseTop = view.paddingTop
        val baseRight = view.paddingRight
        val baseBottom = view.paddingBottom
        ViewCompat.setOnApplyWindowInsetsListener(view) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(
                baseLeft + systemBars.left,
                baseTop,
                baseRight + systemBars.right,
                baseBottom + systemBars.bottom,
            )
            WindowInsetsCompat.CONSUMED
        }
    }

    private fun renderIdle() {
        if (activeRecording == null) {
            recordButton?.setText(R.string.keyboard_record)
            if (readyTranscript == null) insertButton?.visibility = View.GONE
            if (statusView?.text.isNullOrEmpty()) statusView?.setText(R.string.keyboard_ready)
        }
    }

    private data class ReadyTranscript(
        val text: String,
        val target: InputTargetGuard.Snapshot<InputConnection>,
    )
}
