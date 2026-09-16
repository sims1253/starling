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
import dev.starling.mobile.audio.CaptureResult
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
import dev.starling.mobile.ui.InputTargetGuard
import dev.starling.mobile.ui.RequestGenerationGuard

/**
 * Lightweight voice keyboard. It never reads surrounding editor text and
 * never inserts a transcript from an asynchronous callback: insertion is a
 * separate, explicit user action guarded by the original InputConnection.
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
        // Invalidate any earlier request whose callback may still be queued.
        val requestGeneration = requestGuard.begin()
        val recording = runCatching { application.recordings.create() }.getOrElse {
            statusView?.setText(R.string.recording_storage_error)
            return
        }
        val error = capture.start(this, application.recordings.partialFile(recording))
        if (error != null) {
            runCatching { application.recordings.markFailed(recording.id, error) }
            statusView?.text = error
            return
        }
        activeRecording = recording
        recordingTarget = target
        activeRequestGeneration = requestGeneration
        readyTranscript = null
        transcriptView?.visibility = View.GONE
        transcriptView?.text = null
        insertButton?.visibility = View.GONE
        recordButton?.setText(R.string.keyboard_stop)
        statusView?.setText(R.string.keyboard_recording)
    }

    private fun stopAndQueueRecording() {
        val recording = activeRecording ?: return
        // Capture immutable request-local state before stopping. A new editor
        // or recording may replace the fields below while this request waits.
        val requestTarget = recordingTarget
        val requestGeneration = activeRequestGeneration
        activeRecording = null
        recordingTarget = null
        activeRequestGeneration = 0L
        recordButton?.setText(R.string.keyboard_record)

        when (val result = capture.stop()) {
            is CaptureResult.Completed -> {
                val finalized = runCatching {
                    application.recordings.commitAudio(recording, result.durationSeconds)
                }.getOrElse {
                    application.recordings.markFailed(recording.id, "Unable to finalize the private WAV recording")
                    statusView?.setText(R.string.recording_finalize_error)
                    return
                }
                statusView?.setText(R.string.keyboard_sending)
                val config = application.backendSettings.load()
                application.transcription.transcribe(finalized.id, config) { completed ->
                    // The audio and exact transcript are already durable. If a
                    // newer recording started, this response may update only
                    // the store; it must not change this keyboard's UI.
                    if (!requestGuard.isCurrent(requestGeneration)) return@transcribe
                    if (completed.status == RecordingStatus.TRANSCRIBED &&
                        completed.rawTranscript != null && requestTarget != null
                    ) {
                        val snapshot = requestTarget
                        val current = currentInputConnection
                        if (targetGuard.isCurrent(snapshot, current)) {
                            readyTranscript = ReadyTranscript(completed.rawTranscript, snapshot)
                            transcriptView?.visibility = View.VISIBLE
                            transcriptView?.text = completed.rawTranscript
                            insertButton?.visibility = View.VISIBLE
                            statusView?.setText(R.string.keyboard_ready_to_insert)
                        } else {
                            transcriptView?.visibility = View.VISIBLE
                            transcriptView?.text = completed.rawTranscript
                            insertButton?.visibility = View.GONE
                            statusView?.setText(R.string.keyboard_target_changed)
                        }
                    } else {
                        statusView?.setText(R.string.keyboard_transcription_failed)
                    }
                }
            }
            is CaptureResult.Failed -> {
                runCatching { application.recordings.markFailed(recording.id, result.message) }
                statusView?.text = result.message
            }
            CaptureResult.AlreadyStopped -> statusView?.setText(R.string.recording_already_stopped)
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
