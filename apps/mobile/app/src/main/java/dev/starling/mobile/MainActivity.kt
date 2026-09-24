package dev.starling.mobile

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.TextView
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import dev.starling.mobile.audio.AudioCapture
import dev.starling.mobile.audio.AudioChunkListener
import dev.starling.mobile.audio.CaptureResult
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
import dev.starling.mobile.data.TranscriptionProvenance
import dev.starling.mobile.engine.ModelCatalog
import dev.starling.mobile.engine.OnDeviceEngine
import dev.starling.mobile.network.BackendConfig
import dev.starling.mobile.network.EndpointPolicy
import dev.starling.mobile.network.EndpointValidation
import dev.starling.mobile.network.StreamEvent
import dev.starling.mobile.network.StreamSession
import dev.starling.mobile.network.TranscriptionEngine
import java.text.DateFormat
import java.util.Date
import kotlin.concurrent.thread

class MainActivity : Activity() {
    private lateinit var endpointInput: EditText
    private lateinit var allowHttpInput: CheckBox
    private lateinit var modelInput: EditText
    private lateinit var engineInput: RadioGroup
    private lateinit var engineOnDeviceInput: RadioButton
    private lateinit var onDeviceStatus: TextView
    private lateinit var downloadModelButton: Button
    private lateinit var endpointMessage: TextView
    private lateinit var recordingMessage: TextView
    private lateinit var liveTranscript: TextView
    private lateinit var recordButton: Button
    private lateinit var recordingsContainer: LinearLayout

    private val capture = AudioCapture()
    private val application by lazy { starlingApplication() }
    private var activeRecording: Recording? = null
    private var awaitingPermission = false

    // Written on the main thread. Read by the capture worker through the
    // chunk listener, so a stop that nulls it racing an escalated worker is
    // still safe: a stale session simply ignores the audio once finished.
    @Volatile
    private var streamSession: StreamSession? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Edge-to-edge is enforced with targetSdk 35 and the deprecated
        // status/navigation bar color items no longer reserve any space. Pad
        // the scroll root for the system bars, the soft keyboard (the window
        // is no longer resized for it), and the display cutout so the title
        // clears the status bar and the controls clear the gesture bar.
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main_root)) { view, windowInsets ->
            val bars = windowInsets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.ime(),
            )
            val cutout = windowInsets.displayCutout
            view.setPadding(
                maxOf(bars.left, cutout?.safeInsetLeft ?: 0),
                maxOf(bars.top, cutout?.safeInsetTop ?: 0),
                maxOf(bars.right, cutout?.safeInsetRight ?: 0),
                maxOf(bars.bottom, cutout?.safeInsetBottom ?: 0),
            )
            WindowInsetsCompat.CONSUMED
        }

        endpointInput = findViewById(R.id.endpoint_input)
        allowHttpInput = findViewById(R.id.allow_http_input)
        modelInput = findViewById(R.id.model_input)
        engineInput = findViewById(R.id.engine_input)
        engineOnDeviceInput = findViewById(R.id.engine_on_device)
        onDeviceStatus = findViewById(R.id.on_device_status)
        endpointMessage = findViewById(R.id.endpoint_message)
        recordingMessage = findViewById(R.id.recording_message)
        liveTranscript = findViewById(R.id.live_transcript)
        recordButton = findViewById(R.id.record_button)
        recordingsContainer = findViewById(R.id.recordings_container)

        val config = application.backendSettings.load()
        endpointInput.setText(config.endpoint)
        allowHttpInput.isChecked = config.allowTrustedLanHttp
        modelInput.setText(config.model)
        engineOnDeviceInput.isChecked = config.engine == TranscriptionEngine.ON_DEVICE
        findViewById<RadioButton>(R.id.engine_server).isChecked =
            config.engine == TranscriptionEngine.REMOTE
        findViewById<Button>(R.id.save_endpoint_button).setOnClickListener { saveEndpoint() }
        findViewById<Button>(R.id.import_model_button).setOnClickListener { importModel() }
        downloadModelButton = findViewById(R.id.download_model_button)
        downloadModelButton.setOnClickListener {
            val downloads = application.modelDownloads
            if (downloads.isRunning) downloads.cancel() else downloads.start(ModelCatalog.RECOMMENDED_PARAKEET)
        }
        recordButton.setOnClickListener {
            if (activeRecording == null) requestOrStartRecording() else stopAndQueueRecording()
        }
        findViewById<Button>(R.id.open_keyboard_button).setOnClickListener {
            startActivity(Intent("android.settings.INPUT_METHOD_SETTINGS"))
        }
        recordingMessage.text = getString(R.string.ready_to_record)
        refreshOnDeviceStatus()
        refreshRecordings()
    }

    override fun onResume() {
        super.onResume()
        // Recordings can appear outside this Activity's own actions (voice
        // keyboard, recognition service, a transcription that finished while
        // backgrounded), so the list is refreshed on every resume.
        refreshRecordings()
    }

    override fun onStart() {
        super.onStart()
        application.modelDownloads.addListener(downloadListener)
    }

    override fun onStop() {
        application.modelDownloads.removeListener(downloadListener)
        // A backgrounded Activity should never keep the microphone open. The
        // finalized file stays in app-private storage and can be retried later.
        if (activeRecording != null) stopAndQueueRecording()
        super.onStop()
    }

    // The download is app-scoped (it survives rotation and leaving the
    // screen); this Activity only renders its state while visible.
    private val downloadListener: (ModelDownloadController.State) -> Unit = { state ->
        renderDownload(state)
    }

    private fun renderDownload(state: ModelDownloadController.State) {
        val spec = ModelCatalog.RECOMMENDED_PARAKEET
        val totalMb = mb(spec.sizeBytes)
        when (state) {
            is ModelDownloadController.State.Running -> {
                downloadModelButton.setText(R.string.cancel_download)
                onDeviceStatus.text = if (state.verifying) {
                    getString(R.string.on_device_verifying)
                } else {
                    val percent = if (state.total > 0) (state.bytes * 100 / state.total).toInt() else 0
                    getString(R.string.on_device_downloading, percent, mb(state.bytes), totalMb)
                }
                return
            }
            is ModelDownloadController.State.Finished -> when (val result = state.result) {
                // Re-delivered on every onStart, so no one-shot message here.
                is OnDeviceEngine.ImportResult.Imported -> refreshOnDeviceStatus()
                is OnDeviceEngine.ImportResult.Rejected -> {
                    Log.w(TAG, "downloaded model rejected at ${result.stage}: ${result.reason}")
                    onDeviceStatus.text = result.reason
                }
            }
            is ModelDownloadController.State.Failed ->
                onDeviceStatus.text = getString(R.string.on_device_download_failed, state.reason)
            ModelDownloadController.State.Paused ->
                onDeviceStatus.setText(R.string.on_device_download_paused)
            ModelDownloadController.State.Idle -> Unit
        }
        val partialMb = mb(application.modelDownloads.resumableBytes(spec))
        downloadModelButton.text = if (partialMb > 0) {
            getString(R.string.resume_download, partialMb, totalMb)
        } else {
            getString(R.string.download_model, totalMb)
        }
    }

    override fun onDestroy() {
        if (activeRecording != null) stopAndQueueRecording()
        super.onDestroy()
    }

    private fun saveEndpoint(): BackendConfig? {
        val config = currentConfig() ?: return null
        application.backendSettings.save(config)
        endpointMessage.setText(R.string.endpoint_saved)
        return config
    }

    private fun currentConfig(): BackendConfig? {
        val config = BackendConfig(
            endpoint = endpointInput.text.toString(),
            allowTrustedLanHttp = allowHttpInput.isChecked,
            model = modelInput.text.toString().trim(),
            engine = if (engineOnDeviceInput.isChecked) {
                TranscriptionEngine.ON_DEVICE
            } else {
                TranscriptionEngine.REMOTE
            },
        )
        if (config.engine == TranscriptionEngine.REMOTE && config.model.isEmpty()) {
            endpointMessage.setText(R.string.model_required)
            return null
        }
        return when (val validation = EndpointPolicy.validate(config.endpoint, config.allowTrustedLanHttp)) {
            is EndpointValidation.Invalid -> {
                endpointMessage.text = validation.message
                null
            }
            is EndpointValidation.Valid -> config.copy(endpoint = validation.endpoint)
        }
    }

    private fun importModel() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            // No registered MIME type exists for .gguf, and providers label
            // it inconsistently (octet-stream, empty, or a guess), so any
            // file is offered; the import validates the GGUF itself.
            type = "*/*"
        }
        startActivityForResult(intent, REQUEST_IMPORT_MODEL)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQUEST_IMPORT_MODEL || resultCode != RESULT_OK) return
        val uri: Uri = data?.data ?: run {
            onDeviceStatus.setText(R.string.on_device_file_missing)
            return
        }
        onDeviceStatus.setText(R.string.on_device_importing)
        val resolver = contentResolver
        thread {
            val opened = runCatching { resolver.openInputStream(uri) }
            val input = opened.getOrNull()
            val result = if (input == null) {
                val detail = opened.exceptionOrNull()?.localizedMessage?.takeIf(String::isNotBlank)
                OnDeviceEngine.ImportResult.Rejected(
                    "The selected file could not be opened" +
                        (detail?.let { ": $it" } ?: "."),
                    OnDeviceEngine.ImportStage.OPEN,
                )
            } else {
                runCatching { application.onDeviceEngine.importModel(input) }.getOrElse { error ->
                    Log.e(TAG, "model import failed unexpectedly", error)
                    OnDeviceEngine.ImportResult.Rejected(
                        "The model could not be imported: ${error.message ?: error::class.java.simpleName}",
                        OnDeviceEngine.ImportStage.COPY,
                    )
                }
            }
            runOnUiThread {
                // The copy can outlive a user who navigated away
                // mid-import; posting into a destroyed activity leaks it
                // and risks a crash, so a dead activity drops the update.
                if (isDestroyed || isFinishing) return@runOnUiThread
                when (result) {
                    is OnDeviceEngine.ImportResult.Imported -> {
                        refreshOnDeviceStatus()
                        recordingMessage.setText(R.string.on_device_imported)
                    }
                    is OnDeviceEngine.ImportResult.Rejected -> {
                        Log.w(TAG, "model import rejected at ${result.stage}: ${result.reason}")
                        // Keep the failure beside the Import button. A message
                        // in the recording section is off screen during import.
                        onDeviceStatus.text = result.reason
                        recordingMessage.text = result.reason
                    }
                }
            }
        }
    }

    private fun refreshOnDeviceStatus() {
        val engine = application.onDeviceEngine
        onDeviceStatus.text = when {
            engine.hasModel() -> {
                val sizeMb = mb(engine.modelSizeBytes())
                getString(R.string.on_device_model_present, sizeMb)
            }
            else -> getString(R.string.on_device_status_no_model)
        }
    }

    private fun requestOrStartRecording() {
        val config = currentConfig() ?: return
        // Use what is visible in the form for this recording. This also makes
        // recording safe when the user edits the endpoint and skips the Save
        // button before tapping Record.
        runCatching { application.backendSettings.save(config) }
            .onFailure {
                endpointMessage.setText(R.string.endpoint_save_error)
                return
            }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            awaitingPermission = true
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO)
            return
        }
        beginCapture()
    }

    private fun beginCapture() {
        awaitingPermission = false
        val recording = runCatching { application.recordings.create() }.getOrElse {
            recordingMessage.text = getString(R.string.recording_storage_error)
            return
        }
        // The live stream is an observer of the capture, never a gate on it:
        // beginStreaming returns null whenever the configuration or endpoint
        // does not support streaming, and the recording proceeds as before.
        val config = application.backendSettings.load()
        var session: StreamSession? = null
        session = application.transcription.beginStreaming(config) { event ->
            // Events from a superseded session must not rewrite the views of
            // the recording that replaced it.
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
            recordingMessage.text = error
            refreshRecordings()
            return
        }
        activeRecording = recording
        streamSession = session
        recordButton.setText(R.string.stop_and_transcribe)
        liveTranscript.visibility = View.GONE
        liveTranscript.text = null
        recordingMessage.setText(
            if (session == null) R.string.recording_now else R.string.streaming_connecting,
        )
    }

    /**
     * Live-stream events, already marshalled to the main thread by the
     * coordinator. An interrupted stream never interrupts the recording:
     * the message explains that the stop path takes over transcription.
     */
    private fun onStreamEvent(event: StreamEvent) {
        if (isDestroyed || isFinishing) return
        when (event) {
            StreamEvent.Live -> recordingMessage.setText(R.string.streaming_live)
            is StreamEvent.Partial -> {
                liveTranscript.visibility = View.VISIBLE
                // The server's partial is a growing transcript of the whole
                // session so far, so it replaces the previous text.
                liveTranscript.text = event.text
            }
            is StreamEvent.Interrupted -> recordingMessage.text =
                getString(R.string.streaming_interrupted, event.reason)
        }
    }

    private fun stopAndQueueRecording() {
        val recording = activeRecording ?: return
        activeRecording = null
        val session = streamSession
        streamSession = null
        recordButton.setText(R.string.start_recording)

        // The capture settles inline on this main thread in the common
        // case; when the microphone refuses to stop, the outcome is
        // delivered later, still on the main thread, so onStop/onDestroy
        // never block on the forced-release wait.
        capture.stop { result -> settleStoppedRecording(recording, session, result) }
    }

    private fun settleStoppedRecording(
        recording: Recording,
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
                    updateRecordingViews {
                        recordingMessage.setText(R.string.recording_finalize_error)
                        refreshRecordings()
                    }
                    return
                }
                val config = application.backendSettings.load()
                // With a live session, commit the stream and store its final
                // transcript; the coordinator falls back to this same batch
                // upload of the saved WAV whenever the stream failed.
                val queued = if (session != null) {
                    application.transcription.finishStreaming(session, finalized.id, config, ::onTranscriptionSettled)
                } else {
                    application.transcription.transcribe(finalized.id, config, ::onTranscriptionSettled)
                }
                if (queued) updateRecordingViews { recordingMessage.setText(R.string.sending_recording) }
                updateRecordingViews { refreshRecordings() }
            }
            is CaptureResult.Failed -> {
                // Nothing will be transcribed; the server session and the
                // stale live partial go with it.
                session?.close()
                updateRecordingViews { liveTranscript.visibility = View.GONE }
                runCatching { application.recordings.markFailed(recording.id, result.message) }
                updateRecordingViews {
                    recordingMessage.text = result.message
                    refreshRecordings()
                }
            }
            CaptureResult.AlreadyStopped -> {
                session?.close()
                updateRecordingViews {
                    liveTranscript.visibility = View.GONE
                    recordingMessage.setText(R.string.recording_already_stopped)
                }
            }
        }
    }

    private fun onTranscriptionSettled(completed: Recording) {
        // The Activity may have been destroyed (for example by a rotation)
        // while the request was in flight; the store settlement already ran.
        if (isDestroyed || isFinishing) return
        if (completed.status == RecordingStatus.TRANSCRIBED) {
            // The verbatim final is in the recordings list now; a leftover
            // live partial next to it could read as the final text.
            liveTranscript.visibility = View.GONE
            recordingMessage.setText(R.string.transcription_saved)
        } else {
            // Keep the last partial visible: it is the only text the user
            // has while the recording waits for a retry.
            recordingMessage.setText(R.string.transcription_failed_retry)
        }
        refreshRecordings()
    }

    /**
     * An escalated stop settles after a delay, so its outcome can arrive
     * once the Activity is destroyed. The store settlement beside each of
     * these calls must still run — the finalized audio stays retryable —
     * but the views do not outlive the Activity.
     */
    private fun updateRecordingViews(update: () -> Unit) {
        if (isDestroyed || isFinishing) return
        update()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != REQUEST_RECORD_AUDIO || !awaitingPermission) return
        awaitingPermission = false
        if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            beginCapture()
        } else {
            recordingMessage.setText(R.string.microphone_permission_required)
        }
    }

    private fun refreshRecordings() {
        if (!::recordingsContainer.isInitialized) return
        recordingsContainer.removeAllViews()
        val recordings = application.recordings.list()
        if (recordings.isEmpty()) {
            val empty = TextView(this).apply {
                setText(R.string.no_recordings)
                setPadding(0, 12, 0, 12)
            }
            recordingsContainer.addView(empty)
            return
        }
        val inflater = LayoutInflater.from(this)
        recordings.forEach { recording ->
            val row = inflater.inflate(R.layout.recording_row, recordingsContainer, false)
            bindRecordingRow(row, recording)
            recordingsContainer.addView(row)
        }
    }

    private fun bindRecordingRow(row: View, recording: Recording) {
        val title = row.findViewById<TextView>(R.id.recording_title)
        val status = row.findViewById<TextView>(R.id.recording_status)
        val transcript = row.findViewById<TextView>(R.id.recording_transcript)
        val retry = row.findViewById<Button>(R.id.retry_recording_button)
        val delete = row.findViewById<Button>(R.id.delete_recording_button)

        title.text = DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT)
            .format(Date(recording.createdAtMillis))
        status.text = when (recording.status) {
            RecordingStatus.RECORDING -> getString(R.string.status_interrupted)
            RecordingStatus.PENDING -> getString(R.string.status_pending)
            RecordingStatus.TRANSCRIBING -> getString(R.string.status_transcribing)
            RecordingStatus.TRANSCRIBED -> getString(
                if (recording.provenance == TranscriptionProvenance.LIVE_STREAM) {
                    R.string.status_transcribed_streamed
                } else {
                    R.string.status_transcribed
                },
            )
            RecordingStatus.FAILED -> getString(R.string.status_failed)
        }
        if (recording.rawTranscript != null) {
            transcript.visibility = View.VISIBLE
            // The exact server text is shown and retained. There is no cleanup
            // pass that could remove words such as "not", "like", or "auth".
            transcript.text = recording.rawTranscript
        } else {
            transcript.visibility = View.GONE
        }
        recording.errorMessage?.let {
            status.append(" — ")
            status.append(it)
        }

        // Retry is hidden while a transcription for this recording is still
        // in flight, so a double tap cannot queue a second request. A
        // TRANSCRIBING row without an active request (for example after a
        // process restart) still offers Retry to recover it.
        retry.visibility = if ((recording.status == RecordingStatus.PENDING ||
            recording.status == RecordingStatus.FAILED ||
            recording.status == RecordingStatus.TRANSCRIBING) &&
            !application.transcription.isActive(recording.id)
        ) View.VISIBLE else View.GONE
        retry.setOnClickListener {
            val config = application.backendSettings.load()
            val queued = application.transcription.transcribe(recording.id, config) {
                // The Activity may have been destroyed (for example by a
                // rotation) while the request was in flight.
                if (isDestroyed || isFinishing) return@transcribe
                recordingMessage.setText(
                    if (it.status == RecordingStatus.TRANSCRIBED) {
                        R.string.transcription_saved
                    } else {
                        R.string.transcription_failed_retry
                    },
                )
                refreshRecordings()
            }
            if (queued) recordingMessage.setText(R.string.sending_recording)
            refreshRecordings()
        }
        delete.setOnClickListener {
            AlertDialog.Builder(this)
                .setTitle(R.string.delete_recording_title)
                .setMessage(R.string.delete_recording_message)
                .setNegativeButton(android.R.string.cancel, null)
                .setPositiveButton(R.string.delete_recording) { _, _ ->
                    runCatching { application.recordings.delete(recording.id) }
                        .onSuccess { refreshRecordings() }
                        .onFailure { recordingMessage.setText(R.string.delete_recording_error) }
                }
                .show()
        }
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val REQUEST_RECORD_AUDIO = 4001
        private const val REQUEST_IMPORT_MODEL = 4002
        // Decimal megabytes, as Hugging Face and file managers show sizes.
        private const val MB = 1_000_000L

        /** Bytes to decimal MB, rounded to nearest like Hugging Face's listing. */
        private fun mb(bytes: Long): Int = ((bytes + MB / 2) / MB).toInt()
    }
}
