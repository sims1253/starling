package dev.starling.mobile

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
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
import dev.starling.mobile.audio.AudioCapture
import dev.starling.mobile.audio.CaptureResult
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
import dev.starling.mobile.network.BackendConfig
import dev.starling.mobile.network.BackendProtocol
import dev.starling.mobile.network.EndpointPolicy
import dev.starling.mobile.network.EndpointValidation
import java.text.DateFormat
import java.util.Date

class MainActivity : Activity() {
    private lateinit var endpointInput: EditText
    private lateinit var allowHttpInput: CheckBox
    private lateinit var protocolInput: RadioGroup
    private lateinit var openAiProtocolInput: RadioButton
    private lateinit var modelInput: EditText
    private lateinit var endpointMessage: TextView
    private lateinit var recordingMessage: TextView
    private lateinit var recordButton: Button
    private lateinit var recordingsContainer: LinearLayout

    private val capture = AudioCapture()
    private val application by lazy { starlingApplication() }
    private var activeRecording: Recording? = null
    private var awaitingPermission = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        endpointInput = findViewById(R.id.endpoint_input)
        allowHttpInput = findViewById(R.id.allow_http_input)
        protocolInput = findViewById(R.id.protocol_input)
        openAiProtocolInput = findViewById(R.id.protocol_openai)
        modelInput = findViewById(R.id.model_input)
        endpointMessage = findViewById(R.id.endpoint_message)
        recordingMessage = findViewById(R.id.recording_message)
        recordButton = findViewById(R.id.record_button)
        recordingsContainer = findViewById(R.id.recordings_container)

        val config = application.backendSettings.load()
        endpointInput.setText(config.endpoint)
        allowHttpInput.isChecked = config.allowTrustedLanHttp
        openAiProtocolInput.isChecked = config.protocol == BackendProtocol.OPENAI
        findViewById<RadioButton>(R.id.protocol_starling).isChecked =
            config.protocol == BackendProtocol.STARLING
        modelInput.setText(config.model)
        protocolInput.setOnCheckedChangeListener { _, _ -> updateProtocolFields() }
        updateProtocolFields()
        findViewById<Button>(R.id.save_endpoint_button).setOnClickListener { saveEndpoint() }
        recordButton.setOnClickListener {
            if (activeRecording == null) requestOrStartRecording() else stopAndQueueRecording()
        }
        findViewById<Button>(R.id.open_keyboard_button).setOnClickListener {
            startActivity(Intent("android.settings.INPUT_METHOD_SETTINGS"))
        }
        findViewById<Button>(R.id.open_voice_input_button).setOnClickListener {
            startActivity(Intent(android.provider.Settings.ACTION_VOICE_INPUT_SETTINGS))
        }

        recordingMessage.text = getString(R.string.ready_to_record)
        refreshRecordings()
    }

    override fun onStop() {
        // A backgrounded Activity should never keep the microphone open. The
        // finalized file stays in app-private storage and can be retried later.
        if (activeRecording != null) stopAndQueueRecording()
        super.onStop()
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
        val protocol = if (openAiProtocolInput.isChecked) {
            BackendProtocol.OPENAI
        } else {
            BackendProtocol.STARLING
        }
        val config = BackendConfig(
            endpoint = endpointInput.text.toString(),
            allowTrustedLanHttp = allowHttpInput.isChecked,
            protocol = protocol,
            model = modelInput.text.toString().trim(),
        )
        if (config.protocol == BackendProtocol.OPENAI && config.model.isEmpty()) {
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

    private fun updateProtocolFields() {
        if (!::modelInput.isInitialized) return
        modelInput.visibility = if (openAiProtocolInput.isChecked) View.VISIBLE else View.GONE
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
        val error = capture.start(application.recordings.partialFile(recording))
        if (error != null) {
            runCatching { application.recordings.markFailed(recording.id, error) }
            recordingMessage.text = error
            refreshRecordings()
            return
        }
        activeRecording = recording
        recordButton.setText(R.string.stop_and_transcribe)
        recordingMessage.setText(R.string.recording_now)
    }

    private fun stopAndQueueRecording() {
        val recording = activeRecording ?: return
        activeRecording = null
        recordButton.setText(R.string.start_recording)

        when (val result = capture.stop()) {
            is CaptureResult.Completed -> {
                val finalized = runCatching {
                    application.recordings.commitAudio(recording, result.durationSeconds)
                }.getOrElse {
                    application.recordings.markFailed(recording.id, "Unable to finalize the private WAV recording")
                    recordingMessage.setText(R.string.recording_finalize_error)
                    refreshRecordings()
                    return
                }
                recordingMessage.setText(R.string.sending_recording)
                val config = application.backendSettings.load()
                application.transcription.transcribe(finalized.id, config) {
                    recordingMessage.setText(
                        if (it.status == RecordingStatus.TRANSCRIBED) {
                            R.string.transcription_saved
                        } else {
                            R.string.transcription_failed_retry
                        },
                    )
                    refreshRecordings()
                }
                refreshRecordings()
            }
            is CaptureResult.Failed -> {
                runCatching { application.recordings.markFailed(recording.id, result.message) }
                recordingMessage.text = result.message
                refreshRecordings()
            }
            CaptureResult.AlreadyStopped -> {
                recordingMessage.setText(R.string.recording_already_stopped)
            }
        }
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
            RecordingStatus.TRANSCRIBED -> getString(R.string.status_transcribed)
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

        retry.visibility = if (recording.status == RecordingStatus.PENDING ||
            recording.status == RecordingStatus.FAILED ||
            recording.status == RecordingStatus.TRANSCRIBING
        ) View.VISIBLE else View.GONE
        retry.setOnClickListener {
            val config = application.backendSettings.load()
            recordingMessage.setText(R.string.sending_recording)
            application.transcription.transcribe(recording.id, config) {
                recordingMessage.setText(
                    if (it.status == RecordingStatus.TRANSCRIBED) {
                        R.string.transcription_saved
                    } else {
                        R.string.transcription_failed_retry
                    },
                )
                refreshRecordings()
            }
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
        private const val REQUEST_RECORD_AUDIO = 4001
    }
}
