package dev.starling.mobile

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.content.pm.PackageManager
import android.media.MediaPlayer
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.Spinner
import android.widget.TextView
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import dev.starling.mobile.audio.AudioCapture
import dev.starling.mobile.audio.AudioChunkListener
import dev.starling.mobile.audio.CaptureResult
import dev.starling.mobile.audio.WavWriter
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
import dev.starling.mobile.data.RetentionClass
import dev.starling.mobile.data.TranscriptRevision
import dev.starling.mobile.data.TranscriptSource
import dev.starling.mobile.data.TranscriptionProvenance
import dev.starling.mobile.engine.ModelCatalog
import dev.starling.mobile.engine.OnDeviceEngine
import dev.starling.mobile.network.BackendConfig
import dev.starling.mobile.network.BackendSettings
import dev.starling.mobile.network.EndpointPolicy
import dev.starling.mobile.network.EndpointValidation
import dev.starling.mobile.network.StreamEvent
import dev.starling.mobile.network.StreamSession
import dev.starling.mobile.network.TranscriptionEngine
import dev.starling.mobile.storage.AudioUpkeep
import dev.starling.mobile.storage.ClassLimits
import dev.starling.mobile.storage.DiskLevel
import dev.starling.mobile.storage.DiskPolicy
import dev.starling.mobile.storage.HoldReason
import dev.starling.mobile.storage.StorageSettings
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
    private lateinit var installedModelsHeading: TextView
    private lateinit var installedModelsHelp: TextView
    private lateinit var installedModelsContainer: LinearLayout
    private lateinit var endpointMessage: TextView
    private lateinit var recordingMessage: TextView
    private lateinit var liveTranscript: TextView
    private lateinit var recordButton: Button
    private lateinit var recordingsContainer: LinearLayout
    private lateinit var retentionReport: TextView

    private val capture = AudioCapture()
    private val application by lazy { starlingApplication() }
    private var activeRecording: Recording? = null
    private var awaitingPermission = false

    // Playback of one saved recording at a time, and the recording an
    // export document is being picked for.
    private var player: MediaPlayer? = null
    private var playingId: String? = null
    private var pendingExportId: String? = null

    // The format the export document was named for; it is what gets written.
    private var pendingExportFlac = false

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
        installedModelsHeading = findViewById(R.id.installed_models_heading)
        installedModelsHelp = findViewById(R.id.installed_models_help)
        installedModelsContainer = findViewById(R.id.installed_models)
        endpointMessage = findViewById(R.id.endpoint_message)
        recordingMessage = findViewById(R.id.recording_message)
        liveTranscript = findViewById(R.id.live_transcript)
        recordButton = findViewById(R.id.record_button)
        recordingsContainer = findViewById(R.id.recordings_container)
        retentionReport = findViewById(R.id.retention_report)
        bindRetentionLimits()

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
        engineInput.setOnCheckedChangeListener { _, _ ->
            // Download progress owns the status line while it runs.
            if (!application.modelDownloads.isRunning) refreshOnDeviceStatus()
        }
        recordingMessage.text = getString(R.string.ready_to_record)
        refreshOnDeviceStatus()
        refreshRecordings()
        if (savedInstanceState == null) handleKeyboardRequest(intent)
        // The document picker can outlive this instance (rotation, process
        // recreation); its result must still find the recording to export.
        pendingExportId = savedInstanceState?.getString(STATE_PENDING_EXPORT)
        pendingExportFlac = savedInstanceState?.getBoolean(STATE_PENDING_EXPORT_FLAC) ?: false
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        pendingExportId?.let { outState.putString(STATE_PENDING_EXPORT, it) }
        outState.putBoolean(STATE_PENDING_EXPORT_FLAC, pendingExportFlac)
    }

    /**
     * The voice keyboard cannot show a permission dialog, so it opens this
     * screen to ask for the microphone (and, on Android 13+, the take
     * notification that carries Stop). The keyboard opens it in a task of
     * its own, so once the microphone is allowed that task closes and the
     * user is back in the field they were dictating into.
     */
    private fun handleKeyboardRequest(intent: Intent?) {
        if (intent?.action != ACTION_REQUEST_MICROPHONE) return
        // The keyboard decides whether notifications are asked: once only, so
        // a refusal is not asked again on a later microphone hand-off.
        val askNotifications = intent.getBooleanExtra(EXTRA_ASK_NOTIFICATIONS, false)
        val missing = buildList {
            add(Manifest.permission.RECORD_AUDIO)
            if (Build.VERSION.SDK_INT >= 33 && askNotifications) add(Manifest.permission.POST_NOTIFICATIONS)
        }.filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (missing.isEmpty()) {
            finishAndRemoveTask()
            return
        }
        requestPermissions(missing.toTypedArray(), REQUEST_KEYBOARD_MICROPHONE)
    }

    override fun onResume() {
        super.onResume()
        // Recordings can appear outside this Activity's own actions (voice
        // keyboard, recognition service, a transcription that finished while
        // backgrounded), so the list is refreshed on every resume.
        refreshRecordings()
        // The recorder is a voice entry point: warm the selected local model.
        application.preloadOnDeviceModel()
    }

    override fun onStart() {
        super.onStart()
        application.modelDownloads.addListener(downloadListener)
        // A cleanup can remove or compress audio while this screen shows it.
        application.audioUpkeep.onReport = upkeepListener
        renderUpkeepReport()
    }

    override fun onStop() {
        application.modelDownloads.removeListener(downloadListener)
        if (application.audioUpkeep.onReport === upkeepListener) application.audioUpkeep.onReport = null
        // A backgrounded Activity should never keep the microphone open. The
        // finalized file stays in app-private storage and can be retried later.
        if (activeRecording != null) stopAndQueueRecording()
        stopPlayback()
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
        // Listed only once the download is not running: progress ticks
        // never touch the model directory.
        val models by lazy { application.onDeviceEngine.installedModels() }
        when (state) {
            is ModelDownloadController.State.Running -> {
                downloadModelButton.visibility = View.VISIBLE
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
                is OnDeviceEngine.ImportResult.Imported -> renderOnDeviceStatus(models)
                is OnDeviceEngine.ImportResult.Rejected -> {
                    Log.w(TAG, "downloaded model rejected at ${result.stage}: ${result.reason}")
                    onDeviceStatus.text = result.reason
                }
            }
            is ModelDownloadController.State.Failed ->
                onDeviceStatus.text = getString(R.string.on_device_download_failed, state.reason)
            ModelDownloadController.State.Paused ->
                onDeviceStatus.setText(R.string.on_device_download_paused)
            ModelDownloadController.State.Idle -> renderOnDeviceStatus(models)
        }
        renderInstalledModels(models)
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

    private val upkeepListener: (AudioUpkeep.Report) -> Unit = {
        runOnUiThread {
            if (!isDestroyed && !isFinishing) {
                renderUpkeepReport()
                refreshRecordings()
            }
        }
    }

    /**
     * The history-audio limits (#342), saved as soon as one changes. A
     * spinner reports its initial selection too; that matches what is
     * stored and saves nothing.
     */
    private fun bindRetentionLimits() {
        val policy = application.storageSettings.load()
        RetentionClass.entries.forEach { retentionClass ->
            val (ageId, sizeId) = when (retentionClass) {
                RetentionClass.STANDARD -> R.id.retention_standard_age to R.id.retention_standard_size
                RetentionClass.ARCHIVAL -> R.id.retention_archival_age to R.id.retention_archival_size
            }
            val age = findViewById<Spinner>(ageId)
            val size = findViewById<Spinner>(sizeId)
            val limits = policy.limits(retentionClass)
            age.setSelection(AGE_CHOICES.indexOf(limits.maxAgeDays).coerceAtLeast(0), false)
            size.setSelection(SIZE_CHOICES_MB.indexOf(limits.maxTotalMb).coerceAtLeast(0), false)
            val listener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                    // INVALID_POSITION (nothing selected) chooses nothing.
                    val agePosition = age.selectedItemPosition
                    val sizePosition = size.selectedItemPosition
                    if (agePosition !in AGE_CHOICES.indices || sizePosition !in SIZE_CHOICES_MB.indices) return
                    val stored = application.storageSettings.load().limits(retentionClass)
                    val chosen = ClassLimits(
                        maxAgeDays = StorageSettings.chosenLimit(AGE_CHOICES, agePosition, stored.maxAgeDays),
                        maxTotalMb = StorageSettings.chosenLimit(SIZE_CHOICES_MB, sizePosition, stored.maxTotalMb),
                    )
                    if (chosen == stored) return
                    runCatching { application.storageSettings.save(retentionClass, chosen) }
                        .onSuccess { application.audioUpkeep.schedule() }
                        .onFailure {
                            // Show the limits still in force.
                            val kept = application.storageSettings.load().limits(retentionClass)
                            age.setSelection(AGE_CHOICES.indexOf(kept.maxAgeDays).coerceAtLeast(0), false)
                            size.setSelection(SIZE_CHOICES_MB.indexOf(kept.maxTotalMb).coerceAtLeast(0), false)
                            recordingMessage.setText(R.string.retention_save_error)
                        }
                }

                override fun onNothingSelected(parent: AdapterView<*>?) = Unit
            }
            age.onItemSelectedListener = listener
            size.onItemSelectedListener = listener
        }
    }

    /** What the last history-audio cleanup did, in one line (the desktop's wording). */
    private fun renderUpkeepReport() {
        val report = application.audioUpkeep.lastReport
        val parts = buildList {
            if (report == null) return@buildList
            if (report.compressed > 0) {
                add(resources.getQuantityString(R.plurals.cleanup_compressed, report.compressed, report.compressed, mbCeil(report.savedBytes)))
            }
            val removed = report.retention.removed.size
            if (removed > 0) {
                add(resources.getQuantityString(R.plurals.cleanup_removed, removed, removed, mbCeil(report.retention.removedBytes)))
            }
            // Every due take that kept its audio, by why.
            for ((reason, plural) in listOf(
                HoldReason.RECENT to R.plurals.cleanup_recent,
                HoldReason.IN_USE to R.plurals.cleanup_in_use,
                HoldReason.UNTRANSCRIBED to R.plurals.cleanup_untranscribed,
            )) {
                val held = report.retention.held.count { it.reason == reason }
                if (held > 0) add(resources.getQuantityString(plural, held, held))
            }
            report.retention.overLimit.forEach { (retentionClass, bytes) ->
                add(getString(R.string.cleanup_over_limit, retentionClassName(retentionClass), mbCeil(bytes)))
            }
            val undeleted = report.retention.failed.size
            if (undeleted > 0) {
                add(resources.getQuantityString(R.plurals.cleanup_remove_failures, undeleted, undeleted))
            }
            if (report.failures > 0) {
                add(resources.getQuantityString(R.plurals.cleanup_failures, report.failures, report.failures))
            }
            if (report.paused && isNotEmpty()) add(getString(R.string.cleanup_paused))
        }
        retentionReport.visibility = if (parts.isEmpty()) View.GONE else View.VISIBLE
        retentionReport.text = getString(
            R.string.retention_last_cleanup,
            parts.joinToString("; ").replaceFirstChar { it.uppercase() },
        )
    }

    private fun retentionClassName(retentionClass: RetentionClass): String = getString(
        when (retentionClass) {
            RetentionClass.STANDARD -> R.string.retention_class_standard
            RetentionClass.ARCHIVAL -> R.string.retention_class_archival
        },
    )

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
        if (requestCode == REQUEST_EXPORT_RECORDING) {
            val target = data?.data
            if (resultCode == RESULT_OK && target != null) finishExport(target) else pendingExportId = null
            return
        }
        if (requestCode != REQUEST_IMPORT_MODEL || resultCode != RESULT_OK) return
        val uri: Uri = data?.data ?: run {
            onDeviceStatus.setText(R.string.on_device_file_missing)
            return
        }
        onDeviceStatus.setText(R.string.on_device_importing)
        val resolver = contentResolver
        thread {
            val displayName = runCatching {
                resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
                    if (cursor.moveToFirst()) cursor.getString(0) else null
                }
            }.getOrNull()
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
                runCatching {
                    application.onDeviceEngine.importModel(input, OnDeviceEngine.sanitizeModelName(displayName))
                }.getOrElse { error ->
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
        // One snapshot, so the status line and the list always agree.
        val models = application.onDeviceEngine.installedModels()
        renderOnDeviceStatus(models)
        renderInstalledModels(models)
    }

    private fun renderOnDeviceStatus(models: List<OnDeviceEngine.InstalledModel>) {
        val active = models.firstOrNull { it.active }
        onDeviceStatus.text = when {
            active == null -> getString(R.string.on_device_status_no_model)
            engineOnDeviceInput.isChecked ->
                getString(R.string.on_device_model_in_use, modelDisplayName(active), mb(active.sizeBytes))
            else -> getString(
                R.string.on_device_model_not_selected,
                modelDisplayName(active),
                mb(active.sizeBytes),
                getString(R.string.engine_on_device),
            )
        }
    }

    /** Catalog file names are reserved for verified downloads (imports never take them). */
    private fun isRecommended(model: OnDeviceEngine.InstalledModel): Boolean =
        model.name == ModelCatalog.RECOMMENDED_PARAKEET.fileName

    /** Catalog models by their label; anything else by its file name. */
    private fun modelDisplayName(model: OnDeviceEngine.InstalledModel): String {
        return if (isRecommended(model)) {
            getString(R.string.recommended_model_name, ModelCatalog.RECOMMENDED_PARAKEET.label)
        } else {
            model.name
        }
    }

    /**
     * One row per installed model: tap to make it the active one, Delete to
     * free its storage. The download button is hidden once the recommended
     * model is installed (a running or paused download still shows it).
     */
    private fun renderInstalledModels(models: List<OnDeviceEngine.InstalledModel>) {
        val recommended = ModelCatalog.RECOMMENDED_PARAKEET
        val visibility = if (models.isEmpty()) View.GONE else View.VISIBLE
        installedModelsHeading.visibility = visibility
        installedModelsHelp.visibility = visibility
        installedModelsContainer.removeAllViews()
        // Each row is its own layout, so no RadioGroup keeps them exclusive.
        val choices = mutableListOf<RadioButton>()
        for (model in models) {
            val displayName = modelDisplayName(model)
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                // Baseline alignment measures the weighted label before it
                // wraps, clipping long file names.
                isBaselineAligned = false
            }
            val choice = RadioButton(this).apply {
                text = getString(R.string.installed_model_row, displayName, mb(model.sizeBytes))
                isChecked = model.active
                setOnClickListener {
                    if (model.active) return@setOnClickListener
                    // Until the re-render after the change (which also
                    // restores the real selection when it fails).
                    for (other in choices) other.isChecked = other === this
                    changeModels(R.string.select_model_error) { it.selectModel(model.name) }
                }
            }
            choices += choice
            row.addView(choice, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            val delete = Button(this, null, android.R.attr.buttonBarButtonStyle).apply {
                setText(R.string.delete_model)
                setOnClickListener { confirmDeleteModel(model.name, displayName) }
            }
            row.addView(delete)
            installedModelsContainer.addView(row)
        }
        val recommendedInstalled = models.any(::isRecommended)
        val downloads = application.modelDownloads
        downloadModelButton.visibility = if (
            recommendedInstalled && !downloads.isRunning && downloads.resumableBytes(recommended) == 0L
        ) {
            View.GONE
        } else {
            View.VISIBLE
        }
    }

    private fun confirmDeleteModel(name: String, displayName: String) {
        AlertDialog.Builder(this)
            .setTitle(R.string.delete_model_title)
            .setMessage(getString(R.string.delete_model_message, displayName))
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(R.string.delete_model) { _, _ ->
                changeModels(R.string.delete_model_error) { it.deleteModel(name) }
            }
            .show()
    }

    /** Runs [change] off the main thread (it waits for an in-flight transcription), then re-renders. */
    private fun changeModels(errorMessage: Int, change: (OnDeviceEngine) -> Boolean) {
        val engine = application.onDeviceEngine
        thread {
            val changed = runCatching { change(engine) }.getOrElse { error ->
                Log.w(TAG, "model change failed", error)
                false
            }
            runOnUiThread {
                if (isDestroyed || isFinishing) return@runOnUiThread
                if (application.modelDownloads.isRunning) {
                    renderInstalledModels(application.onDeviceEngine.installedModels())
                } else {
                    refreshOnDeviceStatus()
                }
                if (!changed) onDeviceStatus.setText(errorMessage)
            }
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
        // Free space first (#342): no take starts that the disk cannot hold.
        val disk = application.diskBeforeTake()
        if (disk?.level == DiskLevel.CRITICAL) {
            recordingMessage.text = getString(R.string.disk_full_refused, mb(disk.availableBytes))
            return
        }
        val recording = runCatching { application.recordings.create() }.getOrElse {
            recordingMessage.text = getString(R.string.recording_storage_error)
            return
        }
        // The live stream is an observer of the capture, never a gate on it:
        // beginStreaming returns null whenever the configuration or endpoint
        // does not support streaming, and the recording proceeds as before.
        val config = application.backendSettings.load()
        var session: StreamSession? = null
        val savedAudio = application.recordings.partialFile(recording)
        session = application.transcription.beginStreaming(config, savedAudio) { event ->
            // Events from a superseded session must not rewrite the views of
            // the recording that replaced it.
            if (streamSession === session) onStreamEvent(event)
        }
        val error = capture.start(
            this,
            savedAudio,
            onChunk = session?.let { streaming -> AudioChunkListener { bytes, count -> streaming.onAudio(bytes, count) } },
            // The capture ended itself (low storage, the two-hour cap):
            // settle it like Stop.
            onEnded = { if (activeRecording === recording) stopAndQueueRecording() },
        )
        if (error != null) {
            session?.close()
            runCatching { application.recordings.salvageCapture(recording.id, error) }
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
            when {
                session == null -> R.string.recording_now
                application.isOnDeviceModelLoading(config) -> R.string.streaming_loading
                else -> R.string.streaming_connecting
            },
        )
        if (disk?.level == DiskLevel.LOW) {
            recordingMessage.append("\n")
            recordingMessage.append(getString(R.string.disk_low_warning, DiskPolicy.DEFAULT.minutesLeft(disk.availableBytes)))
        }
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
                    runCatching {
                        application.recordings.salvageCapture(recording.id, "Unable to finalize the private WAV recording")
                    }
                    updateRecordingViews {
                        recordingMessage.setText(R.string.recording_finalize_error)
                        refreshRecordings()
                    }
                    return
                }
                val config = application.backendSettings.load()
                // Why a take that ended by itself stopped stays on screen
                // through its transcription.
                val notice = when {
                    result.stoppedForLowDisk -> getString(R.string.recording_stopped_low_disk)
                    result.cappedAtLimit -> getString(R.string.recording_capped)
                    else -> null
                }
                val settled: (Recording) -> Unit = { onTranscriptionSettled(it, notice) }
                // With a live session, commit the stream and store its final
                // transcript; the coordinator falls back to this same batch
                // upload of the saved WAV whenever the stream failed.
                val queued = if (session != null) {
                    application.transcription.finishStreaming(session, finalized.id, config, settled)
                } else {
                    application.transcription.transcribe(finalized.id, config, settled)
                }
                if (queued) updateRecordingViews { recordingMessage.text = notice ?: getString(R.string.sending_recording) }
                updateRecordingViews { refreshRecordings() }
            }
            is CaptureResult.Failed -> {
                // Nothing will be transcribed; the server session and the
                // stale live partial go with it.
                session?.close()
                updateRecordingViews { liveTranscript.visibility = View.GONE }
                runCatching { application.recordings.salvageCapture(recording.id, result.message) }
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

    private fun onTranscriptionSettled(completed: Recording, notice: String? = null) {
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
        notice?.let { recordingMessage.append("\n"); recordingMessage.append(it) }
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
        if (requestCode == REQUEST_KEYBOARD_MICROPHONE) {
            // Back to the field as soon as the microphone is allowed; the
            // notification answer, whatever it is, does not hold that up.
            if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                finishAndRemoveTask()
            } else {
                recordingMessage.setText(R.string.microphone_permission_required)
            }
            return
        }
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
        val recoveryView = row.findViewById<TextView>(R.id.recording_recovery)
        val removedView = row.findViewById<TextView>(R.id.recording_audio_removed)
        val archive = row.findViewById<Button>(R.id.archive_recording_button)
        val transcript = row.findViewById<TextView>(R.id.recording_transcript)
        val revisionsView = row.findViewById<TextView>(R.id.recording_revisions)
        val play = row.findViewById<Button>(R.id.play_recording_button)
        val export = row.findViewById<Button>(R.id.export_recording_button)
        val retry = row.findViewById<Button>(R.id.retry_recording_button)
        val delete = row.findViewById<Button>(R.id.delete_recording_button)

        title.text = DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT)
            .format(Date(recording.createdAtMillis))
        status.text = when (recording.status) {
            // Live only while a capture of this process holds its WAV; one
            // whose settlement failed (say, the disk was full) says so.
            RecordingStatus.RECORDING -> getString(
                if (WavWriter.isOpen(application.recordings.partialFile(recording))) {
                    R.string.status_recording
                } else {
                    R.string.status_unsettled
                },
            )
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
        val latest = recording.revisions.lastOrNull()
        if (recording.status == RecordingStatus.TRANSCRIBED && latest != null) {
            status.append(" · ")
            status.append(revisionSource(latest))
        }
        if (recording.retentionClass == RetentionClass.ARCHIVAL) {
            status.append(" · ")
            status.append(getString(R.string.archived_mark))
        }
        recording.errorMessage?.let {
            status.append(" — ")
            status.append(it)
        }
        // A recovered take is never presented as a complete one.
        val recovery = recording.recovery
        recoveryView.visibility = if (recovery == null) View.GONE else View.VISIBLE
        recoveryView.text = recovery?.let {
            getString(R.string.recovery_note, clock(it.recoveredSeconds), clock(it.confirmedSeconds), it.reason)
        }
        if (recording.rawTranscript != null) {
            transcript.visibility = View.VISIBLE
            // The exact server text is shown and retained. There is no cleanup
            // pass that could remove words such as "not", "like", or "auth".
            transcript.text = recording.rawTranscript
        } else {
            transcript.visibility = View.GONE
        }
        // Retries add results; the earlier ones stay with the recording.
        val earlier = recording.revisions.dropLast(1)
        revisionsView.visibility = if (earlier.isEmpty()) View.GONE else View.VISIBLE
        revisionsView.text = earlier.reversed().joinToString(
            separator = "\n",
            prefix = getString(R.string.revisions_heading) + "\n",
        ) { revision ->
            getString(
                R.string.revision_line,
                DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT).format(Date(revision.createdAtMillis)),
                revisionSource(revision),
                revision.text,
            )
        }

        // Only the audio goes when the retention policy removes it (#342).
        val removal = recording.audioRemoved
        removedView.visibility = if (removal == null) View.GONE else View.VISIBLE
        removedView.text = removal?.let {
            getString(R.string.audio_removed_note, DateFormat.getDateInstance(DateFormat.MEDIUM).format(Date(it.atMillis)))
        }
        archive.visibility = if (removal == null && recording.status != RecordingStatus.RECORDING) View.VISIBLE else View.GONE
        archive.setText(
            if (recording.retentionClass == RetentionClass.ARCHIVAL) R.string.unarchive_recording else R.string.archive_recording,
        )
        archive.setOnClickListener {
            val target = if (recording.retentionClass == RetentionClass.ARCHIVAL) RetentionClass.STANDARD else RetentionClass.ARCHIVAL
            runCatching { application.recordings.setRetentionClass(recording.id, target) }
            application.audioUpkeep.schedule()
            refreshRecordings()
        }

        val hasAudio = recording.status != RecordingStatus.RECORDING && removal == null &&
            application.recordings.audioFile(recording).isFile
        play.visibility = if (hasAudio) View.VISIBLE else View.GONE
        play.setText(if (playingId == recording.id) R.string.stop_playback else R.string.play_recording)
        play.setOnClickListener { togglePlayback(recording) }
        export.visibility = if (hasAudio) View.VISIBLE else View.GONE
        export.setOnClickListener { exportRecording(recording) }

        // Retry is hidden while a transcription for this recording is still
        // in flight, so a double tap cannot queue a second request. A
        // TRANSCRIBING row without an active request (for example after a
        // process restart) still offers Retry to recover it. A transcribed
        // recording can be transcribed again (another model, say); that adds
        // a result instead of replacing the current one.
        retry.visibility = if (recording.status != RecordingStatus.RECORDING &&
            hasAudio &&
            !application.transcription.isActive(recording.id)
        ) View.VISIBLE else View.GONE
        retry.setText(
            if (recording.status == RecordingStatus.TRANSCRIBED) R.string.retranscribe_recording else R.string.retry_recording,
        )
        retry.setOnClickListener { chooseRetryTarget(recording) }
        delete.setOnClickListener {
            AlertDialog.Builder(this)
                .setTitle(R.string.delete_recording_title)
                .setMessage(R.string.delete_recording_message)
                .setNegativeButton(android.R.string.cancel, null)
                .setPositiveButton(R.string.delete_recording) { _, _ ->
                    if (playingId == recording.id) stopPlayback()
                    runCatching { application.recordings.delete(recording.id) }
                        .onSuccess { refreshRecordings() }
                        .onFailure { recordingMessage.setText(R.string.delete_recording_error) }
                }
                .show()
        }
    }

    private fun revisionSource(revision: TranscriptRevision): String {
        val where = when (revision.source) {
            TranscriptSource.ON_DEVICE -> getString(R.string.revision_source_on_device)
            TranscriptSource.SERVER -> getString(R.string.revision_source_server)
            null -> null
        }
        return listOfNotNull(where, revision.model).joinToString(": ")
            .ifEmpty { getString(R.string.revision_source_unknown) }
    }

    /**
     * Retry, or transcribe again, with the configured engine or any other
     * installed model (#356). The same saved audio is sent every time; the
     * choice applies to this attempt only and changes no setting.
     */
    private fun chooseRetryTarget(recording: Recording) {
        val config = application.backendSettings.load()
        val onDevice = application.onDeviceEngine.installedModels()
            .sortedByDescending { it.active }
            .map { model ->
                getString(R.string.retry_with_on_device, modelDisplayName(model)) to
                    config.copy(engine = TranscriptionEngine.ON_DEVICE, onDeviceModel = model.name)
            }
        // A server is offered once one is configured (or it is the engine).
        val serverConfigured = config.engine == TranscriptionEngine.REMOTE ||
            config.endpoint != BackendSettings.DEFAULT_ENDPOINT
        val server = if (serverConfigured && config.model.isNotBlank() &&
            EndpointPolicy.validate(config.endpoint, config.allowTrustedLanHttp) is EndpointValidation.Valid
        ) {
            listOf(getString(R.string.retry_with_server, config.model) to config.copy(engine = TranscriptionEngine.REMOTE))
        } else {
            emptyList()
        }
        val options = if (config.engine == TranscriptionEngine.REMOTE) server + onDevice else onDevice + server
        if (options.size <= 1) {
            retryRecording(recording, options.firstOrNull()?.second ?: config)
            return
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.retry_with_title)
            .setItems(options.map { it.first }.toTypedArray()) { _, which -> retryRecording(recording, options[which].second) }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun retryRecording(recording: Recording, config: BackendConfig) {
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

    private fun togglePlayback(recording: Recording) {
        val playing = playingId == recording.id
        stopPlayback()
        if (playing) {
            refreshRecordings()
            return
        }
        // Prepared off the main thread (a long WAV must not stall it). The
        // row offers Stop at once; a tap while preparing releases the
        // player, and a released or replaced player's callbacks do nothing.
        val player = MediaPlayer()
        val queued = runCatching {
            // Opened under the store lock: a compression that publishes
            // meanwhile unlinks the WAV, but an open file keeps playing.
            application.recordings.openAudio(recording.id).stream.use { player.setDataSource(it.fd) }
            player.setOnPreparedListener { if (this.player === it) it.start() }
            player.setOnCompletionListener {
                if (this.player !== it) return@setOnCompletionListener
                stopPlayback()
                refreshRecordings()
            }
            player.setOnErrorListener { failed, _, _ ->
                if (this.player === failed) {
                    stopPlayback()
                    recordingMessage.setText(R.string.playback_error)
                    refreshRecordings()
                }
                true
            }
            player.prepareAsync()
        }.isSuccess
        if (!queued) {
            player.release()
            recordingMessage.setText(R.string.playback_error)
            return
        }
        this.player = player
        playingId = recording.id
        refreshRecordings()
    }

    private fun stopPlayback() {
        player?.let { runCatching { it.stop() }; it.release() }
        player = null
        playingId = null
    }

    /** Saves a copy of the recording's audio wherever the user picks (Storage Access Framework). */
    private fun exportRecording(recording: Recording) {
        // The audio as it is stored: FLAC once compressed (#342), else WAV.
        val audio = application.recordings.audioFile(recording)
        val stamp = java.text.SimpleDateFormat("yyyyMMdd-HHmmss", java.util.Locale.US).format(Date(recording.createdAtMillis))
        pendingExportId = recording.id
        pendingExportFlac = audio.extension == "flac"
        startActivityForResult(
            Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = if (pendingExportFlac) "audio/flac" else "audio/wav"
                putExtra(Intent.EXTRA_TITLE, "starling-$stamp.${audio.extension}")
            },
            REQUEST_EXPORT_RECORDING,
        )
    }

    private fun finishExport(uri: Uri) {
        val id = pendingExportId ?: run {
            recordingMessage.setText(R.string.export_error)
            return
        }
        pendingExportId = null
        val flac = pendingExportFlac
        val resolver = contentResolver
        thread {
            val exported = runCatching {
                resolver.openOutputStream(uri, "w")!!.use { output ->
                    if (flac) {
                        // Compression only ever goes from WAV to FLAC.
                        application.recordings.openAudio(id).stream.use { it.copyTo(output) }
                    } else {
                        // Named .wav: the request WAV, byte for byte the
                        // original even if upkeep compressed the take since.
                        application.recordings.withRequestAudio(id) { wav -> wav.inputStream().use { it.copyTo(output) } }
                    }
                }
            }.onFailure { Log.w(TAG, "recording export failed", it) }.isSuccess
            runOnUiThread {
                if (isDestroyed || isFinishing) return@runOnUiThread
                recordingMessage.setText(if (exported) R.string.export_done else R.string.export_error)
            }
        }
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val REQUEST_RECORD_AUDIO = 4001
        private const val REQUEST_IMPORT_MODEL = 4002
        private const val REQUEST_KEYBOARD_MICROPHONE = 4003
        private const val REQUEST_EXPORT_RECORDING = 4004
        private const val STATE_PENDING_EXPORT = "pending_export_id"
        private const val STATE_PENDING_EXPORT_FLAC = "pending_export_flac"

        /** The voice keyboard asks for the microphone through this screen. */
        const val ACTION_REQUEST_MICROPHONE = "dev.starling.mobile.action.REQUEST_MICROPHONE"
        const val EXTRA_ASK_NOTIFICATIONS = "dev.starling.mobile.extra.ASK_NOTIFICATIONS"
        // Decimal megabytes, as Hugging Face and file managers show sizes.
        private const val MB = 1_000_000L

        /** Bytes to decimal MB, rounded to nearest like Hugging Face's listing. */
        private fun mb(bytes: Long): Int = ((bytes + MB / 2) / MB).toInt()

        /** Bytes to MiB rounded up, as the desktop's cleanup summary counts. */
        private fun mbCeil(bytes: Long): Int = ((bytes + MIB - 1) / MIB).toInt()
        private const val MIB = 1024L * 1024

        /** The retention spinners' choices, in their string-array order; null is no limit. */
        private val AGE_CHOICES = listOf(null, 30, 90, 365)
        private val SIZE_CHOICES_MB = listOf(null, 1024L, 5 * 1024L, 20 * 1024L)

        /** 612.4 s as "10:12". */
        private fun clock(seconds: Double): String {
            val whole = seconds.toLong()
            return "%d:%02d".format(whole / 60, whole % 60)
        }
    }
}
