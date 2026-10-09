package dev.starling.mobile

import android.Manifest
import android.content.ClipData
import android.content.ClipDescription
import android.content.ClipboardManager
import android.content.Intent
import android.content.pm.PackageManager
import android.inputmethodservice.InputMethodService
import android.os.Build
import android.os.PersistableBundle
import android.view.LayoutInflater
import android.view.View
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputConnection
import android.view.inputmethod.InputMethodManager
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
import dev.starling.mobile.engine.ModelLifetime
import dev.starling.mobile.network.StreamEvent
import dev.starling.mobile.network.StreamSession
import dev.starling.mobile.network.TranscriptionEngine
import dev.starling.mobile.ui.EditorField
import dev.starling.mobile.ui.InputTargetGuard
import dev.starling.mobile.ui.RequestGenerationGuard

/**
 * Lightweight voice keyboard. It never reads surrounding editor text. Two
 * text paths exist, both guarded by the editor target the take is bound to:
 *
 * - **Live streaming** (on-device engine or Starling server): while
 *   recording, growing partial transcripts are shown through
 *   `setComposingText`, the Android dictation idiom — composing text is
 *   ephemeral, replaces itself with each partial, and is removed when the
 *   stream fails. The single `commitText` happens only when the final
 *   transcript arrives after Stop.
 * - **Batch transcription** (fallback after any streaming failure, or an
 *   OpenAI-shaped endpoint): no asynchronous insertion at all; the
 *   transcript is shown first and `commitText` is a separate, explicit user
 *   action on the Insert button.
 *
 * A take outlives the keyboard window. While it records, a microphone
 * foreground service keeps the capture alive through a screen lock or an app
 * switch, and the notification offers Stop. Leaving the field detaches the
 * take (its composing text is removed while the connection still answers). When a field with
 * the same declared [EditorField] attributes (held in memory only) comes
 * back, the take attaches again — but attributes cannot prove it is the same
 * editor (two chats share one layout), so from then on the take never writes
 * by itself: its live text stays in the keyboard and the final waits for an
 * explicit Insert. Any other field can only copy the take's text.
 *
 * Private fields ([EditorField.sensitive]: passwords, incognito) dictate an
 * ephemeral take that never shows up in the history and is deleted as soon
 * as it settles.
 */
class VoiceInputService : InputMethodService() {
    private val application by lazy { starlingApplication() }
    private val capture = AudioCapture()
    private val targetGuard = InputTargetGuard<InputConnection>()
    private val requestGuard = RequestGenerationGuard()

    private var keyboardView: View? = null
    private var recordButton: Button? = null
    private var insertButton: Button? = null
    private var copyButton: Button? = null
    private var switchKeyboardButton: Button? = null
    private var statusView: TextView? = null
    private var transcriptView: TextView? = null
    private var modelStatusView: TextView? = null
    private val modelStateListener: (ModelLifetime.State) -> Unit = { renderModelState(it) }

    /** The focused field, from onStartInput; null when there is none. */
    private var editorField: EditorField? = null

    /** The one take this keyboard follows, from Record until it is settled. */
    private var take: Take? = null

    override fun onCreate() {
        super.onCreate()
        application.modelLifetime.addListener(modelStateListener)
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
        copyButton = view.findViewById(R.id.keyboard_copy_button)
        switchKeyboardButton = view.findViewById(R.id.keyboard_switch_button)
        statusView = view.findViewById(R.id.keyboard_status)
        transcriptView = view.findViewById(R.id.keyboard_transcript)
        modelStatusView = view.findViewById(R.id.keyboard_model_status)
        renderModelState(application.modelLifetime.state())

        recordButton?.setOnClickListener {
            if (take?.capturing == true) stopTake() else requestOrStartRecording()
        }
        insertButton?.setOnClickListener { insertReadyTranscript() }
        copyButton?.setOnClickListener { copyReadyTranscript() }
        switchKeyboardButton?.setOnClickListener { switchToPreviousKeyboard() }
        renderTake()
        return container
    }

    override fun onStartInput(attribute: EditorInfo?, restarting: Boolean) {
        super.onStartInput(attribute, restarting)
        val field = EditorField.from(attribute)
        editorField = field
        val connection = currentInputConnection
        if (connection != null) targetGuard.targetStarted(connection) else targetGuard.targetFinished()
        val current = take
        if (current != null) {
            val previous = current.target
            if (previous != null && connection != null && previous.target === connection) {
                // restartInput on the very same connection: the composing
                // region is still the take's own.
                current.target = targetGuard.capture()
            } else {
                // The guard generation moved on, so the old binding is dead.
                // A connection replaced without onFinishInput may have had
                // its composing text finished by the editor; nothing more is
                // written automatically for this take.
                if (previous != null) current.explicitOnly = true
                current.target = null
                current.composing = false
                current.liveInField = false
                if (connection != null && field != null && current.field.sameFieldAs(field)) attach(current)
            }
        } else {
            // Text of an earlier take never carries over to another field.
            transcriptView?.visibility = View.GONE
            transcriptView?.text = null
        }
        renderTake()
    }

    /**
     * The keyboard is on screen: the user is likely about to dictate, so the
     * selected local model starts loading (or stays warm) now rather than at
     * the Record tap. Deduplicated and off the main thread (ModelLifetime).
     */
    override fun onStartInputView(info: EditorInfo?, restarting: Boolean) {
        super.onStartInputView(info, restarting)
        application.preloadOnDeviceModel()
        renderModelState(application.modelLifetime.state())
        switchKeyboardButton?.visibility = if (offersKeyboardSwitch()) View.VISIBLE else View.GONE
    }

    /**
     * Hiding the keyboard (Back) while the field keeps focus finishes
     * composing in super, which would commit the live partial as ordinary
     * text. The take's composing text is removed first; the next partial
     * (or the final's commitText) writes it again on the same connection.
     */
    override fun onFinishInputView(finishingInput: Boolean) {
        if (!finishingInput) take?.let(::clearComposingText)
        super.onFinishInputView(finishingInput)
    }

    override fun onFinishInput() {
        // The take keeps recording without a field. Its composing text is
        // removed while this connection is still valid, so the default
        // finishComposingText in super commits nothing into the old field.
        take?.let(::detach)
        targetGuard.targetFinished()
        renderTake()
        super.onFinishInput()
    }

    override fun onDestroy() {
        application.modelLifetime.removeListener(modelStateListener)
        // Switching to another keyboard unbinds this one: the take ends here
        // and settles into the store (deleted again if it was private).
        // A stop already in flight releases its own hold when it settles.
        if (take?.capturing == true) stopTake()
        super.onDestroy()
    }

    private fun requestOrStartRecording() {
        val microphone = checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        val preferences = getSharedPreferences(PREFERENCES, MODE_PRIVATE)
        // The take notification (and its Stop) needs its own permission on
        // Android 13+; it is asked once, even when the microphone was
        // already granted elsewhere, and a refusal is not asked again.
        val askNotifications = Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED &&
            !preferences.getBoolean(KEY_NOTIFICATIONS_ASKED, false)
        if (!microphone || askNotifications) {
            if (askNotifications) preferences.edit().putBoolean(KEY_NOTIFICATIONS_ASKED, true).apply()
            // A keyboard cannot show a permission dialog; the app asks and
            // closes again, back to this field.
            statusView?.setText(
                if (microphone) R.string.keyboard_notification_permission else R.string.keyboard_microphone_permission,
            )
            startActivity(
                Intent(this, MainActivity::class.java)
                    .setAction(MainActivity.ACTION_REQUEST_MICROPHONE)
                    .putExtra(MainActivity.EXTRA_ASK_NOTIFICATIONS, askNotifications)
                    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
            )
            return
        }
        beginCapture()
    }

    private fun beginCapture() {
        val target = targetGuard.capture()
        val field = editorField
        if (target == null || currentInputConnection == null || field == null) {
            statusView?.setText(R.string.keyboard_no_target)
            return
        }
        val sensitive = field.sensitive
        val recording = runCatching { application.recordings.create(ephemeral = sensitive) }.getOrElse {
            statusView?.setText(R.string.recording_storage_error)
            return
        }
        // The live stream is an observer of the capture; null means this
        // configuration records in the plain batch mode.
        val config = application.backendSettings.load()
        var session: StreamSession? = null
        val savedAudio = application.recordings.partialFile(recording)
        session = application.transcription.beginStreaming(config, savedAudio) { event ->
            // Events from a superseded or stopped take must not touch the
            // state of the take that replaced it.
            val current = take
            if (current != null && current.capturing && current.session === session) onStreamEvent(current, event)
        }
        val error = capture.start(
            this,
            savedAudio,
            onChunk = session?.let { streaming ->
                AudioChunkListener { bytes, count -> streaming.onAudio(bytes, count) }
            },
        )
        if (error != null) {
            session?.close()
            discardOrFail(recording, sensitive, error)
            statusView?.text = error
            return
        }
        // Only a capture that really started invalidates the earlier take,
        // whose callbacks may still be queued; it stays in the store (or is
        // deleted there, if it was private).
        val requestGeneration = requestGuard.begin()
        // A take still settling owns its composing region until now; the
        // new take does not, so the old region leaves the field first.
        take?.let(::clearComposingText)
        transcriptView?.visibility = View.GONE
        transcriptView?.text = null
        take = Take(recording, requestGeneration, field, sensitive).also { started ->
            started.session = session
            started.target = target
            started.liveInField = session != null && field.supportsComposing
        }
        take?.foregroundHold = CaptureForegroundService.hold(this) { stopTake() }
        renderTake()
        statusView?.setText(
            when {
                session == null -> R.string.keyboard_recording
                // Audio is already being saved; live text follows the load.
                application.isOnDeviceModelLoading(config) -> R.string.keyboard_recording_loading
                else -> R.string.keyboard_streaming
            },
        )
    }

    /**
     * Binds a detached take to a field with the same attributes. Only an
     * explicit Insert writes there from now on; the live text shows in the
     * keyboard.
     */
    private fun attach(current: Take) {
        current.target = targetGuard.capture() ?: return
        current.explicitOnly = true
        current.composing = false
        current.liveInField = false
        if (current.ready == null) {
            current.lastPartial?.let { partial ->
                transcriptView?.visibility = View.VISIBLE
                transcriptView?.text = visibleText(current, partial)
            }
        }
    }

    /** Unbinds the take from a field that is going away, leaving no text behind. */
    private fun detach(current: Take) {
        clearComposingText(current)
        current.liveInField = false
        current.target = null
    }

    /** Shows where the on-device model stands; hidden when it is ready or unused. */
    private fun renderModelState(state: ModelLifetime.State) {
        val view = modelStatusView ?: return
        val text = if (application.backendSettings.load().engine != TranscriptionEngine.ON_DEVICE) {
            null
        } else {
            when (state) {
                is ModelLifetime.State.Loading -> getString(R.string.model_status_loading)
                is ModelLifetime.State.Failed -> getString(R.string.model_status_failed, state.reason)
                is ModelLifetime.State.DriverFailed -> getString(R.string.model_status_driver_failed, state.reason)
                is ModelLifetime.State.Ready, ModelLifetime.State.Unloaded -> null
            }
        }
        view.text = text
        view.visibility = if (text == null) View.GONE else View.VISIBLE
    }

    /**
     * Live-stream events, already marshalled to the main thread. Composing
     * text is written only while the take is bound to the focused editor,
     * and is removed when the stream fails — the batch fallback then takes
     * over after Stop.
     */
    private fun onStreamEvent(current: Take, event: StreamEvent) {
        when (event) {
            StreamEvent.Live -> statusView?.setText(R.string.keyboard_streaming)
            is StreamEvent.Partial -> {
                current.lastPartial = event.text
                val target = current.target
                val connection = currentInputConnection
                if (current.liveInField && target != null && targetGuard.isCurrent(target, connection)) {
                    // The partial grows over the whole session, so each one
                    // replaces the composing region entirely. An empty text
                    // without a region of ours would replace the selection.
                    if (current.composing || event.text.isNotEmpty()) {
                        connection.setComposingText(event.text, 1)
                        current.composing = event.text.isNotEmpty()
                    }
                } else {
                    // No composing region (detached, or a field that cannot
                    // compose): the keyboard shows the live text itself.
                    transcriptView?.visibility = View.VISIBLE
                    transcriptView?.text = visibleText(current, event.text)
                }
            }
            is StreamEvent.Interrupted -> {
                current.liveInField = false
                clearComposingText(current)
                statusView?.text = getString(R.string.keyboard_stream_interrupted, event.reason)
            }
        }
    }

    /**
     * Removes any composing region the take owns, without committing its
     * partial text, so a failed stream or a departed field leaves the editor
     * unchanged.
     */
    private fun clearComposingText(current: Take) {
        if (!current.composing) return
        current.composing = false
        val target = current.target ?: return
        val connection = currentInputConnection
        if (targetGuard.isCurrent(target, connection)) {
            connection.setComposingText("", 0)
            connection.finishComposingText()
        }
    }

    private fun stopTake() {
        val current = take ?: return
        if (!current.capturing) return
        // Request-local state is captured in the take itself; a new editor
        // or take may replace the keyboard's fields while this one settles.
        current.capturing = false
        val session = current.session
        current.session = null
        renderTake()

        // The capture settles inline on this main thread in the common
        // case; when the microphone refuses to stop, the outcome is
        // delivered later, still on the main thread, so input teardown and
        // onDestroy never block on the forced-release wait.
        capture.stop { result ->
            // The foreground lasts until the capture has really let go of the
            // microphone and finalized its WAV; the token keeps this late
            // callback from releasing a newer take's hold.
            CaptureForegroundService.release(current.foregroundHold)
            settleStoppedRecording(current, session, result)
        }
    }

    private fun settleStoppedRecording(current: Take, session: StreamSession?, result: CaptureResult) {
        val recording = current.recording
        when (result) {
            is CaptureResult.Completed -> {
                // The WAV is finalized and durable before any network use.
                val finalized = runCatching {
                    application.recordings.commitAudio(recording, result.durationSeconds)
                }.getOrElse {
                    session?.close()
                    discardOrFail(recording, current.sensitive, "Unable to finalize the private WAV recording")
                    endTake(current, R.string.recording_finalize_error)
                    return
                }
                if (take === current) {
                    statusView?.setText(
                        if (result.cappedAtLimit) R.string.recording_capped else R.string.keyboard_sending,
                    )
                }
                val config = application.backendSettings.load()
                val settled: (Recording) -> Unit = { completed -> onTranscriptionSettled(current, completed) }
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
                discardOrFail(recording, current.sensitive, result.message)
                endTake(current, null, result.message)
            }
            CaptureResult.AlreadyStopped -> {
                session?.close()
                if (current.sensitive) runCatching { application.recordings.delete(recording.id) }
                endTake(current, R.string.recording_already_stopped)
            }
        }
    }

    /**
     * One settlement for both the batch and the streamed path: the audio and
     * exact transcript are already durable, so a superseded take may update
     * only the store. A private take leaves the store here, whatever the
     * outcome; its text lives on only in this keyboard until it is inserted.
     */
    private fun onTranscriptionSettled(current: Take, completed: Recording) {
        if (current.sensitive) runCatching { application.recordings.delete(completed.id) }
        if (!requestGuard.isCurrent(current.requestGeneration) || take !== current) return
        val text = completed.rawTranscript
        if (completed.status != RecordingStatus.TRANSCRIBED || text == null) {
            endTake(
                current,
                if (current.sensitive) R.string.keyboard_transcription_failed_private
                else R.string.keyboard_transcription_failed,
            )
            return
        }
        val target = current.target
        val connection = currentInputConnection
        if (completed.provenance == TranscriptionProvenance.LIVE_STREAM && current.liveInField &&
            target != null && targetGuard.isCurrent(target, connection)
        ) {
            // The single asynchronous commitText of the live path, in the
            // field the take started in: it replaces the composing region
            // (or, with none left, inserts at the cursor) and finishes it.
            // An empty final with no region of ours writes nothing: an empty
            // commit would replace whatever the user has selected.
            if (text.isNotEmpty() || current.composing) connection.commitText(text, 1)
            current.composing = false
            endTake(current, R.string.keyboard_inserted, shown = text.takeUnless { current.sensitive })
            return
        }
        // A batch result, or a live final whose composing region is gone
        // (the field changed, or never could compose): the explicit Insert
        // flow, with any leftover composing removed.
        clearComposingText(current)
        current.ready = text
        renderTake()
    }

    private fun insertReadyTranscript() {
        val current = take ?: return
        val text = current.ready ?: return
        val target = current.target
        val connection = currentInputConnection
        if (target == null || !targetGuard.isCurrent(target, connection)) {
            renderTake()
            return
        }
        // Reached only from the explicit Insert button, into the field the
        // take was dictated in.
        connection.commitText(text, 1)
        endTake(current, R.string.keyboard_inserted, shown = text.takeUnless { current.sensitive })
    }

    /** The take's field is gone; the user can still take the text along. */
    private fun copyReadyTranscript() {
        val current = take ?: return
        val text = current.ready ?: return
        val clipboard = getSystemService(ClipboardManager::class.java) ?: return
        val clip = ClipData.newPlainText(getString(R.string.keyboard_name), text)
        if (current.sensitive && Build.VERSION.SDK_INT >= 33) {
            // Keeps a password out of the clipboard preview and history.
            clip.description.extras = PersistableBundle().apply {
                putBoolean(ClipDescription.EXTRA_IS_SENSITIVE, true)
            }
        }
        clipboard.setPrimaryClip(clip)
        endTake(current, R.string.keyboard_copied, shown = text.takeUnless { current.sensitive })
    }

    /** Clears the take and leaves a final status (and, optionally, its text) on screen. */
    private fun endTake(current: Take, statusRes: Int?, detail: String? = null, shown: String? = null) {
        if (take !== current) return
        clearComposingText(current)
        take = null
        renderTake()
        transcriptView?.visibility = if (shown == null) View.GONE else View.VISIBLE
        transcriptView?.text = shown
        statusView?.text = listOfNotNull(statusRes?.let(::getString), detail).joinToString(" ")
    }

    /** A private take is deleted outright; any other failure stays retryable. */
    private fun discardOrFail(recording: Recording, sensitive: Boolean, message: String) {
        runCatching {
            if (sensitive) application.recordings.delete(recording.id)
            else application.recordings.markFailed(recording.id, message)
        }
    }

    /**
     * Buttons and the transcript for the current take. Status lines that
     * report progress are set where the progress happens; this only owns
     * the state that follows from the take and the focused field.
     */
    private fun renderTake() {
        val current = take
        recordButton?.setText(if (current?.capturing == true) R.string.keyboard_stop else R.string.keyboard_record)
        val ready = current?.ready
        val bound = current?.target?.let { targetGuard.isCurrent(it, currentInputConnection) } == true
        insertButton?.visibility = if (ready != null && bound) View.VISIBLE else View.GONE
        copyButton?.visibility = if (ready != null && !bound) View.VISIBLE else View.GONE
        if (ready != null) {
            transcriptView?.visibility = View.VISIBLE
            transcriptView?.text = visibleText(current, ready)
            statusView?.setText(
                when {
                    bound -> R.string.keyboard_ready_to_insert
                    current.sensitive -> R.string.keyboard_target_changed_private
                    else -> R.string.keyboard_target_changed
                },
            )
        } else if (current != null) {
            // Every binding change re-renders the live text the keyboard
            // shows, so private text never outlives its field on screen.
            val partial = current.lastPartial
            if (partial != null && !current.composing) {
                transcriptView?.visibility = View.VISIBLE
                transcriptView?.text = visibleText(current, partial)
            }
        } else if (statusView?.text.isNullOrEmpty()) {
            statusView?.setText(R.string.keyboard_ready)
        }
    }

    /**
     * A private take's text shows only while its own field is focused; in
     * any other field (or none) the keyboard says it is hidden. It can
     * still be copied there, marked sensitive.
     */
    private fun visibleText(current: Take, text: String): String {
        // A field matched only by its attributes is not proof of the private
        // field itself, so its text stays hidden there too.
        val bound = !current.explicitOnly &&
            current.target?.let { targetGuard.isCurrent(it, currentInputConnection) } == true
        return if (current.sensitive && !bound) getString(R.string.keyboard_private_hidden) else text
    }

    /**
     * After dictating, one tap returns to the typing keyboard the user came
     * from. This also makes Starling usable as the voice key of keyboards
     * that switch to a voice input method (HeliBoard, FUTO Keyboard).
     */
    private fun switchToPreviousKeyboard() {
        if (Build.VERSION.SDK_INT >= 28) {
            if (!switchToPreviousInputMethod()) switchToNextInputMethod(false)
        } else {
            val token = window?.window?.attributes?.token ?: return
            @Suppress("DEPRECATION")
            getSystemService(InputMethodManager::class.java)?.switchToLastInputMethod(token)
        }
    }

    private fun offersKeyboardSwitch(): Boolean =
        if (Build.VERSION.SDK_INT >= 28) {
            shouldOfferSwitchingToNextInputMethod()
        } else {
            val token = window?.window?.attributes?.token
            @Suppress("DEPRECATION")
            token != null && getSystemService(InputMethodManager::class.java)
                ?.shouldOfferSwitchingToNextInputMethod(token) == true
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

    private companion object {
        const val PREFERENCES = "keyboard"
        const val KEY_NOTIFICATIONS_ASKED = "notifications_asked"
    }

    /**
     * One take, from Record until its text is inserted, copied or given up.
     * Main thread only; the capture worker sees only the stream session it
     * was handed at start.
     */
    private class Take(
        val recording: Recording,
        val requestGeneration: Long,
        /** The field the take was started in; it attaches only to that field. */
        val field: EditorField,
        val sensitive: Boolean,
    ) {
        var capturing = true
        var session: StreamSession? = null

        /** The editor binding; null while the take's field is not focused. */
        var target: InputTargetGuard.Snapshot<InputConnection>? = null

        /**
         * Whether the take writes its live text and final into [target] by
         * itself: a live stream into the verified field it started in.
         */
        var liveInField = false

        /** Whether the take owns an established (non-empty) composing region in [target]. */
        var composing = false

        /**
         * The take lost its original connection; it never writes into a
         * field by itself again, only on an explicit Insert.
         */
        var explicitOnly = false

        /** This take's foreground-service hold, released when its capture settles. */
        var foregroundHold = 0L

        var lastPartial: String? = null

        /** Final text waiting for an explicit Insert or Copy. */
        var ready: String? = null
    }
}
