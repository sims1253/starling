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
import android.os.PowerManager
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.BackgroundColorSpan
import android.text.style.ForegroundColorSpan
import android.text.style.StrikethroughSpan
import android.text.style.StyleSpan
import android.graphics.Typeface
import android.util.Log
import android.view.MotionEvent
import android.view.LayoutInflater
import android.view.View
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputConnection
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import dev.starling.mobile.audio.AudioCapture
import dev.starling.mobile.audio.AudioChunkListener
import dev.starling.mobile.audio.CaptureResult
import dev.starling.mobile.data.DerivedRevision
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
import dev.starling.mobile.data.TranscriptionProvenance
import dev.starling.mobile.engine.ModelLifetime
import dev.starling.mobile.network.StreamEvent
import dev.starling.mobile.network.StreamSession
import dev.starling.mobile.network.TranscriptionEngine
import dev.starling.mobile.processing.InsertionBoundary
import dev.starling.mobile.processing.ModeCatalog
import dev.starling.mobile.processing.Mode
import dev.starling.mobile.processing.RegionKind
import dev.starling.mobile.processing.StagedTake
import dev.starling.mobile.storage.DiskLevel
import dev.starling.mobile.ui.BoundaryDelivery
import dev.starling.mobile.ui.EditorField
import dev.starling.mobile.ui.InputTargetGuard

/**
 * Lightweight voice keyboard. Two text paths exist, both guarded by the
 * editor target the take is bound to:
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
 *
 * Every write into a field follows the insertion-boundary rules (#341,
 * [BoundaryDelivery]): the text around the cursor is read right before the
 * write and decides the leading space and the case of the first letter.
 * That text is used for the decision only, never stored or sent anywhere;
 * private fields are never read, and verbatim modes write the text as
 * recognized. An adjusted delivery is recorded on the take as a derived
 * revision; the transcript stays as recognized.
 *
 * Modes (#302) come from [ModeCatalog]. Direct mode is the flow above. A
 * staged mode ([StagedTake]) never writes while it records: Stop leaves an
 * editable draft above the keyboard — live tail, mode chip, the processed
 * proposal with the raw text one tap away — and only Insert writes, once,
 * into the take's field (and, for insert_enter modes, presses the field's own
 * action). A leading mode phrase may switch a direct take to a staged one
 * while it is spoken; until it is clear whether the first words are a phrase,
 * nothing is composed into the field, so command text never reaches it. A
 * trailing instruction always stages the take. Private fields never stage.
 */
class VoiceInputService : InputMethodService() {
    private val application by lazy { starlingApplication() }
    private val capture = AudioCapture()
    private val targetGuard = InputTargetGuard<InputConnection>()

    private var keyboardView: View? = null
    private var recordButton: Button? = null
    private var insertButton: Button? = null
    private var copyButton: Button? = null
    private var switchKeyboardButton: Button? = null
    private var statusView: TextView? = null
    private var transcriptView: TextView? = null
    private var modelStatusView: TextView? = null
    private var diskWarningView: TextView? = null
    private var modeChip: Button? = null
    private var modePicker: HorizontalScrollView? = null
    private var modeList: LinearLayout? = null
    private var decisionRow: View? = null
    private var decisionView: TextView? = null
    private var decisionUndo: Button? = null
    private var draftTools: View? = null
    private var viewToggle: Button? = null
    private val modelStateListener: (ModelLifetime.State) -> Unit = { renderModelState(it) }

    /** The focused field, from onStartInput; null when there is none. */
    private var editorField: EditorField? = null

    /** The one take this keyboard follows, from Record until it is settled. */
    private var take: Take? = null

    private val catalog: ModeCatalog by lazy { application.modeCatalog }

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
        diskWarningView = view.findViewById(R.id.keyboard_disk_warning)
        modeChip = view.findViewById(R.id.keyboard_mode_chip)
        modePicker = view.findViewById(R.id.keyboard_mode_picker)
        modeList = view.findViewById(R.id.keyboard_mode_list)
        decisionRow = view.findViewById(R.id.keyboard_decision_row)
        decisionView = view.findViewById(R.id.keyboard_decision)
        draftTools = view.findViewById(R.id.keyboard_draft_tools)
        viewToggle = view.findViewById(R.id.keyboard_view_toggle)
        renderModelState(application.modelLifetime.state())

        recordButton?.setOnClickListener {
            if (take?.capturing == true) stopTake() else requestOrStartRecording()
        }
        // Long-press on the mic is the thumb-reachable way to the modes.
        recordButton?.setOnLongClickListener { toggleModePicker(); true }
        modeChip?.setOnClickListener { toggleModePicker() }
        buildModePicker()
        insertButton?.setOnClickListener {
            if (take?.staged != null) insertDraft() else insertReadyTranscript()
        }
        copyButton?.setOnClickListener { copyReadyTranscript() }
        switchKeyboardButton?.setOnClickListener { switchToPreviousKeyboard() }
        decisionUndo = view.findViewById(R.id.keyboard_decision_undo)
        decisionUndo?.setOnClickListener {
            take?.staged?.undoDecision()
            renderTake()
        }
        view.findViewById<Button>(R.id.keyboard_delete_word).setOnClickListener {
            take?.staged?.deleteWord()
            renderTake()
        }
        view.findViewById<Button>(R.id.keyboard_discard).setOnClickListener {
            take?.let { endTake(it, R.string.staging_discarded) }
        }
        viewToggle?.setOnClickListener {
            val staged = take?.staged ?: return@setOnClickListener
            if (staged.proposal != null) staged.toggleView() else staged.backToRaw()
            renderTake()
        }
        transcriptView?.setOnTouchListener { v, event -> onDraftTouch(v, event) }
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
            if (previous != null && connection != null && previous.target === connection &&
                field != null && current.field.sameFieldAs(field)
            ) {
                // restartInput on the very same connection and field: the
                // composing region is still the take's own. (A restart that
                // changes the field — say, into a password — is a new field.)
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
            // A keyboard cannot show a permission dialog; the app asks in a
            // task of its own and removes it again, back to this field.
            statusView?.setText(
                if (microphone) R.string.keyboard_notification_permission else R.string.keyboard_microphone_permission,
            )
            startActivity(
                Intent(this, MainActivity::class.java)
                    .setAction(MainActivity.ACTION_REQUEST_MICROPHONE)
                    .putExtra(MainActivity.EXTRA_ASK_NOTIFICATIONS, askNotifications)
                    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_MULTIPLE_TASK),
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
        if (take?.staged != null && take?.awaitingFinal == true) {
            // The last capture's text has not reached the draft yet; a new
            // one now would invalidate it.
            statusView?.setText(R.string.staging_wait)
            return
        }
        // Recording again with a draft open (in its own field, never a
        // private one) adds to it, or replaces its selected word.
        val continuing = take?.takeIf {
            !sensitive && !it.capturing && it.staged != null &&
                it.target?.let { t -> targetGuard.isCurrent(t, currentInputConnection) } == true
        }
        val replacing = continuing?.staged?.selection != null
        // A private field gets the default mode: its route is blocked, so no
        // phrase, rule or processing ever applies there.
        val mode = continuing?.mode ?: if (sensitive) catalog.mode(null) else selectedMode()
        // Free space first (#342): no take starts that the disk cannot hold.
        val disk = application.diskBeforeTake()
        if (disk?.level == DiskLevel.CRITICAL) {
            statusView?.text = getString(R.string.disk_full_refused, (disk.availableBytes / 1_000_000).toInt())
            return
        }
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
            // A microphone that ends by itself settles like Stop, which also
            // releases the foreground hold.
            onEnded = { if (take?.recording === recording) stopTake() },
        )
        if (error != null) {
            session?.close()
            discardOrFail(recording, sensitive, error)
            statusView?.text = error
            return
        }
        // Only a capture that really started replaces the earlier take, whose
        // callbacks may still be queued; it stays in the store (or is deleted
        // there, if it was private).
        // A take still settling owns its composing region until now; the
        // new take does not, so the old region leaves the field first.
        take?.let(::clearComposingText)
        transcriptView?.visibility = View.GONE
        transcriptView?.text = null
        take = Take(recording, continuing?.field ?: field, sensitive, mode = mode).also { started ->
            started.session = session
            started.target = target
            started.staged = continuing?.staged
                ?: if (!sensitive && mode.processingDelivery == STAGED) newStagedTake(mode, recording.id) else null
            started.replacing = replacing && started.staged?.beginCorrection() == true
            if (!started.replacing) started.staged?.beginSegment()
            started.liveInField = session != null && field.supportsComposing && started.staged == null
        }
        take?.foregroundHold = CaptureForegroundService.hold(this) { stopTake() }
        renderTake()
        statusView?.setText(
            when {
                take?.replacing == true -> R.string.staging_say_replacement
                take?.staged != null -> R.string.staging_drafting
                session == null -> R.string.keyboard_recording
                // Audio is already being saved; live text follows the load.
                application.isOnDeviceModelLoading(config) -> R.string.keyboard_recording_loading
                else -> R.string.keyboard_streaming
            },
        )
        showDiskWarning(application.diskWarning(disk))
    }

    /** The low-storage warning of the running take, on its own line; null hides it. */
    private fun showDiskWarning(text: String?) {
        val view = diskWarningView ?: return
        view.text = text
        view.visibility = if (text == null) View.GONE else View.VISIBLE
    }

    /**
     * Binds a detached take to a field with the same attributes. Only an
     * explicit Insert writes there from now on; the live text shows in the
     * keyboard.
     */
    private fun attach(current: Take) {
        current.target = targetGuard.capture() ?: return
        current.explicitOnly = true
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
                if (current.replacing) {
                    statusView?.text = getString(R.string.staging_replacing, event.text)
                    return
                }
                current.staged?.let { staged ->
                    staged.partial(event.text)
                    renderTake()
                    return
                }
                var shown = event.text
                var holdBack = false
                var verbatim = current.mode.behavior == VERBATIM
                if (!current.sensitive) {
                    val routed = catalog.route(event.text, current.mode, secure = false)
                    val routedMode = routed.mode?.let(catalog::mode)
                    // A hand-picked mode is locked: a phrase is still kept
                    // out of the field, but it switches nothing.
                    if (!current.manualLocked && routed.prefixSpanCodepoints != null &&
                        routedMode?.processingDelivery == STAGED
                    ) {
                        // "draft mode …": the rest of this take is a draft.
                        switchToStaged(current)
                        current.staged?.partial(event.text)
                        renderTake()
                        return
                    }
                    // A phrase that keeps the take direct ("literal …") is
                    // never composed; neither are words that may still
                    // become one.
                    if (routed.prefixSpanCodepoints != null) {
                        shown = routed.payload
                        if (!current.manualLocked && routedMode != null) verbatim = routedMode.behavior == VERBATIM
                    }
                    holdBack = catalog.couldBecomePhrase(event.text, current.mode)
                    // From a possible "Starling, …" delimiter on, nothing is
                    // composed: an instruction never reaches the field, even
                    // for a moment. The final decides whether it was one.
                    catalog.instructions.liveCut(shown, current.mode.language)?.let { cut ->
                        shown = shown.substring(0, cut)
                    }
                }
                current.lastPartial = shown
                val target = current.target
                val connection = currentInputConnection
                if (!holdBack && current.liveInField && target != null && targetGuard.isCurrent(target, connection)) {
                    // The partial grows over the whole session, so each one
                    // replaces the composing region entirely. An empty text
                    // without a region of ours would replace the selection.
                    if (current.composing || shown.isNotEmpty()) {
                        // The boundary is read once, when the region starts:
                        // after that the field's text before it is the
                        // region's own.
                        if (!current.composing) {
                            current.liveBoundary = if (verbatim) null else BoundaryDelivery.context(connection, current.field)
                        }
                        val composed = BoundaryDelivery.adjust(shown, current.liveBoundary, verbatim)
                        connection.setComposingText(composed, 1)
                        current.composing = composed.isNotEmpty()
                        current.composedText = composed
                    }
                } else {
                    // No composing region (held back, detached, or a field
                    // that cannot compose): the keyboard shows the live text.
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
        current.composedText = null
        current.liveBoundary = null
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
        current.awaitingFinal = true
        current.stoppedAtNanos = System.nanoTime()
        showDiskWarning(null)
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
                    failCapture(current, R.string.recording_finalize_error)
                    return
                }
                if (take === current) {
                    statusView?.setText(
                        when {
                            result.stoppedForLowDisk -> R.string.recording_stopped_low_disk
                            result.cappedAtLimit -> R.string.recording_capped
                            else -> R.string.keyboard_sending
                        },
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
                failCapture(current, null, result.message)
            }
            CaptureResult.AlreadyStopped -> {
                session?.close()
                if (current.sensitive) runCatching { application.recordings.delete(recording.id) }
                failCapture(current, R.string.recording_already_stopped)
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
        if (take !== current) return
        var text = completed.rawTranscript
        if (completed.status != RecordingStatus.TRANSCRIBED || text == null) {
            if (keepDraft(current)) {
                // An earlier capture's text is still in the draft; only this
                // one failed (its audio stays in Starling for retry).
                statusView?.setText(R.string.staging_segment_failed)
                return
            }
            endTake(
                current,
                if (current.sensitive) R.string.keyboard_transcription_failed_private
                else R.string.keyboard_transcription_failed,
            )
            return
        }
        // Measured before any processing runs, so stop->raw is the recognizer's alone.
        val rawMs = (System.nanoTime() - current.stoppedAtNanos) / 1_000_000
        val staged = current.staged
        if (staged != null) {
            current.awaitingFinal = false
            if (current.replacing) {
                val replaced = staged.finishCorrection(text)
                renderTake()
                statusView?.setText(if (replaced) R.string.staging_replaced else R.string.staging_replace_failed)
            } else {
                staged.final(completed.id, text)
                settleDraft(current, staged, rawMs)
            }
            return
        }
        current.verbatim = current.mode.behavior == VERBATIM
        if (!current.sensitive) {
            val routed = catalog.route(text, current.mode, secure = false)
            val routedMode = if (current.manualLocked) current.mode else routed.mode?.let(catalog::mode) ?: current.mode
            val instruction = routedMode.behavior != VERBATIM &&
                catalog.instructions.split(routed.payload, routedMode.language).matched
            // A staged mode phrase, a spoken instruction, an ambiguous phrase
            // or nothing left after the phrase: the take becomes a draft
            // instead of being delivered.
            if (routedMode.processingDelivery == STAGED || instruction ||
                (routed.mode == null && !current.manualLocked) || routed.payload.isBlank()
            ) {
                current.awaitingFinal = false
                val draft = switchToStaged(current)
                if (current.manualLocked) draft.switchMode(current.mode)
                draft.final(completed.id, text)
                settleDraft(current, draft, rawMs)
                return
            }
            text = routed.payload
            current.verbatim = routedMode.behavior == VERBATIM
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
            if (text.isEmpty() && !current.composing) {
                endTake(current, R.string.keyboard_inserted)
                return
            }
            val delivered = commitTake(current, connection, text, current.verbatim)
            if (delivered != null) {
                endTake(current, insertedStatus(current, delivered), shown = delivered.text.takeUnless { current.sensitive })
                return
            }
        }
        // A batch result, a live final whose composing region is gone (the
        // field changed, or never could compose), or one the editor refused:
        // the explicit Insert flow, with any leftover composing removed.
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
        // take was dictated in. A refused commit keeps the text for Copy.
        val delivered = commitTake(current, connection, text, current.verbatim)
        if (delivered == null) {
            renderTake()
            return
        }
        endTake(current, insertedStatus(current, delivered), shown = delivered.text.takeUnless { current.sensitive })
    }

    /**
     * The one commitText of a delivery: the boundary rules (#341) applied
     * from the field's text read right before it, replacing the take's own
     * composing region when it has one. An adjusted delivery is recorded on
     * the take's recording as a derived revision. Null when the editor
     * refused the text.
     */
    private fun commitTake(
        current: Take,
        connection: InputConnection,
        text: String,
        verbatim: Boolean,
    ): BoundaryDelivery.Result? {
        val result = BoundaryDelivery.deliver(
            connection,
            current.field,
            text,
            verbatim,
            composing = current.composedText.takeIf { current.composing },
        )
        if (!result.committed) return null
        current.composing = false
        current.composedText = null
        current.liveBoundary = null
        if (result.changes.isNotEmpty() && !current.sensitive) {
            // The text is in the field either way; a failed write only
            // means the history lacks the derived revision, and says so.
            current.derivedUnsaved = runCatching {
                application.recordings.addDerived(
                    current.recording.id,
                    DerivedRevision(
                        text = result.text,
                        derivedFrom = text,
                        provenance = DerivedRevision.INSERTION_BOUNDARY,
                        changes = result.changes.map { it.kind },
                        createdAtMillis = System.currentTimeMillis(),
                    ),
                )
            }.isFailure
        }
        return result
    }

    /**
     * A field that did not report its text got the text as dictated, and an
     * adjustment the history could not keep is reported; the status says so.
     */
    private fun insertedStatus(current: Take, result: BoundaryDelivery.Result): Int = when {
        current.derivedUnsaved -> R.string.keyboard_inserted_unrecorded
        result.skipped == BoundaryDelivery.Skip.UNREADABLE -> R.string.keyboard_inserted_unadjusted
        else -> R.string.keyboard_inserted
    }

    /** The take's field is gone; the user can still take the text along. */
    private fun copyReadyTranscript() {
        val current = take ?: return
        val text = current.staged?.deliveryText() ?: current.ready ?: return
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

    /** The draft's text goes into the take's field, once, on this explicit tap. */
    private fun insertDraft() {
        val current = take ?: return
        val staged = current.staged ?: return
        val target = current.target
        val connection = currentInputConnection
        if (current.capturing || staged.recording || target == null || !targetGuard.isCurrent(target, connection)) {
            renderTake()
            return
        }
        if (!staged.hasPayload) {
            // Only a mode phrase or an instruction was said: an empty commit
            // would replace the field's selection, and nothing is sent.
            renderTake()
            statusView?.setText(R.string.staging_nothing_to_insert)
            return
        }
        val delivered = commitTake(current, connection, staged.deliveryText(), staged.mode.behavior == VERBATIM)
        if (delivered == null) {
            // The editor refused the text: the draft stays, and nothing is
            // pressed — a Send now would submit whatever the field held.
            renderTake()
            statusView?.setText(R.string.staging_insert_failed)
            return
        }
        val status = if (staged.mode.delivery == INSERT_ENTER && !pressEditorAction(connection)) {
            R.string.staging_no_action
        } else {
            insertedStatus(current, delivered)
        }
        endTake(current, status, shown = delivered.text)
    }

    /**
     * Presses the field's own action (Send, Search, Go) the way the app
     * declared it, never a synthetic Enter key. False when the field
     * declares none.
     */
    private fun pressEditorAction(connection: InputConnection): Boolean {
        val info = currentInputEditorInfo ?: return false
        // A custom action the app labelled is its own, whatever its id (0 included).
        if (info.actionLabel != null) return connection.performEditorAction(info.actionId)
        val action = info.imeOptions and EditorInfo.IME_MASK_ACTION
        if (action == EditorInfo.IME_ACTION_NONE || action == EditorInfo.IME_ACTION_UNSPECIFIED) return false
        return connection.performEditorAction(action)
    }

    /** What the Insert button says in an insert_enter mode: the field's own action. */
    private fun insertActionLabel(): String {
        val info = currentInputEditorInfo
        val label = info?.actionLabel?.toString()?.takeIf { it.isNotBlank() } ?: when (
            (info?.imeOptions ?: 0) and EditorInfo.IME_MASK_ACTION
        ) {
            EditorInfo.IME_ACTION_SEND -> getString(R.string.staging_action_send)
            EditorInfo.IME_ACTION_SEARCH -> getString(R.string.staging_action_search)
            EditorInfo.IME_ACTION_GO -> getString(R.string.staging_action_go)
            EditorInfo.IME_ACTION_DONE -> getString(R.string.staging_action_done)
            EditorInfo.IME_ACTION_NEXT -> getString(R.string.staging_action_next)
            else -> return getString(R.string.keyboard_insert)
        }
        return getString(R.string.staging_insert_action, label)
    }

    private fun selectedMode(): Mode =
        catalog.mode(getSharedPreferences(PREFERENCES, MODE_PRIVATE).getString(KEY_MODE, null))

    private fun newStagedTake(mode: Mode, id: String) = StagedTake(
        catalog,
        mode,
        id,
        powerSaver = getSystemService(PowerManager::class.java)?.isPowerSaveMode == true,
    )

    /**
     * A direct take becomes a draft (a staged phrase was spoken, or the
     * final needs a decision): its composing text leaves the field first.
     */
    private fun switchToStaged(current: Take): StagedTake {
        current.staged?.let { return it }
        clearComposingText(current)
        current.liveInField = false
        return newStagedTake(current.mode, current.recording.id).also { staged ->
            staged.beginSegment()
            current.staged = staged
        }
    }

    /** A capture's final landed in the draft: show it, and log the timings for #226. */
    private fun settleDraft(current: Take, staged: StagedTake, rawMs: Long) {
        val processedMs = staged.proposal?.elapsedMs
        runCatching {
            Log.i(
                TIMING_TAG,
                "take ${current.recording.id}: stop->raw ${rawMs} ms, processing " +
                    "${processedMs ?: "-"} ms (${staged.plan}), stop->processed ${rawMs + (processedMs ?: 0)} ms",
            )
        }
        renderTake()
        statusView?.text = when {
            staged.plan == ModeCatalog.Plan.RULES_POWER_SAVER -> getString(R.string.staging_power_saver)
            staged.plan == ModeCatalog.Plan.RULES_NO_MODEL -> getString(R.string.staging_no_model)
            processedMs != null -> getString(R.string.staging_ready_processed, processedMs)
            staged.plan == ModeCatalog.Plan.RULES -> getString(R.string.staging_no_changes)
            else -> getString(R.string.staging_ready)
        }
    }

    private fun buildModePicker() {
        val list = modeList ?: return
        list.removeAllViews()
        catalog.modes.filter { it.id != VERBATIM }.forEach { mode ->
            val chip = Button(this, null, android.R.attr.buttonBarButtonStyle).apply {
                text = mode.name
                isAllCaps = false
                contentDescription = mode.description
                setOnClickListener { chooseMode(mode) }
            }
            list.addView(chip)
        }
    }

    private fun toggleModePicker() {
        val picker = modePicker ?: return
        val show = picker.visibility != View.VISIBLE
        picker.visibility = if (show) View.VISIBLE else View.GONE
        if (show) statusView?.setText(R.string.staging_mode_picker_hint)
    }

    /**
     * The picked mode is the default for the next takes; an open draft or
     * a running direct take switches to it at once.
     */
    private fun chooseMode(mode: Mode) {
        getSharedPreferences(PREFERENCES, MODE_PRIVATE).edit().putString(KEY_MODE, mode.id).apply()
        modePicker?.visibility = View.GONE
        val current = take
        when {
            current == null || current.sensitive || current.ready != null -> Unit
            current.staged != null -> {
                current.mode = mode
                current.manualLocked = true
                current.staged?.switchMode(mode)
            }
            // A stopped direct take whose final is still on its way switches too.
            current.capturing || current.awaitingFinal -> {
                current.mode = mode
                current.manualLocked = true
                if (mode.processingDelivery == STAGED) {
                    val staged = switchToStaged(current)
                    // Picked by hand: it outranks a phrase the final may still carry.
                    staged.switchMode(mode)
                    current.lastPartial?.let(staged::partial)
                }
            }
        }
        renderTake()
    }

    /** A tap on a draft word selects it for Delete word. */
    private fun onDraftTouch(view: View, event: MotionEvent): Boolean {
        val staged = take?.staged ?: return false
        if (take?.capturing == true || take?.awaitingFinal == true || staged.busy) return false
        if (event.action != MotionEvent.ACTION_UP) return event.action == MotionEvent.ACTION_DOWN
        val text = (view as TextView).text.toString()
        val offset = view.getOffsetForPosition(event.x, event.y).coerceIn(0, text.length)
        staged.selectWordAt(text.codePointCount(0, offset))
        renderTake()
        statusView?.text = if (staged.selection != null) getString(R.string.staging_word_selected) else null
        view.performClick()
        return true
    }

    /** The draft on screen: command spans struck through, the live tail muted, the selection marked. */
    private fun renderDraft(staged: StagedTake): CharSequence {
        val text = staged.displayText()
        val styled = SpannableStringBuilder(text)
        fun utf16(codePoint: Int) = text.offsetByCodePoints(0, codePoint.coerceIn(0, text.codePointCount(0, text.length)))
        val muted = getColor(R.color.starling_muted)
        if (!staged.showingProcessed) {
            staged.draft.regions().forEach { region ->
                val start = utf16(region.span.start)
                val end = utf16(region.span.end)
                when (region.kind) {
                    RegionKind.COMMAND -> {
                        styled.setSpan(ForegroundColorSpan(muted), start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                        styled.setSpan(StrikethroughSpan(), start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                    }
                    RegionKind.PARTIAL -> {
                        styled.setSpan(ForegroundColorSpan(muted), start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                        styled.setSpan(StyleSpan(Typeface.ITALIC), start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                    }
                    else -> Unit
                }
            }
        }
        staged.selection?.let { word ->
            styled.setSpan(
                BackgroundColorSpan(getColor(R.color.starling_line)),
                utf16(word.first),
                utf16(word.last + 1),
                Spanned.SPAN_EXCLUSIVE_EXCLUSIVE,
            )
        }
        return styled
    }

    private fun decisionText(staged: StagedTake): String? = when (val decision = staged.decision) {
        is StagedTake.Decision.Phrase -> getString(R.string.staging_matched_phrase, decision.phrase, decision.mode.name)
        StagedTake.Decision.Literal -> getString(R.string.staging_matched_literal)
        is StagedTake.Decision.Instruction -> getString(R.string.staging_instruction, decision.instruction)
        null -> null
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

    /**
     * A capture failed before its text existed. A draft keeps whatever
     * earlier captures put there; anything else ends the take.
     */
    private fun failCapture(current: Take, statusRes: Int?, detail: String? = null) {
        if (keepDraft(current)) {
            statusView?.text = listOfNotNull(statusRes?.let(::getString), detail).joinToString(" ")
            return
        }
        endTake(current, statusRes, detail)
    }

    /** Drops the failed capture from the take's draft; true when the draft still has text to keep. */
    private fun keepDraft(current: Take): Boolean {
        val staged = current.staged ?: return false
        if (take !== current) return false
        val kept = if (current.replacing) {
            staged.cancelCorrection()
            true
        } else {
            staged.abandonSegment()
        }
        if (kept) {
            current.awaitingFinal = false
            renderTake()
        }
        return kept
    }

    /**
     * A private take is deleted outright; any other keeps whatever audio it
     * captured, retryable in Starling (RecordingStore.salvageCapture).
     */
    private fun discardOrFail(recording: Recording, sensitive: Boolean, message: String) {
        runCatching {
            if (sensitive) application.recordings.delete(recording.id)
            else application.recordings.salvageCapture(recording.id, message)
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
        val bound = current?.target?.let { targetGuard.isCurrent(it, currentInputConnection) } == true
        val staged = current?.staged
        modeChip?.text = getString(R.string.staging_mode_chip, (staged?.mode ?: current?.mode ?: selectedMode()).name)
        val decision = staged?.let(::decisionText)
        decisionRow?.visibility =
            if (decision != null && !current.capturing && !current.awaitingFinal) View.VISIBLE else View.GONE
        decisionView?.text = decision
        // Undoing the literal escape means the word "literal" was meant.
        decisionUndo?.setText(
            if (staged?.decision == StagedTake.Decision.Literal) R.string.staging_undo_literal else R.string.staging_that_was_literal,
        )
        if (staged != null) {
            val settled = !current.capturing && !current.awaitingFinal && !staged.busy
            transcriptView?.visibility = View.VISIBLE
            transcriptView?.text = renderDraft(staged)
            draftTools?.visibility = if (settled) View.VISIBLE else View.GONE
            viewToggle?.visibility = when {
                staged.proposal != null || staged.canRevertToRaw -> View.VISIBLE
                else -> View.INVISIBLE
            }
            viewToggle?.setText(
                when {
                    staged.proposal == null -> R.string.staging_back_to_raw
                    staged.showingProcessed -> R.string.staging_show_raw
                    else -> R.string.staging_show_processed
                },
            )
            insertButton?.visibility = if (settled && bound && staged.hasPayload) View.VISIBLE else View.GONE
            insertButton?.text = if (staged.mode.delivery == INSERT_ENTER) insertActionLabel() else getString(R.string.keyboard_insert)
            copyButton?.visibility = if (settled && !bound) View.VISIBLE else View.GONE
            return
        }
        draftTools?.visibility = View.GONE
        insertButton?.setText(R.string.keyboard_insert)
        val ready = current?.ready
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
            val manager = getSystemService(InputMethodManager::class.java) ?: return
            @Suppress("DEPRECATION")
            if (!manager.switchToLastInputMethod(token)) manager.switchToNextInputMethod(token, false)
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
        const val KEY_MODE = "mode"
        const val STAGED = "staged"
        const val VERBATIM = "verbatim"
        const val INSERT_ENTER = "insert_enter"
        const val TIMING_TAG = "StarlingTiming"
    }

    /**
     * One take, from Record until its text is inserted, copied or given up.
     * Main thread only; the capture worker sees only the stream session it
     * was handed at start.
     */
    private class Take(
        val recording: Recording,
        /** The field the take was started in; it attaches only to that field. */
        val field: EditorField,
        val sensitive: Boolean,
        /** The mode the take runs in; a spoken phrase or the picker may change it. */
        var mode: Mode,
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

        /** The text of that composing region, as last written. */
        var composedText: String? = null

        /**
         * The text around the cursor when the composing region started, for
         * the boundary of the live text (#341). In memory only, never stored.
         */
        var liveBoundary: InsertionBoundary.Context? = null

        /** The final's mode is verbatim: its delivery skips the boundary rules. */
        var verbatim = false

        /** The delivered boundary adjustment could not be saved as a derived revision. */
        var derivedUnsaved = false

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

        /** The take's draft, for a staged mode (or a direct take that became one). */
        var staged: StagedTake? = null

        /** When Stop was tapped, for the stop→raw/processed timings. */
        var stoppedAtNanos = 0L

        /** The mode was picked by hand during this take; spoken phrases no longer switch it. */
        var manualLocked = false

        /** Stopped, but its transcript has not settled yet. */
        var awaitingFinal = false

        /** The capture is a spoken correction for the draft's selected word, not a new segment. */
        var replacing = false
    }
}
