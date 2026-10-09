package dev.starling.mobile.processing

/**
 * One staged dictation in the keyboard (#302): the editable draft above the
 * keyboard, from the first live partial until the user inserts, copies or
 * discards it. Android-free, so every decision here is unit-tested.
 *
 * All text lives in a [StagedDraft] (the #293 contract): each capture is one
 * segment whose partials are the live tail and whose final is an immutable
 * raw attempt; a leading mode phrase and a trailing instruction become
 * command regions, which never reach the field. Processing reads the
 * draft's payload at one revision and comes back as a proposal; it never
 * replaces the draft by itself. The user sees the processed proposal by
 * default and can always flip back to the raw text, and the text that is
 * delivered is exactly the view they had on screen.
 *
 * Offsets are Unicode code points, as everywhere in the contract.
 */
class StagedTake(
    private val catalog: ModeCatalog,
    /** The mode the draft started in, before any spoken phrase. */
    private val startMode: Mode,
    draftId: String,
    private val powerSaver: Boolean = false,
    private val nanoTime: () -> Long = System::nanoTime,
) {
    val draft = StagedDraft(draftId, draftId)

    var mode: Mode = startMode
        private set

    /** Why the draft looks the way it does, for the one-line explanation. */
    var decision: Decision? = null
        private set

    /** The open processed proposal for the current revision, if any. */
    var proposal: Proposal? = null
        private set

    /** Whether the processed proposal (rather than the draft) is on screen. */
    var showProcessed = true
        private set

    /** The selected word in the on-screen text, in code points. */
    var selection: IntRange? = null
        private set

    var plan: ModeCatalog.Plan = catalog.plan(startMode, powerSaver)
        private set

    private var segment = -1
    private var separator = ""
    private var requests = 0
    private var phraseIsLiteral = false
    private var instructionIsLiteral = false

    // Segment 0's routing and the last segment's instruction, kept so a
    // rebuild after "that was literal" can re-mark the other one.
    private var phraseEnd = 0
    private var phraseMode: Mode? = null
    private var instructionSpan: IntRange? = null

    sealed interface Decision {
        data class Phrase(val phrase: String, val mode: Mode) : Decision
        data object Literal : Decision
        data class Instruction(val instruction: String) : Decision
    }

    data class Proposal(val requestId: String, val text: String, val elapsedMs: Long)

    val recording: Boolean get() = draft.regions().any { it.kind == RegionKind.PARTIAL }

    /** A new capture adds a segment at the end of the draft. */
    fun beginSegment() {
        segment += 1
        val text = draft.text()
        separator = if (text.isEmpty() || text.last().isWhitespace()) "" else " "
        selection = null
    }

    fun partial(text: String) {
        draft.partial(segment, separator + text)
    }

    /**
     * The final text of the current segment. The first segment of a draft
     * may start with a mode phrase; the newest segment may end with a
     * spoken instruction. Then the draft is processed.
     */
    fun final(attemptId: String, text: String) {
        draft.final(segment, attemptId, separator + text)
        val start = draft.text().codePointCount() - (separator + text).codePointCount()
        if (segment == 0 && !phraseIsLiteral) {
            val routed = catalog.route(text, startMode, secure = false)
            val span = routed.prefixSpanCodepoints
            if (span != null && routed.mode != null && span.second > span.first) {
                phraseEnd = start + separator.codePointCount() + span.second
                phraseMode = catalog.mode(routed.mode)
                mode = catalog.mode(routed.mode)
                plan = catalog.plan(mode, powerSaver)
                decision = if (routed.source.startsWith("escape:")) {
                    Decision.Literal
                } else {
                    Decision.Phrase(routed.source.removePrefix("phrase:"), mode)
                }
            }
        }
        if (!instructionIsLiteral && mode.behavior != "verbatim") {
            val split = catalog.instructions.split(separator + text, null)
            val delimiter = split.delimiterSpan
            if (split.matched && delimiter != null && start + delimiter.first >= phraseEnd) {
                instructionSpan = (start + delimiter.first) until draft.text().codePointCount()
                decision = Decision.Instruction(split.instruction)
            }
        }
        markCommands()
        process()
    }

    /**
     * The current capture failed: its live tail leaves the draft (partials
     * are not durable; the contract's restart drops them) and the earlier
     * text stays. True when the draft still has text.
     */
    fun abandonSegment(): Boolean {
        draft.crash()
        process()
        return draft.text().isNotEmpty()
    }

    /** Switches the draft's mode by hand (the mode picker) and processes again. */
    fun switchMode(target: Mode) {
        mode = target
        plan = catalog.plan(mode, powerSaver)
        if (decision is Decision.Phrase || decision is Decision.Literal) {
            // A hand-picked mode overrides the spoken one; the phrase text
            // stays out of the payload all the same.
            decision = null
        }
        process()
    }

    /**
     * "That was literal": the mode phrase or instruction was meant as
     * words. The draft goes back to its raw attempts with that span kept
     * as ordinary text, and processing runs again.
     */
    fun undoDecision() {
        when (decision) {
            is Decision.Phrase, Decision.Literal -> {
                phraseIsLiteral = true
                phraseEnd = 0
                phraseMode = null
                mode = startMode
                plan = catalog.plan(mode, powerSaver)
            }
            is Decision.Instruction -> {
                instructionIsLiteral = true
                instructionSpan = null
            }
            null -> return
        }
        decision = null
        draft.revertRaw()
        markCommands()
        process()
    }

    /** The text on screen: the processed proposal, or the draft itself. */
    fun displayText(): String {
        val open = proposal
        return if (showProcessed && open != null) open.text else draft.text()
    }

    /** Whether the on-screen text is the processed proposal. */
    val showingProcessed: Boolean get() = showProcessed && proposal != null

    fun toggleView() {
        showProcessed = !showProcessed
        selection = null
    }

    /**
     * "Back to raw" after the processed text was taken into the draft: the
     * raw attempts come back (edits are dropped) and processing runs again.
     */
    fun backToRaw() {
        draft.revertRaw()
        markCommands()
        showProcessed = false
        process()
        showProcessed = false
    }

    /** Whether the draft text currently is an accepted processed revision. */
    val processedInDraft: Boolean
        get() = draft.regions().any { it.kind == RegionKind.PROCESSED }

    /**
     * Selects the word at [offset] (code points into [displayText]), or
     * clears the selection when that word is already selected. Editing a
     * processed proposal takes it into the draft first, so the edit applies
     * to the text the user is looking at.
     */
    fun selectWordAt(offset: Int) {
        val text = displayText()
        val word = wordAt(text, offset) ?: run { selection = null; return }
        selection = if (selection == word) null else word
    }

    /** Deletes the selected word, or the last word when nothing is selected. */
    fun deleteWord() {
        if (recording) return
        takeProcessedIntoDraft()
        val text = draft.text()
        val target = selection ?: lastWord(text) ?: return
        // One adjacent space goes with the word, so no double space is left.
        val cps = text.codePoints().toArray()
        var start = target.first
        var end = target.last + 1
        if (end < cps.size && cps[end] == ' '.code) end += 1
        else if (start > 0 && cps[start - 1] == ' '.code) start -= 1
        draft.delete(start, end)
        selection = null
        process()
        // The user edited the text on screen; keep showing it, not a new proposal.
        showProcessed = false
    }

    /**
     * The text to deliver, exactly as shown. Taking the processed view
     * accepts the proposal into the draft first, so the draft records what
     * was delivered. Command regions never leave the draft.
     */
    fun deliveryText(): String {
        takeProcessedIntoDraft()
        val text = draft.payloadText()
        return if (instructionSpan != null) text.trimEnd() else text
    }

    /**
     * Records one delivery for [targetDigest] (the editor binding); false
     * when this revision was already delivered there, so a double tap or a
     * replayed callback cannot insert twice.
     */
    fun deliver(deliveryId: String, targetDigest: String): Boolean =
        draft.deliver(deliveryId, targetDigest) == Outcome.DELIVERED

    private fun takeProcessedIntoDraft() {
        val open = proposal ?: return
        if (!showProcessed) return
        if (draft.accept(open.requestId) == Outcome.APPLIED) {
            proposal = null
            showProcessed = false
        }
    }

    private fun markCommands() {
        if (phraseEnd > 0 && !phraseIsLiteral) {
            draft.markCommand(0, phraseEnd, CommandKind.MODE_PHRASE)
        }
        instructionSpan?.let { span ->
            draft.markCommand(span.first, span.last + 1, CommandKind.TRAILING_INSTRUCTION)
        }
    }

    /**
     * Runs the deterministic step on the draft's payload at its current
     * revision. Rules take microseconds, so the result is judged at once;
     * it still goes through the contract as a request and a proposal.
     */
    private fun process() {
        proposal = null
        selection = null
        showProcessed = true
        if (recording || plan == ModeCatalog.Plan.NONE) return
        val requestId = "processing-${++requests}"
        if (draft.request(requestId) != Outcome.PENDING) return
        val input = draft.payloadText()
        val started = nanoTime()
        val output = catalog.commands.apply(
            input,
            mode.language,
            mode.spokenCommands,
            mode.snippets.map { Snippet(it.spoken, it.expansion) },
        )
        val elapsedMs = (nanoTime() - started) / 1_000_000
        if (draft.result(requestId, true, output) != Outcome.CURRENT) return
        if (output == input) {
            draft.reject(requestId)
            return
        }
        proposal = Proposal(requestId, output, elapsedMs)
    }

    companion object {
        private fun String.codePointCount(): Int = codePointCount(0, length)

        /** The word (maximal non-whitespace run) at [offset], in code points. */
        internal fun wordAt(text: String, offset: Int): IntRange? {
            val cps = text.codePoints().toArray()
            if (offset !in cps.indices || Character.isWhitespace(cps[offset])) return null
            var start = offset
            var end = offset
            while (start > 0 && !Character.isWhitespace(cps[start - 1])) start -= 1
            while (end + 1 < cps.size && !Character.isWhitespace(cps[end + 1])) end += 1
            return start..end
        }

        internal fun lastWord(text: String): IntRange? {
            val cps = text.codePoints().toArray()
            var end = cps.size - 1
            while (end >= 0 && Character.isWhitespace(cps[end])) end -= 1
            return if (end < 0) null else wordAt(text, end)
        }
    }
}
