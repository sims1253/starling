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
 * Commands are remembered per segment, in that segment's raw coordinates:
 * they are marked once, when the segment's final lands at the end of the
 * draft, and again for every segment whenever the draft is rebuilt from its
 * raw attempts. Edits in between never shift them onto other text.
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

    /** A mode picked by hand; it outranks any spoken phrase. */
    private var manualMode: Mode? = null

    /** The mode a leading phrase picked, while that phrase stands. */
    private var phraseMode: Mode? = null

    val mode: Mode get() = manualMode ?: phraseMode ?: startMode

    /** The open processed proposal for the current revision, if any. */
    var proposal: Proposal? = null
        private set

    /** Whether the processed proposal (rather than the draft) is on screen. */
    var showProcessed = true
        private set

    /** The selected word in the on-screen text, in code points. */
    var selection: IntRange? = null
        private set

    val plan: ModeCatalog.Plan get() = catalog.plan(mode, powerSaver)

    private var segment = -1
    private var separator = ""
    private var requests = 0

    /** The raw text of every finalized segment (separator included), by segment. */
    private val segmentTexts = sortedMapOf<Int, String>()

    /** Commands that stand, in their segment's raw coordinates. */
    private val commands = mutableListOf<SegmentCommand>()

    private class SegmentCommand(
        val segment: Int,
        val start: Int,
        val end: Int,
        val kind: CommandKind,
        val decision: Decision,
    )

    sealed interface Decision {
        data class Phrase(val phrase: String, val mode: Mode) : Decision
        data object Literal : Decision
        data class Instruction(val instruction: String) : Decision
    }

    data class Proposal(val requestId: String, val text: String, val elapsedMs: Long)

    /**
     * The newest spoken decision still standing, for the one-line
     * explanation and its undo. A phrase overruled by a hand-picked mode is
     * still kept out of the text but is no longer the explanation.
     */
    val decision: Decision?
        get() = commands.lastOrNull { it.kind == CommandKind.TRAILING_INSTRUCTION }?.decision
            ?: commands.firstOrNull { it.kind == CommandKind.MODE_PHRASE && manualMode == null }?.decision

    val recording: Boolean get() = draft.regions().any { it.kind == RegionKind.PARTIAL }

    /**
     * A new capture adds a segment at the end of the draft. What the user
     * was looking at becomes the base: a processed view on screen is taken
     * into the draft, so the live tail shows right after it.
     */
    fun beginSegment() {
        takeProcessedIntoDraft()
        proposal = null
        selection = null
        segment += 1
        val text = draft.text()
        separator = if (text.isEmpty() || text.last().isWhitespace()) "" else " "
    }

    fun partial(text: String) {
        draft.partial(segment, separator + text)
    }

    /**
     * The final text of the current segment. The first segment of a draft
     * may start with a mode phrase; any segment may end with a spoken
     * instruction. Then the draft is processed.
     */
    fun final(attemptId: String, text: String) {
        val segmentText = separator + text
        draft.final(segment, attemptId, segmentText)
        segmentTexts[segment] = segmentText
        val found = mutableListOf<SegmentCommand>()
        var phraseEnd = 0
        if (segment == 0) {
            val routed = catalog.route(text, startMode, secure = false)
            val span = routed.prefixSpanCodepoints
            if (span != null && routed.mode != null && span.second > span.first) {
                val routedMode = catalog.mode(routed.mode)
                phraseEnd = separator.codePointCount() + span.second
                phraseMode = routedMode
                val decision = if (routed.source.startsWith("escape:")) {
                    Decision.Literal
                } else {
                    Decision.Phrase(routed.source.removePrefix("phrase:"), routedMode)
                }
                found += SegmentCommand(segment, 0, phraseEnd, CommandKind.MODE_PHRASE, decision)
            }
        }
        if (mode.behavior != VERBATIM) {
            val split = catalog.instructions.split(segmentText, mode.language)
            val delimiter = split.delimiterSpan
            if (split.matched && delimiter != null) {
                // The whitespace before the delimiter goes with it, so the
                // payload ends where the spoken text did.
                val cps = segmentText.codePoints().toArray()
                var start = delimiter.first
                while (start > 0 && Character.isWhitespace(cps[start - 1])) start -= 1
                if (start >= phraseEnd) {
                    found += SegmentCommand(
                        segment,
                        start,
                        cps.size,
                        CommandKind.TRAILING_INSTRUCTION,
                        Decision.Instruction(split.instruction),
                    )
                }
            }
        }
        // The segment's final is the end of the draft right now.
        val base = draft.text().codePointCount() - segmentText.codePointCount()
        found.forEach { draft.markCommand(base + it.start, base + it.end, it.kind) }
        commands += found
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
        manualMode = target
        if (!recording) process()
    }

    /**
     * "That was literal" for the decision on screen: that phrase or
     * instruction was meant as words. The draft goes back to its raw
     * attempts with every other command still set aside, and is processed
     * again.
     */
    fun undoDecision() {
        if (recording) return
        val shown = decision ?: return
        val command = commands.first { it.decision === shown }
        commands.remove(command)
        if (command.kind == CommandKind.MODE_PHRASE) phraseMode = null
        rebuildFromRaw()
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
        if (recording) return
        rebuildFromRaw()
        process()
        showProcessed = false
    }

    /** Whether the draft text currently is an accepted processed revision. */
    val processedInDraft: Boolean
        get() = draft.regions().any { it.kind == RegionKind.PROCESSED }

    /**
     * Selects the word at [offset] (code points into [displayText]), or
     * clears the selection when that word is already selected.
     */
    fun selectWordAt(offset: Int) {
        val word = wordAt(displayText(), offset) ?: run { selection = null; return }
        selection = if (selection == word) null else word
    }

    /** Deletes the selected word, or the last word when nothing is selected. */
    fun deleteWord() {
        if (recording) return
        val target = selection ?: lastWord(displayText()) ?: return
        takeProcessedIntoDraft()
        val cps = draft.text().codePoints().toArray()
        // One adjacent space goes with the word, so no double space is left.
        var start = target.first
        var end = target.last + 1
        if (end < cps.size && cps[end] == ' '.code) end += 1
        else if (start > 0 && cps[start - 1] == ' '.code) start -= 1
        draft.delete(start, end)
        afterEdit()
    }

    /**
     * Replaces the selected word with [text] (a spoken correction): user
     * text in the draft, at the word's place. False when nothing is selected.
     */
    fun replaceSelection(text: String): Boolean {
        if (recording) return false
        val target = selection ?: return false
        takeProcessedIntoDraft()
        draft.delete(target.first, target.last + 1)
        draft.insert(target.first, text)
        afterEdit()
        return true
    }

    /**
     * The text to deliver, exactly as shown. Taking the processed view
     * accepts the proposal into the draft first, so the draft records what
     * was delivered. Command regions never leave the draft.
     */
    fun deliveryText(): String {
        takeProcessedIntoDraft()
        return draft.payloadText()
    }

    /**
     * Records one delivery for [targetDigest] (the editor binding); false
     * when this revision was already delivered there.
     */
    fun deliver(deliveryId: String, targetDigest: String): Boolean =
        draft.deliver(deliveryId, targetDigest) == Outcome.DELIVERED

    /** The user edited the text on screen: keep showing it, processed again underneath. */
    private fun afterEdit() {
        selection = null
        process()
        showProcessed = false
    }

    private fun takeProcessedIntoDraft() {
        val open = proposal ?: return
        if (!showProcessed) return
        if (draft.accept(open.requestId) == Outcome.APPLIED) {
            proposal = null
            showProcessed = false
        }
    }

    /** The raw attempts again, with every standing command set aside. */
    private fun rebuildFromRaw() {
        draft.revertRaw()
        var base = 0
        segmentTexts.forEach { (index, text) ->
            commands.filter { it.segment == index }.forEach {
                draft.markCommand(base + it.start, base + it.end, it.kind)
            }
            base += text.codePointCount()
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
        private const val VERBATIM = "verbatim"

        private fun String.codePointCount(): Int = codePointCount(0, length)

        /**
         * The word at [offset], in code points: the maximal non-whitespace
         * run, without the punctuation a recognizer attaches to its end, so
         * a correction keeps the comma that follows the word.
         */
        internal fun wordAt(text: String, offset: Int): IntRange? {
            val cps = text.codePoints().toArray()
            if (offset !in cps.indices || Character.isWhitespace(cps[offset])) return null
            var start = offset
            var end = offset
            while (start > 0 && !Character.isWhitespace(cps[start - 1])) start -= 1
            while (end + 1 < cps.size && !Character.isWhitespace(cps[end + 1])) end += 1
            while (end > start && cps[end] < 0x10000 && cps[end].toChar() in ATTACHED) end -= 1
            return start..end
        }

        private const val ATTACHED = ",.;:!?"

        internal fun lastWord(text: String): IntRange? {
            val cps = text.codePoints().toArray()
            var end = cps.size - 1
            while (end >= 0 && Character.isWhitespace(cps[end])) end -= 1
            return if (end < 0) null else wordAt(text, end)
        }
    }
}
