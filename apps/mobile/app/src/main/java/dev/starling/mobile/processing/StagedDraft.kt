package dev.starling.mobile.processing

/**
 * The staged draft of one take: a single editable text with typed regions over
 * it (#293). A Kotlin port of `Draft` in `tests/staging.py`, which is the
 * executable semantics; `packages/contracts/mode-routing/fixtures/staging.json`
 * is replayed against both, and a behaviour change here is a contract change.
 *
 * Why it is shaped this way:
 * - Every offset (at, start, end, spans) counts Unicode code points, the
 *   contract's `span_encoding`. Kotlin strings are UTF-16, so all index
 *   arithmetic goes through the code-point helpers below; a raw `length` or
 *   `substring` would split surrogate pairs.
 * - Attempts are the immutable recognitions. No edit, proposal or failure
 *   touches them, so [rawText] rebuilds exactly what the recognizer produced.
 * - [revision] counts text changes only. A request records the revision it
 *   read; its result is a proposal that is current only while the draft is
 *   still at that revision. A result never changes the text by itself:
 *   [accept] (the user) or [swapCheck] (direct delivery into an unchanged
 *   target) does.
 * - Typing or deleting inside a live partial pins its segment, so the user's
 *   words are never overwritten by later recognition of that segment.
 * - A deleted draft ignores everything except [result] and [crash]: a late
 *   result is discarded, and a restart still settles what it must.
 *
 * Not thread-safe; the caller serializes operations on one draft.
 */
class StagedDraft(val draftId: String, val captureId: String) {
    /** Counts text changes; a request's base revision is compared against it. */
    var revision: Int = 0
        private set

    var deleted: Boolean = false
        private set

    private var pieces = mutableListOf<Piece>()
    private val attemptLog = mutableListOf<Attempt>()
    private val pinnedSegments = mutableListOf<Int>()
    private val finalizedSegments = mutableListOf<Int>()
    private val requestLog = mutableListOf<RequestEntry>()
    private val proposalLog = mutableListOf<ProposalEntry>()
    private val deliveryLog = mutableListOf<Delivery>()

    // ------------------------------------------------------------------ //
    // Views
    // ------------------------------------------------------------------ //

    fun text(): String = pieces.joinToString("") { it.text }

    /** The transform input: every region except commands. */
    fun payloadText(): String = pieces.filter { it.kind != RegionKind.COMMAND }.joinToString("") { it.text }

    /** The trailing instruction, or null when there is none. */
    fun instruction(): String? {
        val parts = pieces.filter {
            it.kind == RegionKind.COMMAND && it.command == CommandKind.TRAILING_INSTRUCTION
        }
        return if (parts.isEmpty()) null else parts.joinToString("") { it.text }
    }

    /** The latest final attempt per segment, in segment order: the recognition as it arrived. */
    fun rawText(): String {
        val latest = sortedMapOf<Int, String>()
        for (attempt in attemptLog) latest[attempt.segment] = attempt.text
        return latest.values.joinToString("")
    }

    fun regions(): List<Region> {
        val out = ArrayList<Region>(pieces.size)
        var pos = 0
        for (piece in pieces) {
            val end = pos + cpLen(piece.text)
            out += Region(
                kind = piece.kind,
                text = piece.text,
                span = Span(pos, end),
                segment = piece.segment,
                attemptId = piece.attemptId,
                command = piece.command,
                requestId = piece.requestId,
            )
            pos = end
        }
        return out
    }

    fun attempts(): List<Attempt> = attemptLog.toList()

    fun pinnedSegments(): List<Int> = pinnedSegments.sorted()

    fun requests(): List<Request> = requestLog.map {
        Request(it.requestId, it.baseRevision, it.retryOf, it.input, it.instruction, it.status)
    }

    /** Open proposals read as [ProposalStatus.CURRENT] or [ProposalStatus.STALE] against the revision. */
    fun proposals(): List<Proposal> = proposalLog.map { entry ->
        val status = when (entry.life) {
            ProposalLife.OPEN ->
                if (entry.baseRevision == revision) ProposalStatus.CURRENT else ProposalStatus.STALE
            ProposalLife.SUPERSEDED -> ProposalStatus.SUPERSEDED
            ProposalLife.ACCEPTED -> ProposalStatus.ACCEPTED
            ProposalLife.REJECTED -> ProposalStatus.REJECTED
        }
        Proposal(entry.requestId, entry.baseRevision, entry.text, status)
    }

    fun deliveries(): List<Delivery> = deliveryLog.toList()

    // ------------------------------------------------------------------ //
    // Operations. Each returns the oracle's outcome.
    // ------------------------------------------------------------------ //

    fun partial(segment: Int, text: String): Outcome {
        if (deleted) return Outcome.IGNORED_DELETED
        if (segment in pinnedSegments) return Outcome.IGNORED_PINNED
        if (segment in finalizedSegments) return Outcome.IGNORED_FINAL
        val next = pieces.toMutableList()
        val index = partialIndex(segment)
        if (index == null) {
            next.add(Piece(RegionKind.PARTIAL, text, segment = segment))
        } else {
            next[index] = next[index].copy(text = text)
        }
        return if (setText(next)) Outcome.APPLIED else Outcome.UNCHANGED
    }

    fun final(segment: Int, attemptId: String, text: String): Outcome {
        if (deleted) return Outcome.IGNORED_DELETED
        val known = attemptLog.firstOrNull { it.attemptId == attemptId }
        if (known != null) {
            return if (known.segment == segment && known.text == text) Outcome.DUPLICATE
            else Outcome.ATTEMPT_CONFLICT
        }
        attemptLog += Attempt(attemptId, segment, text)
        val firstFinal = segment !in finalizedSegments
        if (firstFinal) finalizedSegments += segment
        if (segment in pinnedSegments) return Outcome.RECORDED_PINNED
        // A re-recognition of a segment the draft already shows is kept as an
        // attempt; the visible text is not replaced behind the user's back.
        if (!firstFinal) return Outcome.RECORDED
        val next = pieces.toMutableList()
        val raw = Piece(RegionKind.RAW, text, segment = segment, attemptId = attemptId)
        val index = partialIndex(segment)
        if (index == null) next.add(raw) else next[index] = raw
        setText(next)
        return Outcome.APPLIED
    }

    fun insert(at: Int, text: String): Outcome {
        if (deleted) return Outcome.IGNORED_DELETED
        if (at !in 0..cpLen(text())) return Outcome.OUT_OF_RANGE
        if (text.isEmpty()) return Outcome.UNCHANGED
        // Typing strictly inside a live partial pins it: the user owns that
        // segment's text from now on.
        var pos = 0
        for (i in pieces.indices) {
            val piece = pieces[i]
            val end = pos + cpLen(piece.text)
            if (piece.kind == RegionKind.PARTIAL && at > pos && at < end) pin(i)
            pos = end
        }
        val index = splitAt(at)
        val next = pieces.toMutableList()
        next.add(index, Piece(RegionKind.USER, text))
        setText(next)
        return Outcome.APPLIED
    }

    fun delete(start: Int, end: Int): Outcome {
        if (deleted) return Outcome.IGNORED_DELETED
        if (!(0 <= start && start <= end && end <= cpLen(text()))) return Outcome.OUT_OF_RANGE
        if (start == end) return Outcome.UNCHANGED
        // Deleting inside a live partial pins it, just like typing.
        var pos = 0
        for (i in pieces.indices) {
            val piece = pieces[i]
            val regionEnd = pos + cpLen(piece.text)
            if (piece.kind == RegionKind.PARTIAL && start < regionEnd && end > pos) pin(i)
            pos = regionEnd
        }
        val next = ArrayList<Piece>(pieces.size)
        pos = 0
        for (piece in pieces) {
            val length = cpLen(piece.text)
            val regionEnd = pos + length
            if (start < regionEnd && end > pos) {
                val left = cpSlice(piece.text, 0, start - pos)
                val right = cpSlice(piece.text, end - pos, length)
                next += piece.copy(text = left + right)
            } else {
                next += piece
            }
            pos = regionEnd
        }
        setText(next)
        return Outcome.APPLIED
    }

    /**
     * Marks `[start, end)` as a command. Splits happen before the partial check,
     * so a refused mark still leaves the region boundaries it created (the
     * oracle does the same; text and revision are untouched).
     */
    fun markCommand(start: Int, end: Int, kind: CommandKind): Outcome {
        if (deleted) return Outcome.IGNORED_DELETED
        if (!(0 <= start && start < end && end <= cpLen(text()))) return Outcome.OUT_OF_RANGE
        val first = splitAt(start)
        val last = splitAt(end)
        val span = pieces.subList(first, last)
        if (span.any { it.kind == RegionKind.PARTIAL }) return Outcome.REFUSED_PARTIAL
        val spanText = span.joinToString("") { it.text }
        span.clear()
        pieces.add(first, Piece(RegionKind.COMMAND, spanText, command = kind))
        return Outcome.APPLIED
    }

    fun request(requestId: String, retryOf: String? = null): Outcome {
        if (deleted) return Outcome.IGNORED_DELETED
        if (findRequest(requestId) != null) return Outcome.DUPLICATE
        // Processing reads finals only; a live tail would make the request
        // stale the moment the next partial lands.
        if (pieces.any { it.kind == RegionKind.PARTIAL }) return Outcome.REFUSED_PARTIAL
        if (retryOf != null) {
            val earlier = findRequest(retryOf) ?: return Outcome.UNKNOWN_REQUEST
            if (earlier.status == RequestStatus.PENDING || earlier.status == RequestStatus.INTERRUPTED) {
                earlier.status = RequestStatus.SUPERSEDED
            }
        }
        requestLog += RequestEntry(requestId, revision, retryOf, payloadText(), instruction())
        return Outcome.PENDING
    }

    fun cancel(requestId: String): Outcome {
        if (deleted) return Outcome.IGNORED_DELETED
        val request = findRequest(requestId) ?: return Outcome.UNKNOWN_REQUEST
        if (request.status != RequestStatus.PENDING && request.status != RequestStatus.INTERRUPTED) {
            return Outcome.NOT_PENDING
        }
        request.status = RequestStatus.CANCELLED
        return Outcome.CANCELLED
    }

    /**
     * A job's result. [completed] false, or a completed result with no [text],
     * is a failure: the request settles and the raw text is untouched.
     */
    fun result(requestId: String, completed: Boolean, text: String?): Outcome {
        // Results run on a deleted draft: they are discarded, not ignored.
        val request = findRequest(requestId) ?: return Outcome.UNKNOWN_REQUEST
        if (deleted) return Outcome.DISCARDED
        val status = request.status
        if (status == RequestStatus.CANCELLED) return Outcome.DISCARDED
        if (status == RequestStatus.SETTLED || findProposal(requestId) != null) return Outcome.DUPLICATE
        if (!completed || text == null) {
            if (status == RequestStatus.PENDING || status == RequestStatus.INTERRUPTED) {
                request.status = RequestStatus.SETTLED
            }
            return Outcome.FAILED
        }
        val proposal = ProposalEntry(requestId, request.baseRevision, text)
        proposalLog += proposal
        if (status == RequestStatus.SUPERSEDED) {
            proposal.life = ProposalLife.SUPERSEDED
            return Outcome.SUPERSEDED
        }
        request.status = RequestStatus.SETTLED
        return if (request.baseRevision == revision) Outcome.CURRENT else Outcome.STALE
    }

    /** Accepting a stale proposal needs [force]: only an explicit user choice takes one. */
    fun accept(requestId: String, force: Boolean = false): Outcome {
        if (deleted) return Outcome.IGNORED_DELETED
        val proposal = findProposal(requestId) ?: return Outcome.UNKNOWN_REQUEST
        if (proposal.life != ProposalLife.OPEN) return Outcome.NOT_OPEN
        if (proposal.baseRevision != revision && !force) return Outcome.STALE_REJECTED
        acceptProposal(proposal)
        return Outcome.APPLIED
    }

    fun reject(requestId: String): Outcome {
        if (deleted) return Outcome.IGNORED_DELETED
        val proposal = findProposal(requestId) ?: return Outcome.UNKNOWN_REQUEST
        if (proposal.life != ProposalLife.OPEN) return Outcome.NOT_OPEN
        proposal.life = ProposalLife.REJECTED
        return Outcome.REJECTED
    }

    /** Rebuilds the text from the latest attempts; live partials of unfinished segments stay at the tail. */
    fun revertRaw(): Outcome {
        if (deleted) return Outcome.IGNORED_DELETED
        val latest = sortedMapOf<Int, Attempt>()
        for (attempt in attemptLog) latest[attempt.segment] = attempt
        val next = latest.values
            .map { Piece(RegionKind.RAW, it.text, segment = it.segment, attemptId = it.attemptId) }
            .toMutableList()
        next += pieces.filter { it.kind == RegionKind.PARTIAL }
        return if (setText(next)) Outcome.APPLIED else Outcome.UNCHANGED
    }

    fun deliver(deliveryId: String, targetDigest: String): Outcome {
        if (deleted) return Outcome.IGNORED_DELETED
        if (deliveryLog.any { it.deliveryId == deliveryId }) return Outcome.DUPLICATE
        if (deliveryLog.any { it.revision == revision && it.targetDigest == targetDigest }) return Outcome.DUPLICATE
        if (pieces.any { it.kind == RegionKind.PARTIAL }) return Outcome.REFUSED_PARTIAL
        deliveryLog += Delivery(deliveryId, revision, targetDigest)
        return Outcome.DELIVERED
    }

    /**
     * Direct delivery: the processed text may replace the raw text already in
     * the target only if the proposal is current and the target still holds
     * exactly the last delivery made at its base revision.
     */
    fun swapCheck(requestId: String, targetDigest: String): Outcome {
        if (deleted) return Outcome.IGNORED_DELETED
        val proposal = findProposal(requestId)
        if (proposal == null || proposal.life != ProposalLife.OPEN) return Outcome.KEEP
        if (proposal.baseRevision != revision) return Outcome.KEEP
        val delivered = deliveryLog.filter { it.revision == proposal.baseRevision }
        if (delivered.isEmpty() || delivered.last().targetDigest != targetDigest) return Outcome.KEEP
        acceptProposal(proposal)
        return Outcome.SWAP
    }

    /** Cancels the requests still in flight; later results are discarded. */
    fun deleteDraft(): Outcome {
        if (deleted) return Outcome.IGNORED_DELETED
        deleted = true
        for (request in requestLog) {
            if (request.status == RequestStatus.PENDING || request.status == RequestStatus.INTERRUPTED) {
                request.status = RequestStatus.CANCELLED
            }
        }
        return Outcome.DELETED
    }

    /**
     * Process death and restart. Live partials are not durable, so they are
     * dropped; requests in flight become interrupted, and a result that still
     * arrives for one is judged like any other against its base.
     */
    fun crash(): Outcome {
        for (request in requestLog) {
            if (request.status == RequestStatus.PENDING) request.status = RequestStatus.INTERRUPTED
        }
        if (!deleted) setText(pieces.filter { it.kind != RegionKind.PARTIAL })
        return Outcome.RESTARTED
    }

    // ------------------------------------------------------------------ //
    // Internals
    // ------------------------------------------------------------------ //

    private fun findRequest(requestId: String): RequestEntry? =
        requestLog.firstOrNull { it.requestId == requestId }

    private fun findProposal(requestId: String): ProposalEntry? =
        proposalLog.firstOrNull { it.requestId == requestId }

    private fun partialIndex(segment: Int): Int? =
        pieces.indexOfFirst { it.kind == RegionKind.PARTIAL && it.segment == segment }.takeIf { it >= 0 }

    /** Installs [next] (dropping empty pieces, merging adjacent user text) and bumps the revision on a text change. */
    private fun setText(next: List<Piece>): Boolean {
        val before = text()
        val cleaned = mutableListOf<Piece>()
        for (piece in next) {
            if (piece.text.isEmpty()) continue
            val last = cleaned.lastOrNull()
            if (last != null && piece.kind == RegionKind.USER && last.kind == RegionKind.USER) {
                cleaned[cleaned.lastIndex] = last.copy(text = last.text + piece.text)
                continue
            }
            cleaned += piece
        }
        pieces = cleaned
        val changed = text() != before
        if (changed) revision++
        return changed
    }

    /** Pins the segment of the partial at [index]; the partial's text becomes the user's. */
    private fun pin(index: Int) {
        val piece = pieces[index]
        val segment = checkNotNull(piece.segment) { "a partial always names its segment" }
        if (segment !in pinnedSegments) pinnedSegments += segment
        pieces[index] = Piece(RegionKind.USER, piece.text)
    }

    /** Splits the piece containing code point [at] so a boundary lies there; returns the index of the piece starting at [at]. */
    private fun splitAt(at: Int): Int {
        var pos = 0
        for (i in pieces.indices) {
            val piece = pieces[i]
            val length = cpLen(piece.text)
            if (at == pos) return i
            if (at > pos && at < pos + length) {
                val cut = at - pos
                pieces[i] = piece.copy(text = cpSlice(piece.text, 0, cut))
                pieces.add(i + 1, piece.copy(text = cpSlice(piece.text, cut, length)))
                return i + 1
            }
            pos += length
        }
        return pieces.size
    }

    private fun acceptProposal(proposal: ProposalEntry) {
        setText(listOf(Piece(RegionKind.PROCESSED, proposal.text, requestId = proposal.requestId)))
        proposal.life = ProposalLife.ACCEPTED
    }
}

enum class RegionKind(val wire: String) {
    RAW("raw"),
    PARTIAL("partial"),
    USER("user"),
    COMMAND("command"),
    PROCESSED("processed"),
}

enum class CommandKind(val wire: String) {
    MODE_PHRASE("mode_phrase"),
    TRAILING_INSTRUCTION("trailing_instruction"),
}

enum class RequestStatus(val wire: String) {
    PENDING("pending"),
    INTERRUPTED("interrupted"),
    SETTLED("settled"),
    SUPERSEDED("superseded"),
    CANCELLED("cancelled"),
}

enum class ProposalStatus(val wire: String) {
    CURRENT("current"),
    STALE("stale"),
    SUPERSEDED("superseded"),
    ACCEPTED("accepted"),
    REJECTED("rejected"),
}

/** What an operation did; [wire] is the oracle's outcome string. */
enum class Outcome(val wire: String) {
    APPLIED("applied"),
    UNCHANGED("unchanged"),
    IGNORED_PINNED("ignored_pinned"),
    IGNORED_FINAL("ignored_final"),
    DUPLICATE("duplicate"),
    ATTEMPT_CONFLICT("attempt_conflict"),
    RECORDED_PINNED("recorded_pinned"),
    RECORDED("recorded"),
    OUT_OF_RANGE("out_of_range"),
    REFUSED_PARTIAL("refused_partial"),
    PENDING("pending"),
    UNKNOWN_REQUEST("unknown_request"),
    NOT_PENDING("not_pending"),
    CANCELLED("cancelled"),
    DISCARDED("discarded"),
    FAILED("failed"),
    SUPERSEDED("superseded"),
    CURRENT("current"),
    STALE("stale"),
    NOT_OPEN("not_open"),
    STALE_REJECTED("stale_rejected"),
    REJECTED("rejected"),
    DELIVERED("delivered"),
    KEEP("keep"),
    SWAP("swap"),
    DELETED("deleted"),
    RESTARTED("restarted"),
    IGNORED_DELETED("ignored_deleted"),
}

/** A half-open span of code points `[start, end)` over the draft text. */
data class Span(val start: Int, val end: Int)

/** One region of the draft, as the snapshot shows it. */
data class Region(
    val kind: RegionKind,
    val text: String,
    val span: Span,
    val segment: Int?,
    val attemptId: String?,
    val command: CommandKind?,
    val requestId: String?,
)

/** An immutable raw recognition attempt. */
data class Attempt(val attemptId: String, val segment: Int, val text: String)

data class Request(
    val requestId: String,
    val baseRevision: Int,
    val retryOf: String?,
    val input: String,
    val instruction: String?,
    val status: RequestStatus,
)

data class Proposal(
    val requestId: String,
    val baseRevision: Int,
    val text: String,
    val status: ProposalStatus,
)

data class Delivery(val deliveryId: String, val revision: Int, val targetDigest: String)

/** The internal shape of a region: its text and the metadata that is kept with it. */
private data class Piece(
    val kind: RegionKind,
    val text: String,
    val segment: Int? = null,
    val attemptId: String? = null,
    val command: CommandKind? = null,
    val requestId: String? = null,
)

private class RequestEntry(
    val requestId: String,
    val baseRevision: Int,
    val retryOf: String?,
    val input: String,
    val instruction: String?,
) {
    var status: RequestStatus = RequestStatus.PENDING
}

private class ProposalEntry(val requestId: String, val baseRevision: Int, val text: String) {
    var life: ProposalLife = ProposalLife.OPEN
}

/** A proposal's lifecycle; the public [ProposalStatus] derives current or stale from it while open. */
private enum class ProposalLife { OPEN, SUPERSEDED, ACCEPTED, REJECTED }

// Code-point helpers: the contract counts code points, Kotlin indexes UTF-16.

private fun cpLen(text: String): Int = text.codePointCount(0, text.length)

/** Code points `[from, to)` of [text], clamped like Python slicing. */
private fun cpSlice(text: String, from: Int, to: Int): String {
    val count = cpLen(text)
    val a = from.coerceIn(0, count)
    val b = to.coerceIn(0, count)
    if (a >= b) return ""
    val start = text.offsetByCodePoints(0, a)
    return text.substring(start, text.offsetByCodePoints(start, b - a))
}
