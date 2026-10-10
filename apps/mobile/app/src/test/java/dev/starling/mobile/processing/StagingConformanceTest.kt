package dev.starling.mobile.processing

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.Parameterized
import java.nio.charset.StandardCharsets

/**
 * Replays every case of the shared `staging.json` corpus through [StagedDraft],
 * one JUnit case per fixture case. Each op's outcome and text, the invariants
 * after every op, and the case's closing `expect` block are checked against
 * the oracle in `tests/staging.py`; a case fails once, listing every mismatch.
 */
@RunWith(Parameterized::class)
class StagingConformanceTest(
    private val caseName: String,
    private val case: JSONObject,
) {
    @Test
    fun replaysLikeTheOracle() {
        val mismatches = replay(caseName, case)
        assertTrue(
            "$caseName: ${mismatches.size} mismatch(es)\n" + mismatches.joinToString("\n"),
            mismatches.isEmpty(),
        )
    }

    companion object {
        @JvmStatic
        @Parameterized.Parameters(name = "{0}")
        fun cases(): List<Array<Any>> {
            val corpus = JSONArray(Contracts.fixture("staging.json"))
            return (0 until corpus.length()).map { i ->
                val case = corpus.getJSONObject(i)
                arrayOf<Any>(case.getString("name"), case)
            }
        }
    }
}

/**
 * Offsets are code points, not UTF-16 units. An astral character is two units
 * and must count once; an edit must never land between its surrogates.
 */
class StagedDraftCodePointTest {
    // "a😀b𝄞c": five code points, seven UTF-16 units.
    private val text = "a$GRIN" + "b$CLEF" + "c"

    private fun draftWithText(): StagedDraft = StagedDraft("draft-1", "cap-1").also {
        assertEquals(Outcome.APPLIED, it.final(0, "att-1", text))
    }

    @Test
    fun insertCountsCodePoints() {
        val draft = draftWithText()
        // Seven UTF-16 units would accept 6; five code points do not.
        assertEquals(Outcome.OUT_OF_RANGE, draft.insert(6, "!"))
        assertEquals(Outcome.APPLIED, draft.insert(3, "["))
        assertEquals("a$GRIN" + "b[" + CLEF + "c", draft.text())
        assertEquals(listOf(Span(0, 3), Span(3, 4), Span(4, 6)), draft.regions().map { it.span })
        assertEquals(Outcome.APPLIED, draft.insert(6, "!"))
        assertEquals("a$GRIN" + "b[" + CLEF + "c!", draft.text())
    }

    @Test
    fun deleteCountsCodePoints() {
        val draft = draftWithText()
        assertEquals(Outcome.OUT_OF_RANGE, draft.delete(3, 6))
        assertEquals(Outcome.APPLIED, draft.delete(1, 2)) // exactly the grin
        assertEquals("ab$CLEF" + "c", draft.text())
        assertEquals(Outcome.APPLIED, draft.delete(2, 3)) // exactly the clef
        assertEquals("abc", draft.text())
    }

    @Test
    fun markCommandCountsCodePoints() {
        val draft = draftWithText()
        assertEquals(Outcome.OUT_OF_RANGE, draft.markCommand(3, 6, CommandKind.TRAILING_INSTRUCTION))
        assertEquals(Outcome.APPLIED, draft.markCommand(3, 5, CommandKind.TRAILING_INSTRUCTION))
        assertEquals("a$GRIN" + "b", draft.payloadText())
        assertEquals(CLEF + "c", draft.instruction())
        assertEquals(Span(3, 5), draft.regions().last().span)
        assertEquals(Outcome.PENDING, draft.request("r1"))
        val request = draft.requests().single()
        assertEquals("a$GRIN" + "b", request.input)
        assertEquals(CLEF + "c", request.instruction)
    }

    @Test
    fun revertRawRebuildsTheAstralRecognitionByteForByte() {
        val draft = draftWithText()
        draft.insert(2, "x")
        draft.delete(3, 4)
        assertEquals(Outcome.APPLIED, draft.revertRaw())
        assertEquals(text, draft.text())
        assertArrayEquals(text.toByteArray(StandardCharsets.UTF_8), draft.rawText().toByteArray(StandardCharsets.UTF_8))
    }
}

private const val GRIN = "😀" // U+1F600: one code point, two UTF-16 units
private const val CLEF = "𝄞" // U+1D11E: one code point, two UTF-16 units

/** One `final` op as the oracle's `finals` list keeps it. */
private class FedFinal(val attemptId: String, val segment: Int, val text: String)

/** Replays one case; returns every mismatch, each prefixed with the case name and step. */
private fun replay(name: String, case: JSONObject): List<String> {
    val draft = StagedDraft(case.optString("draft_id", "draft-1"), case.optString("capture_id", "cap-1"))
    val ops = case.getJSONArray("ops")
    val finals = mutableListOf<FedFinal>()
    val mismatches = mutableListOf<String>()
    for (step in 0 until ops.length()) {
        val op = ops.getJSONObject(step)
        val kind = op.getString("op")
        val where = "$name step $step ($kind)"
        if (kind == "final") {
            finals += FedFinal(op.getString("attempt_id"), op.getInt("segment"), op.getString("text"))
        }
        val outcome = dispatch(draft, op)
        if (op.has("expect") && op.getString("expect") != outcome.wire) {
            mismatches += "$where: expected ${q(op.getString("expect"))}, got ${q(outcome.wire)}"
        }
        if (op.has("expect_text") && op.getString("expect_text") != draft.text()) {
            mismatches += "$where: expected text ${q(op.getString("expect_text"))}, got ${q(draft.text())}"
        }
        invariantViolations(draft, finals).forEach { mismatches += "$where: $it" }
    }
    mismatches += expectationMismatches(draft, case.getJSONObject("expect")).map { "$name: $it" }
    return mismatches
}

private fun dispatch(draft: StagedDraft, op: JSONObject): Outcome = when (val kind = op.getString("op")) {
    "partial" -> draft.partial(op.getInt("segment"), op.getString("text"))
    "final" -> draft.final(op.getInt("segment"), op.getString("attempt_id"), op.getString("text"))
    "insert" -> draft.insert(op.getInt("at"), op.getString("text"))
    "delete" -> draft.delete(op.getInt("start"), op.getInt("end"))
    "mark_command" -> draft.markCommand(
        op.getInt("start"),
        op.getInt("end"),
        CommandKind.entries.single { it.wire == op.getString("command") },
    )
    "request" -> draft.request(op.getString("request_id"), op.stringOrNull("retry_of"))
    "cancel" -> draft.cancel(op.getString("request_id"))
    "result" -> draft.result(
        op.getString("request_id"),
        completed = op.getString("status") == "completed",
        text = op.stringOrNull("text"),
    )
    "accept" -> draft.accept(op.getString("request_id"), force = op.optBoolean("force", false))
    "reject" -> draft.reject(op.getString("request_id"))
    "revert_raw" -> draft.revertRaw()
    "deliver" -> draft.deliver(op.getString("delivery_id"), op.getString("target_digest"))
    "swap_check" -> draft.swapCheck(op.getString("request_id"), op.getString("target_digest"))
    "delete_draft" -> draft.deleteDraft()
    "crash" -> draft.crash()
    else -> throw IllegalArgumentException("unknown staging op $kind")
}

/** The oracle's `invariant_violations`, checked after every op. */
private fun invariantViolations(draft: StagedDraft, finals: List<FedFinal>): List<String> {
    val found = mutableListOf<String>()
    var pos = 0
    for (region in draft.regions()) {
        if (region.span.start != pos || region.span.end <= region.span.start) {
            found += "regions do not tile the text at $region"
        }
        pos = region.span.end
    }
    val text = draft.text()
    if (pos != text.codePointCount(0, text.length)) found += "regions do not cover the text"

    // The first occurrence of each attempt id is the recognition it came from.
    val seen = HashMap<String, Pair<Int, String>>()
    for (fed in finals) seen.putIfAbsent(fed.attemptId, fed.segment to fed.text)
    for (attempt in draft.attempts()) {
        if (seen[attempt.attemptId] != (attempt.segment to attempt.text)) {
            found += "attempt ${attempt.attemptId} is not the recognition it came from"
        }
    }

    val recorded = draft.attempts().map { it.attemptId }.toSet()
    val latest = sortedMapOf<Int, String>()
    for (id in finals.map { it.attemptId }.distinct()) {
        if (id in recorded) {
            val (segment, recognized) = seen.getValue(id)
            latest[segment] = recognized
        }
    }
    val expectedRaw = latest.values.joinToString("")
    if (!draft.rawText().toByteArray(StandardCharsets.UTF_8)
            .contentEquals(expectedRaw.toByteArray(StandardCharsets.UTF_8))
    ) {
        found += "raw text does not round-trip byte for byte"
    }
    for (proposal in draft.proposals()) {
        if (proposal.status == ProposalStatus.CURRENT && proposal.baseRevision != draft.revision) {
            found += "proposal ${proposal.requestId} current on an old base"
        }
    }
    return found
}

/** The oracle's `expectation_mismatches`: the case's closing `expect` block. */
private fun expectationMismatches(draft: StagedDraft, expect: JSONObject): List<String> {
    val found = mutableListOf<String>()
    if (expect.has("text") && expect.getString("text") != draft.text()) {
        found += "text: expected ${q(expect.getString("text"))}, got ${q(draft.text())}"
    }
    if (expect.has("revision") && expect.getInt("revision") != draft.revision) {
        found += "revision: expected ${expect.getInt("revision")}, got ${draft.revision}"
    }
    if (expect.has("deleted") && expect.getBoolean("deleted") != draft.deleted) {
        found += "deleted: expected ${expect.getBoolean("deleted")}, got ${draft.deleted}"
    }
    if (expect.has("pinned_segments")) {
        val want = expect.getJSONArray("pinned_segments").ints()
        val got = draft.pinnedSegments()
        if (want != got) found += "pinned_segments: expected $want, got $got"
    }
    if (expect.has("raw_text") && expect.getString("raw_text") != draft.rawText()) {
        found += "raw_text: expected ${q(expect.getString("raw_text"))}, got ${q(draft.rawText())}"
    }
    if (expect.has("regions")) {
        val want = expect.getJSONArray("regions").pairs()
        val got = draft.regions().map { listOf(it.kind.wire, it.text) }
        if (want != got) found += "regions: expected $want, got $got"
    }
    if (expect.has("requests")) {
        val want = expect.getJSONObject("requests").strings()
        val got = draft.requests().associate { it.requestId to it.status.wire }
        if (want != got) found += "requests: expected $want, got $got"
    }
    if (expect.has("proposals")) {
        val want = expect.getJSONObject("proposals").strings()
        val got = draft.proposals().associate { it.requestId to it.status.wire }
        if (want != got) found += "proposals: expected $want, got $got"
    }
    if (expect.has("request_inputs")) {
        val expected = expect.getJSONObject("request_inputs")
        val want = expected.keys().asSequence().associateWith { key ->
            val pair = expected.getJSONArray(key)
            listOf(pair.stringOrNull(0), pair.stringOrNull(1))
        }
        val got = draft.requests().associate { it.requestId to listOf(it.input, it.instruction) }
        if (want != got) found += "request_inputs: expected $want, got $got"
    }
    return found
}

private fun q(value: String): String = JSONObject.quote(value)

private fun JSONObject.stringOrNull(key: String): String? =
    if (has(key) && !isNull(key)) getString(key) else null

private fun JSONObject.strings(): Map<String, String> =
    keys().asSequence().associateWith { getString(it) }

private fun JSONArray.stringOrNull(index: Int): String? = if (isNull(index)) null else getString(index)

private fun JSONArray.ints(): List<Int> = (0 until length()).map { getInt(it) }

private fun JSONArray.pairs(): List<List<String>> = (0 until length()).map { i ->
    val pair = getJSONArray(i)
    listOf(pair.getString(0), pair.getString(1))
}
