package dev.starling.mobile

import android.os.Bundle
import android.os.Handler
import android.text.InputType
import android.view.KeyEvent
import android.view.inputmethod.CompletionInfo
import android.view.inputmethod.CorrectionInfo
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.ExtractedText
import android.view.inputmethod.ExtractedTextRequest
import android.view.inputmethod.InputConnection
import android.view.inputmethod.InputContentInfo
import dev.starling.mobile.processing.InsertionBoundary.Change
import dev.starling.mobile.ui.BoundaryDelivery
import dev.starling.mobile.ui.EditorField
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** #341 in the keyboard: the boundary rules against a field's InputConnection. */
class BoundaryDeliveryTest {
    /**
     * An editor holding [text] with the cursor at [cursor] (or the selection
     * [cursor]..[selectionEnd]). Reads and commits are counted; any other
     * call fails the test.
     */
    private class FakeConnection(
        var text: String,
        var cursor: Int = text.length,
        var selectionEnd: Int = cursor,
        /** An editor that does not share its text answers null. */
        val readable: Boolean = true,
        val accepts: Boolean = true,
        /** An editor that caps its answers at this many characters. */
        val readLimit: Int = Int.MAX_VALUE,
    ) : InputConnection {
        var reads = 0
        val commits = mutableListOf<String>()

        override fun getTextBeforeCursor(n: Int, flags: Int): CharSequence? {
            reads++
            return if (readable) text.substring(maxOf(0, cursor - minOf(n, readLimit)), cursor) else null
        }

        // No rule reads past the boundary, so the text after the cursor is never asked for.
        override fun getTextAfterCursor(n: Int, flags: Int): CharSequence = unexpected()

        override fun commitText(text: CharSequence, newCursorPosition: Int): Boolean {
            commits += text.toString()
            if (!accepts) return false
            this.text = this.text.substring(0, cursor) + text + this.text.substring(selectionEnd)
            cursor += text.length
            selectionEnd = cursor
            return true
        }

        private fun unexpected(): Nothing = throw AssertionError("unexpected InputConnection call")
        override fun getSelectedText(flags: Int): CharSequence = unexpected()
        override fun getCursorCapsMode(reqModes: Int): Int = unexpected()
        override fun getExtractedText(request: ExtractedTextRequest?, flags: Int): ExtractedText = unexpected()
        override fun deleteSurroundingText(beforeLength: Int, afterLength: Int): Boolean = unexpected()
        override fun deleteSurroundingTextInCodePoints(beforeLength: Int, afterLength: Int): Boolean = unexpected()
        override fun setComposingText(text: CharSequence?, newCursorPosition: Int): Boolean = unexpected()
        override fun setComposingRegion(start: Int, end: Int): Boolean = unexpected()
        override fun finishComposingText(): Boolean = unexpected()
        override fun commitCompletion(text: CompletionInfo?): Boolean = unexpected()
        override fun commitCorrection(correctionInfo: CorrectionInfo?): Boolean = unexpected()
        override fun setSelection(start: Int, end: Int): Boolean = unexpected()
        override fun performEditorAction(editorAction: Int): Boolean = unexpected()
        override fun performContextMenuAction(id: Int): Boolean = unexpected()
        override fun beginBatchEdit(): Boolean = unexpected()
        override fun endBatchEdit(): Boolean = unexpected()
        override fun sendKeyEvent(event: KeyEvent?): Boolean = unexpected()
        override fun clearMetaKeyStates(states: Int): Boolean = unexpected()
        override fun reportFullscreenMode(enabled: Boolean): Boolean = unexpected()
        override fun performPrivateCommand(action: String?, data: Bundle?): Boolean = unexpected()
        override fun requestCursorUpdates(cursorUpdateMode: Int): Boolean = unexpected()
        override fun getHandler(): Handler = unexpected()
        override fun closeConnection() = unexpected()
        override fun commitContent(inputContentInfo: InputContentInfo, flags: Int, opts: Bundle?): Boolean = unexpected()
    }

    private fun field(
        inputType: Int = InputType.TYPE_CLASS_TEXT,
        imeOptions: Int = 0,
        hint: String? = "Message",
    ) = EditorField("com.example.chat", 7, null, inputType, imeOptions, hint)

    private fun deliver(
        connection: FakeConnection,
        raw: String,
        field: EditorField = field(),
        verbatim: Boolean = false,
    ): BoundaryDelivery.Result = BoundaryDelivery.deliver(connection, field, raw, verbatim)

    /** Delivers [raw] after [before] and returns what the field then holds. */
    private fun fieldAfter(before: String, raw: String, after: String = ""): String {
        val connection = FakeConnection(before + after, cursor = before.length)
        val result = deliver(connection, raw)
        assertTrue(result.committed)
        assertEquals("one commit per delivery", 1, connection.commits.size)
        return connection.text
    }

    @Test
    fun startOfFieldKeepsTheTextAsDictated() {
        val connection = FakeConnection("")
        val result = deliver(connection, "Hello there")
        assertEquals("Hello there", connection.text)
        assertEquals(emptyList<Change>(), result.changes)
        assertNull(result.skipped)
    }

    @Test
    fun midSentenceAddsTheSpaceAndLowercases() {
        val connection = FakeConnection("The quick brown")
        val result = deliver(connection, "Fox jumps")
        assertEquals("The quick brown fox jumps", connection.text)
        assertEquals(listOf(Change.LEADING_SPACE, Change.FIRST_LETTER_CASE), result.changes)
        assertEquals(" fox jumps", result.text)
    }

    @Test
    fun theCursorInTheMiddleOfTheFieldIsTheBoundary() {
        assertEquals("Hello big world", fieldAfter("Hello", "Big", after = " world"))
    }

    @Test
    fun sentenceEndersKeepTheCapital() {
        assertEquals("Done. Next one", fieldAfter("Done.", "Next one"))
        assertEquals("Really? Yes", fieldAfter("Really?", "Yes"))
        assertEquals("Wow! Great", fieldAfter("Wow!", "Great"))
        assertEquals("Done. Next", fieldAfter("Done. ", "Next"))
    }

    @Test
    fun aNewLineIsAFieldStart() {
        assertEquals("line one\nSecond line", fieldAfter("line one\n", "Second line"))
    }

    @Test
    fun openingQuotesAndBracketsTakeNoSpace() {
        assertEquals("He said \"hello", fieldAfter("He said \"", "hello"))
        assertEquals("(see note", fieldAfter("(", "see note"))
        assertEquals("Er sagte „hallo", fieldAfter("Er sagte „", "hallo"))
    }

    @Test
    fun existingPunctuationAfterTheCursorIsNotTouched() {
        assertEquals("Note this, please", fieldAfter("Note", "This", after = ", please"))
    }

    @Test
    fun codeIdentifiersKeepTheirCase() {
        assertEquals("Use MyClass here", fieldAfter("Use", "MyClass here"))
        assertEquals("Take Snake_case token", fieldAfter("Take", "Snake_case token"))
        assertEquals("See https://Example.com", fieldAfter("See", "https://Example.com"))
        assertEquals("and I said", fieldAfter("and", "I said"))
    }

    @Test
    fun anEmptyFieldWithAHintIsAFieldStart() {
        // The connection reports the content, not the placeholder.
        val connection = FakeConnection("")
        val result = deliver(connection, "Fox jumps", field(hint = "Type a message,"))
        assertEquals("Fox jumps", connection.commits.single())
        assertEquals(emptyList<Change>(), result.changes)
    }

    @Test
    fun textThatEqualsTheHintIsRealText() {
        val connection = FakeConnection("Message")
        deliver(connection, "Next", field(hint = "Message"))
        assertEquals("Message next", connection.text)
    }

    @Test
    fun passwordFieldsAreNeverRead() {
        listOf(
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD,
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD,
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD,
            InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_VARIATION_PASSWORD,
        ).forEach { type ->
            val connection = FakeConnection("secret")
            val result = deliver(connection, "Word", field(inputType = type))
            assertEquals("type $type", 0, connection.reads)
            assertEquals("Word", connection.commits.single())
            assertEquals(BoundaryDelivery.Skip.PROTECTED, result.skipped)
            assertNull(BoundaryDelivery.context(connection, field(inputType = type)))
            assertEquals(0, connection.reads)
        }
    }

    @Test
    fun incognitoFieldsAreNeverRead() {
        val connection = FakeConnection("private words")
        val result = deliver(connection, "Next", field(imeOptions = EditorInfo.IME_FLAG_NO_PERSONALIZED_LEARNING))
        assertEquals(0, connection.reads)
        assertEquals("Next", connection.commits.single())
        assertEquals(BoundaryDelivery.Skip.PROTECTED, result.skipped)
    }

    @Test
    fun verbatimDeliversUnchangedWithoutReading() {
        val connection = FakeConnection("The quick brown")
        val result = deliver(connection, "Fox", verbatim = true)
        assertEquals(0, connection.reads)
        assertEquals("Fox", connection.commits.single())
        assertEquals(BoundaryDelivery.Skip.VERBATIM, result.skipped)
    }

    @Test
    fun aFieldThatDoesNotShareItsTextGetsTheTextAsDictated() {
        val connection = FakeConnection("The quick brown", readable = false)
        val result = deliver(connection, "Fox")
        assertEquals("Fox", connection.commits.single())
        assertEquals(BoundaryDelivery.Skip.UNREADABLE, result.skipped)
    }

    @Test
    fun aSelectionIsReplacedWithTheBoundaryOfItsStart() {
        val connection = FakeConnection("Hello WORLD today", cursor = 5, selectionEnd = 11)
        deliver(connection, "There")
        assertEquals("Hello there today", connection.text)
    }

    @Test
    fun aRefusedCommitIsReported() {
        val connection = FakeConnection("word", accepts = false)
        assertFalse(deliver(connection, "Next").committed)
        assertEquals(1, connection.commits.size)
    }

    @Test
    fun anEmptyTextWithoutARegionWritesNothing() {
        // An empty commit would replace the user's selection.
        val connection = FakeConnection("Hello WORLD today", cursor = 6, selectionEnd = 11)
        val result = deliver(connection, "")
        assertTrue(result.committed)
        assertEquals(0, connection.reads)
        assertTrue(connection.commits.isEmpty())
        assertEquals("Hello WORLD today", connection.text)
        assertNull(result.skipped)
    }

    @Test
    fun anEmptyTextReplacesTheTakesOwnRegionWithoutReading() {
        val connection = FakeConnection("word next")
        val result = BoundaryDelivery.deliver(connection, field(), "", verbatim = false, composing = " next")
        assertTrue(result.committed)
        assertEquals(0, connection.reads)
        assertEquals("", connection.commits.single())
        assertNull(result.skipped)
    }

    @Test
    fun anEmptyTextWritesNothingOnceTheRegionIsGone() {
        // The editor dropped the region and the user selected "WORLD": an
        // empty commit would delete it.
        val connection = FakeConnection("Hello WORLD today", cursor = 6, selectionEnd = 11)
        val result = BoundaryDelivery.deliver(
            connection,
            field(imeOptions = EditorInfo.IME_FLAG_NO_PERSONALIZED_LEARNING),
            "",
            verbatim = false,
            composing = " next",
            anchored = BoundaryDelivery.emptyCommitSafe(6, 11, -1, -1),
        )
        assertTrue(result.committed)
        assertTrue(connection.commits.isEmpty())
        assertEquals("Hello WORLD today", connection.text)
    }

    @Test
    fun anEmptyCommitIsSafeOnlyWithoutASelectionOrAtTheRegionEnd() {
        assertTrue(BoundaryDelivery.emptyCommitSafe(9, 9, 4, 9))
        assertTrue(BoundaryDelivery.emptyCommitSafe(9, 9, -1, -1))
        assertFalse(BoundaryDelivery.emptyCommitSafe(6, 11, -1, -1))
        assertFalse(BoundaryDelivery.emptyCommitSafe(15, 15, 4, 9))
    }

    @Test
    fun theTakesOwnComposingRegionIsNotTheBoundary() {
        // The live text " fox jum" is composing at the cursor; the boundary is
        // where it starts.
        val connection = FakeConnection("The quick brown fox jum")
        val context = BoundaryDelivery.context(connection, field(), composing = " fox jum")!!
        assertEquals("The quick brown", context.before)
        val result = BoundaryDelivery.deliver(connection, field(), "Fox jumps", verbatim = false, composing = " fox jum")
        assertEquals(" fox jumps", result.text)
    }

    @Test
    fun aCursorThatLeftTheComposingRegionLeavesTheBoundaryUnknown() {
        val connection = FakeConnection("The quick brown fox jum and more", cursor = 32)
        val result = BoundaryDelivery.deliver(connection, field(), "Fox jumps", verbatim = false, composing = " fox jum")
        assertEquals("Fox jumps", result.text)
        assertEquals(BoundaryDelivery.Skip.CURSOR_MOVED, result.skipped)
    }

    @Test
    fun aReadCutShortOfTheRegionIsUnreadableNotAMovedCursor() {
        // The editor caps its answer below the composing text's own length.
        val connection = FakeConnection("The quick brown fox jum", readLimit = 4)
        val result = BoundaryDelivery.deliver(connection, field(), "Fox jumps", verbatim = false, composing = " fox jum")
        assertEquals("Fox jumps", result.text)
        assertEquals(BoundaryDelivery.Skip.UNREADABLE, result.skipped)
    }

    @Test
    fun theFieldIsReReadAtEachDelivery() {
        // The field changed between the live start and the final.
        val connection = FakeConnection("Done.")
        val live = BoundaryDelivery.context(connection, field())!!
        assertEquals(" Next", BoundaryDelivery.adjust("Next", live, verbatim = false))
        connection.text = "Done, and"
        connection.cursor = connection.text.length
        connection.selectionEnd = connection.cursor
        assertEquals(" next", deliver(connection, "Next").text)
    }

    @Test
    fun onlyAWindowAroundTheCursorIsRead() {
        val long = "x".repeat(10_000) + " end"
        val connection = FakeConnection(long, cursor = long.length)
        val context = BoundaryDelivery.context(connection, field())!!
        assertEquals(BoundaryDelivery.WINDOW, context.before.length)
    }

    @Test
    fun aLostComposingAnchorDeliversUnchangedWithoutReading() {
        // "word next. next": the take composes the first " next", the cursor
        // sits after the second; the suffix alone would match.
        val connection = FakeConnection("word next. next")
        val result = BoundaryDelivery.deliver(
            connection,
            field(),
            "Next",
            verbatim = false,
            composing = " next",
            anchored = BoundaryDelivery.cursorAtComposingEnd(15, 15, 4, 9),
        )
        assertEquals(0, connection.reads)
        assertEquals("Next", result.text)
        assertEquals(BoundaryDelivery.Skip.CURSOR_MOVED, result.skipped)
    }

    @Test
    fun theCursorAtTheComposingEndIsAnchored() {
        assertTrue(BoundaryDelivery.cursorAtComposingEnd(9, 9, 4, 9))
        assertFalse(BoundaryDelivery.cursorAtComposingEnd(15, 15, 4, 9))
        assertFalse(BoundaryDelivery.cursorAtComposingEnd(4, 9, 4, 9))
        // No region reported: the suffix check decides.
        assertTrue(BoundaryDelivery.cursorAtComposingEnd(15, 15, -1, -1))
    }
}
