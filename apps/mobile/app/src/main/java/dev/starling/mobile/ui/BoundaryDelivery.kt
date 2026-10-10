package dev.starling.mobile.ui

import android.view.inputmethod.InputConnection
import dev.starling.mobile.processing.InsertionBoundary

/**
 * One delivery of dictated text into the focused field with the
 * insertion-boundary rules (#341): the leading space and the case of the
 * first letter follow the text before the cursor, read from the field right
 * before the single `commitText`.
 *
 * Only what the rules need is read: no v1 rule looks past the insertion
 * point, so the text after the cursor is never requested. What is read
 * serves this decision only; nothing here keeps it, logs it or hands it on. Private fields ([EditorField.sensitive]:
 * passwords, incognito) are never read, and verbatim modes skip the read:
 * both get their text unchanged, as does a field that does not report its
 * text.
 */
object BoundaryDelivery {
    /** Characters read before the boundary; the rules only need the end of that text. */
    const val WINDOW = 128

    /** Why the rules were not applied. */
    enum class Skip {
        /** A password or incognito field: its text is not read. */
        PROTECTED,

        /** A verbatim mode: no rule applies. */
        VERBATIM,

        /**
         * The field did not report the text before the cursor (it answered
         * null, or cut its answer short of the take's own composing text).
         */
        UNREADABLE,

        /**
         * The cursor left the take's composing region, so the text before it
         * no longer ends where the region starts. The field read fine.
         */
        CURSOR_MOVED,
    }

    data class Result(
        /** Whether the editor accepted the commit. */
        val committed: Boolean,
        /** The text handed to `commitText`. */
        val text: String,
        val changes: List<InsertionBoundary.Change>,
        /** Set when the rules could not run; [text] is then the input. */
        val skipped: Skip?,
    )

    /** The boundary as read, or why it is unknown. */
    private sealed interface Read {
        data class Known(val context: InsertionBoundary.Context) : Read

        data class Unknown(val reason: Skip) : Read
    }

    /**
     * The text before the boundary, or null when it is unknown (see
     * [Skip.UNREADABLE], [Skip.CURSOR_MOVED]). [composing] is the take's own
     * composing text, with the cursor at its end: the boundary is where that
     * region starts, so it is cut from the text before the cursor.
     *
     * A field showing only its hint needs no detection here: an
     * InputConnection reports the editor's content, never its placeholder,
     * so such a field reads as empty. (Comparing the text with
     * `EditorInfo.hintText` would misread a field the user filled with the
     * hint's words.)
     */
    fun context(connection: InputConnection, field: EditorField, composing: String? = null): InsertionBoundary.Context? {
        if (field.sensitive) return null
        return (read(connection, composing.orEmpty()) as? Read.Known)?.context
    }

    private fun read(connection: InputConnection, own: String): Read {
        val beforeCursor = connection.getTextBeforeCursor(WINDOW + own.length, 0)?.toString()
            ?: return Read.Unknown(Skip.UNREADABLE)
        if (beforeCursor.endsWith(own)) {
            // No rule reads the text after the boundary; it is not requested.
            return Read.Known(InsertionBoundary.Context(beforeCursor.dropLast(own.length), after = ""))
        }
        // An answer shorter than the region itself was cut by the editor; a
        // full-length one that does not end with it means the cursor moved.
        return Read.Unknown(if (beforeCursor.length < own.length) Skip.UNREADABLE else Skip.CURSOR_MOVED)
    }

    /**
     * Reads the field, applies the rules to [raw] and commits the result
     * once. [composing] as in [context]; the commit replaces that region.
     */
    fun deliver(
        connection: InputConnection,
        field: EditorField,
        raw: String,
        verbatim: Boolean,
        composing: String? = null,
        /** False when the boundary is known to be lost (see [cursorAtComposingEnd]): nothing is read. */
        anchored: Boolean = true,
    ): Result {
        // An empty text has no boundary; nothing is read for it. It is only
        // written to replace the take's own composing region, and only while
        // that region is still known to be there ([anchored], from
        // [emptyCommitSafe]): otherwise an empty commit would replace the
        // user's selection, and nothing is written.
        if (raw.isEmpty()) {
            val committed = composing.isNullOrEmpty() || !anchored || connection.commitText(raw, 1)
            return Result(committed, raw, emptyList(), null)
        }
        val read = when {
            field.sensitive -> Read.Unknown(Skip.PROTECTED)
            verbatim -> Read.Unknown(Skip.VERBATIM)
            !anchored -> Read.Unknown(Skip.CURSOR_MOVED)
            else -> read(connection, composing.orEmpty())
        }
        val adjusted = (read as? Read.Known)?.let { InsertionBoundary.adjust(raw, it.context, verbatim = false) }
            ?: InsertionBoundary.Adjustment(raw, emptyList())
        val committed = connection.commitText(adjusted.text, 1)
        return Result(committed, adjusted.text, adjusted.changes, (read as? Read.Unknown)?.reason)
    }

    /**
     * Whether the cursor still sits at the end of the composing region, from
     * the editor's last `onUpdateSelection`: only then does the text before
     * the cursor end with the region. An editor that reports no region
     * (`-1`) leaves it to [context]'s suffix check.
     */
    fun cursorAtComposingEnd(selStart: Int, selEnd: Int, candidatesStart: Int, candidatesEnd: Int): Boolean =
        candidatesStart < 0 || candidatesEnd < 0 || (selStart == selEnd && selEnd == candidatesEnd)

    /**
     * Whether an empty commit could only clear the take's composing region:
     * the editor reports the region with the cursor at its end, or reports
     * none and nothing is selected. With a selection and no region (the
     * editor dropped it), the empty commit would delete the selected text.
     */
    fun emptyCommitSafe(selStart: Int, selEnd: Int, candidatesStart: Int, candidatesEnd: Int): Boolean =
        selStart == selEnd && cursorAtComposingEnd(selStart, selEnd, candidatesStart, candidatesEnd)

    /** The rules applied to [raw] without writing anything, for live composing text. */
    fun adjust(raw: String, context: InsertionBoundary.Context?, verbatim: Boolean): String =
        if (context == null) raw else InsertionBoundary.adjust(raw, context, verbatim).text
}
